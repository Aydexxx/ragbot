# RAGBot Architecture

This document describes how RAGBot is put together and, in depth, the four
subsystems that distinguish it from a generic chatbot: **retrieval**,
**citations**, **conversation**, and **grounding/honesty**.

For setup and the API reference, see the [README](../README.md).

---

## Layering & dependency flow

RAGBot is layered so the core logic never depends on a web framework or a
specific model vendor. Dependencies point inward:

```
app/api  ──▶  app/services (interfaces)  ◀──  app/core (factory)  ──▶  concrete providers
   │                  │
   └─ FastAPI         └─ RagService, ingestion, vector store, conversation store
      routers,           depend only on the abstract EmbeddingService / LLMService
      DI wiring
```

- **`app/services/base.py`** defines the abstract `EmbeddingService` and
  `LLMService`. Everything in the core depends on these, never on Ollama or
  OpenAI directly.
- **`app/core/factory.py`** is the *only* module that constructs concrete
  providers. Selecting a provider is therefore a config-only change; a missing
  chat model yields a `NullLLMService` so the system degrades instead of
  crashing.
- **`app/api/dependencies.py`** wires settings + services into routes via
  FastAPI `Depends`. Process-wide singletons (vector store, document registry,
  conversation store) are built lazily behind locks, because FastAPI runs sync
  dependencies in a thread pool and a first-use race on ChromaDB's client
  corrupts its on-disk tenant metadata.

### Request lifecycle for `POST /ask`

```
AskRequest
  │  (conversation_id?, document_ids?, question)
  ▼
ask route ── mint/lookup conversation_id ──▶ load bounded history
  │
  ▼
RagService.answer(question, document_ids, history)
  ├─ reformulate(question, history)        → standalone question (if follow-up)
  ├─ retrieve(standalone)                  → diversity-aware top-k chunks
  ├─ _coverage(sources)                    → best-source similarity, 0..1
  ├─ grounding gate:
  │     no sources           → no_documents
  │     coverage < threshold → insufficient_context (LLM NOT called)
  │     else                 → answered
  ├─ build_prompt + LLM.generate           → answer text (answered path only)
  └─ _extract_citations                    → validated [n], low_confidence flag
  ▼
AnswerResult ── route appends the turn to history ──▶ response (+ conversation_id)
```

---

## 1. Retrieval subsystem

**Files:** `app/services/ingestion.py`, `app/services/indexer.py`,
`app/services/vector_store.py`, `app/services/rag.py` (`retrieve`, `_diversify`).

### Ingestion → chunking → embedding → storage
1. **Extract.** `extract()` returns the document text plus, for PDFs, a list of
   per-page start offsets (`page_starts`). TXT/MD have no pages → `None`.
2. **Chunk.** `chunk_text()` produces overlapping, word-boundary-aligned chunks,
   each carrying its `char_start`/`char_end` in the original text.
3. **Assign pages.** `assign_pages()` maps each chunk's start offset back to a
   1-based page via binary search over `page_starts` (skipped for unpaginated
   formats, leaving `page = None`).
4. **Embed & store.** The indexer embeds chunks and writes them to ChromaDB with
   metadata: `document_id`, `chunk_index`, `char_start`, `char_end`, and `page`
   (omitted when `None`, since ChromaDB rejects null metadata values). The
   collection uses cosine space, so `similarity = 1 − distance`.

### Diversity-aware retrieval (the multi-document differentiator)
A plain top-_k_ similarity search lets one large or repetitive document fill
every slot, starving other relevant files. `RagService.retrieve` avoids this:

- For a multi-document search it pulls an **expanded candidate pool**
  (`top_k × _POOL_MULTIPLIER`, capped at `_MAX_POOL_SIZE`) instead of just
  `top_k`.
- `_diversify` then **caps chunks per document** at `ceil(top_k / distinct_docs)`
  before taking the global top-_k_ by score, so each represented document gets a
  fair share.
- A search **scoped to a single document** (`document_ids` of length ≤ 1) skips
  the expansion entirely — there's nothing to diversify against.

**Tested by:** `test_retrieve_diversifies_across_many_documents`,
`test_retrieve_single_document_scope_is_not_diversified`,
`test_e2e_cross_document_question_draws_from_multiple_docs`.

---

## 2. Citation subsystem

**Files:** `app/services/rag.py` (`build_prompt`, `_extract_citations`,
`_to_sources`), `app/models/answer.py` (`Source`),
`frontend/src/components/{CitedAnswer,SourceList}.tsx`.

### Producing citations
- `build_prompt` numbers each retrieved passage `[1]`, `[2]`, … and the system
  prompt instructs the model to **answer only from those passages and cite each
  claim** with its bracketed number. A note is added when passages span multiple
  documents, telling the model to cite each under its own number.
- Each `Source` carries everything needed to verify the claim: `citation` index,
  `filename`, `page`, `char_start`/`char_end`, the **full chunk `text`** (not a
  truncated snippet), and the similarity `score`.

### Validating citations (no fabricated sources)
- `_extract_citations` scans the answer for `[n]` markers and keeps only those
  that map to a **real retrieved source**, deduped in first-seen order. A
  hallucinated `[9]` pointing at a non-existent source is **dropped**.
- If a generated answer ends up citing nothing valid, `low_confidence` is set —
  a best-effort signal that the model may have drifted beyond its context.

### Frontend
- `CitedAnswer` renders each valid `[n]` as a clickable badge; clicking it
  scrolls to and flashes (`citation-flash`) the matching source card.
- `SourceList` shows each source as a card with the filename, a **locator**
  (`p. 3 · chunk 4 · chars 1820–2110`), a **confidence bar** colored by score,
  and the full passage with the **most query-relevant sentence(s) highlighted**
  (a round-trip-safe, stopword-filtered token-overlap heuristic — no NLP
  dependency). `idPrefix` namespaces card DOM ids per chat turn so citations in
  different turns don't collide.

**Tested by:** `test_extract_citations_*`,
`test_answer_maps_citations_back_to_sources` (dangling `[9]` dropped),
`test_answer_sources_carry_full_text_and_locators`, `test_build_prompt_*`.

---

## 3. Conversation subsystem

**Files:** `app/services/conversation.py`, `app/models/conversation.py`,
`app/services/rag.py` (`_reformulate`, `build_reformulation_prompt`),
`app/api/ask.py`, `frontend/src/components/{AskPanel,ChatTurn}.tsx`.

### State
- `ConversationStore` is an in-memory, thread-safe map of
  `conversation_id → list[ConversationTurn]`. It is **bounded** to the last
  `MAX_TURNS` (6) turns, capping both memory and the history fed into the
  reformulation prompt (cost control). It is not persisted — history only needs
  to survive the process lifetime to ground a back-and-forth.
- The `/ask` route mints a `conversation_id` on the first turn and returns it;
  the client echoes it back to continue. Omitting it starts fresh — the basis of
  the frontend's **"New conversation"** reset.

### Query reformulation (the follow-up differentiator)
Before retrieval, if there is usable prior context, `_reformulate` asks the LLM
to rewrite the follow-up into a **standalone question** (e.g. *"what about the
second one?"* + history → *"what is the deadline of the second contract?"*).
Retrieval and the final prompt then operate on the rewritten question — the prior
turns are **never** themselves sent to retrieval. The rewrite is returned as
`standalone_question` so the UI can show how the follow-up was interpreted.

Reformulation **degrades gracefully**: it is skipped (falling back to the raw
question) when there's no history, when no prior turn produced an answer to
ground on, or when generation is disabled — since, like answering, it needs the
LLM.

**Tested by:** `test_answer_reformulates_followup_using_history`,
`test_reformulated_query_changes_retrieval_results`,
`test_reformulation_skipped_when_generation_disabled`,
`test_reformulation_skips_turns_with_no_answer`,
`test_ask_followup_reuses_conversation_id_and_reformulates`,
`test_ask_new_conversation_has_no_carried_over_history`, and the
`ConversationStore` unit tests in `test_conversation.py`.

---

## 4. Grounding & honesty subsystem

**Files:** `app/services/rag.py` (`answer`, `_coverage`), `app/config.py`
(`grounding_threshold`), `app/models/answer.py` (`GroundingStatus`,
`AnswerResult`), `frontend/src/components/{ChatTurn,GroundingIndicator}.tsx`.

This is the anti-hallucination guarantee. Every `AnswerResult` reports an honest
grounding signal so a weak match can never be dressed up as a confident answer.

### Status & coverage
- **`coverage`** = the best source's similarity, clamped to `0..1`
  (`_coverage`). It is the confidence signal shown in the UI.
- **`status`** (`GroundingStatus`):
  - `no_documents` — nothing retrieved (none indexed, or none in scope).
  - `insufficient_context` — sources exist but `coverage < grounding_threshold`.
  - `answered` — well-grounded; an answer was produced (or is available in
    retrieval-only mode).

### The two-point hallucination guard
1. **No sources** → return `no_documents` without calling the LLM.
2. **Coverage below threshold** → return `insufficient_context` with a canned,
   honest message (`INSUFFICIENT_CONTEXT_ANSWER`) and the weak sources still
   listed — **the LLM is never asked to answer a weak-match question**, so it
   can't guess.

Only on the `answered` path is the LLM invoked. Even then, an answer that cites
no valid source is flagged `low_confidence`.

`grounding_threshold` is configurable via `GROUNDING_THRESHOLD` in `.env`
(default `0.4`) and should be tuned to the embedding model — lower it if real
questions are wrongly refused, raise it to be stricter.

### Frontend
`ChatTurn` renders a distinct state per status: a normal grounded answer (with a
green `GroundingIndicator` "Grounded in N sources" + coverage bar), an explicit
*"the documents don't clearly answer this — here's the closest related material"*
decline that still shows the weak sources, or a friendly upload prompt for
`no_documents`. A standing explainer makes the trust model obvious: *"RAGBot only
answers from your documents and shows its sources."*

**Tested by:** `test_answer_well_grounded_reports_answered_with_high_coverage`,
`test_answer_insufficient_context_when_top_match_is_weak` (asserts the LLM is not
called), `test_answer_no_documents_status_when_store_empty`,
`test_answer_low_confidence_when_grounded_answer_cites_nothing`,
`test_answer_not_low_confidence_when_citations_present`,
`test_coverage_clamps_negative_similarity_and_takes_the_best`,
`test_ask_well_grounded_answer_reports_status_and_coverage`.

---

## Graceful degradation summary

| Configuration | Embeddings | Generation | `POST /ask` behavior |
| --- | --- | --- | --- |
| `PROVIDER=ollama`/`openai`, chat model set | ✅ | ✅ | Full pipeline: reformulate → retrieve → ground → generate → cite |
| Embeddings only (no chat model) | ✅ | ❌ | **Retrieval-only**: sources + grounding status, `answer = null`; reformulation skipped |
| `PROVIDER=none` | ❌ | ❌ | `503` — retrieval is impossible without embeddings |
| Backend configured but offline | ⚠️ | ⚠️ | `/health` reports `reachable: false`; `/ask` surfaces `502` rather than crashing |

## Testing philosophy

All 119 backend tests run with **every provider faked** — deterministic
keyword/bag-of-words embedders and canned-response LLMs (including a
`SequencedLLMService` for the reformulate-then-answer flow). **No test makes a
real model or network call**, so the suite is fast, hermetic, and reproducible
regardless of the local `.env`.
