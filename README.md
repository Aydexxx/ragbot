# RAGBot

[![Tests passing](https://github.com/Aydexxx/ragbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Aydexxx/ragbot/actions/workflows/ci.yml)

**Ask questions about *your* documents and get answers you can actually trust.**
RAGBot is a document Q&A system built on Retrieval-Augmented Generation (RAG):
upload PDF / TXT / Markdown files, then ask questions and get answers grounded in
your sources — with inline citations, an honest "I don't know" when the documents
don't cover something, and follow-up conversation that stays on your corpus.

It **runs for free out of the box** on a local [Ollama](https://ollama.com)
instance — no API key, nothing leaves your machine — and switches to OpenAI with a
single `.env` change.

---

## Why not just use ChatGPT?

A generic chatbot answers from a frozen, opaque training set. It can't read your
contract, your handbook, or last quarter's report, and when it doesn't know
something it often makes up a confident, unsourced answer. RAGBot is built to be
the opposite:

| | Generic chatbot | **RAGBot** |
| --- | --- | --- |
| **Answers from your files** | ❌ Only its training data | ✅ Retrieves from documents *you* upload |
| **Citations** | ❌ None, or fabricated | ✅ Every claim links to the exact passage, page & similarity score |
| **Honesty** | ❌ Guesses confidently | ✅ Says *"the documents don't clearly cover this"* and shows the closest material |
| **Confidence signal** | ❌ Hidden | ✅ A coverage score + grounding badge on every answer |
| **Multi-document reasoning** | ❌ — | ✅ Diversity-aware retrieval pulls from *all* relevant files, not just one |
| **Follow-up questions** | ✅ but ungrounded | ✅ Rewrites "what about the second one?" into a standalone, still-grounded query |
| **Privacy** | ❌ Sent to a vendor | ✅ Local-first: free Ollama path keeps documents on your machine |

These five differentiators — **multi-document retrieval, verifiable citations,
grounding/honesty, conversational follow-ups, and local-first privacy** — are the
whole point. They're each documented in depth in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Differentiators in depth

### 1. Multi-document, diversity-aware retrieval
A naïve top-_k_ search lets one large or highly-repetitive document monopolize
every result slot. RAGBot expands the candidate pool and **caps chunks per
document** before the final ranking, so a cross-cutting question draws from *all*
the relevant files. The answer then shows **"Answer drawn from N documents: …"**
so you can see its breadth at a glance.

### 2. Verifiable citations
The model is constrained to cite each claim with `[n]`. Those markers are parsed,
**validated against the real retrieved chunks, and any dangling/hallucinated
citation is dropped**. In the UI each `[n]` is a clickable badge that scrolls to
and highlights a source card showing the full passage, its filename, **page
number** (for PDFs), character range, and a **similarity confidence bar** — the
exact provenance a chat paste can never give you.

### 3. Grounding & honesty (the anti-hallucination guarantee)
Every answer carries a `status` — `answered`, `insufficient_context`, or
`no_documents` — plus a `coverage` score derived from retrieval similarity. When
the best match falls below a configurable threshold, RAGBot **does not ask the LLM
to guess**: it returns an explicit *"the documents don't clearly cover this"* with
the closest related passages for you to judge. Answers that cite nothing are
flagged low-confidence. The "I don't know" path is a **feature**, not a failure.

### 4. Conversational follow-ups
Ask "what about the second one?" and RAGBot uses the conversation history to
**rewrite it into a standalone question** ("what is the deadline of the second
contract?") *before* retrieval — so follow-ups stay grounded and cited. History is
bounded for cost control, and the rewritten query is shown for transparency.

### 5. Local-first & provider-agnostic
Embeddings and generation sit behind abstract interfaces. The default is a **free,
local Ollama** instance (your documents never leave your machine); switching to
OpenAI is one config value, no code changes.

---

## Architecture

```
                  ┌──────────────────────── ingestion pipeline ────────────────────────┐
  upload  ──▶   extract text  ──▶  chunk  ──▶  embed  ──▶  store (vector DB)
  PDF/TXT/MD      pypdf / utf-8     overlap     Ollama /     ChromaDB (cosine)
                  + page offsets    windows     OpenAI       + page / char metadata
                                                                          │
  ════════════════════════════ query path ═══════════════════════════════╪═════════
                                                                          ▼
  question ─▶ reformulate ─▶ embed query ─▶ retrieve top-k ─▶ diversify ─▶ generate ─▶ cite
  POST /ask   (if follow-up,  Ollama/OpenAI   ChromaDB         cap per      grounded   parse &
              using history)                  similarity       document     answer     validate [n]
                                                  │                                       │
                                       coverage < threshold?                              ▼
                                       ├─ yes ─▶ insufficient_context (decline, show closest)
                                       └─ no  ─▶ answered  ──────────▶  AnswerResult + sources + status
                                       (no sources at all ─▶ no_documents)

  retrieval-only (no chat model configured): everything above runs except `generate`;
  the API returns sources with answer = null instead of crashing.
```

Layering keeps providers swappable and the core logic free of web-framework or
vendor coupling:

| Layer          | Responsibility |
| -------------- | -------------- |
| `app/api`      | FastAPI routers, dependency wiring, error → HTTP mapping |
| `app/core`     | Provider **factory** — the *only* place that knows concrete providers |
| `app/services` | `EmbeddingService` / `LLMService` interfaces + Ollama/OpenAI/null impls; ingestion, vector store, document registry, RAG orchestration, conversation store |
| `app/models`   | Pydantic request/response schemas |
| `app/config`   | `pydantic-settings` configuration loaded from `.env` |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the retrieval, citation,
conversation, and grounding subsystems in detail.

## Tech stack

| Area             | Technology |
| ---------------- | ---------- |
| Backend          | Python 3.11+, [FastAPI](https://fastapi.tiangolo.com), Uvicorn |
| Validation       | Pydantic v2 + pydantic-settings |
| Vector store     | [ChromaDB](https://www.trychroma.com) (local, persistent, cosine) |
| Embeddings / LLM | [Ollama](https://ollama.com) (default, free) or [OpenAI](https://platform.openai.com) |
| Ingestion        | pypdf (PDF, with page offsets), native text for TXT/MD |
| HTTP client      | httpx (async) |
| Frontend         | React + TypeScript, [Vite](https://vitejs.dev), Tailwind CSS |
| Testing          | pytest + pytest-asyncio (119 tests, all providers faked) |

---

## Screenshots

> _Placeholder images — replace the files in `docs/screenshots/` with real
> captures._

| Multi-document query | Clickable citations |
| --- | --- |
| ![Multi-document query — an answer drawn from several files with the "Answer drawn from N documents" attribution line](docs/screenshots/multi-doc-query.png) | ![Clickable `[n]` citations linking to source cards with passage, page, and confidence bar](docs/screenshots/citations.png) |

| Insufficient-context honesty | Conversation thread |
| --- | --- |
| ![The honest "the documents don't clearly answer this" state showing the closest related material](docs/screenshots/insufficient-context.png) | ![A multi-turn chat thread with a follow-up rewritten into a standalone question](docs/screenshots/chat-thread.png) |

---

## Local setup

### Prerequisites
- Python 3.11+ and `pip`
- Node.js 18+ and `npm` (for the frontend)
- Optional: [Ollama](https://ollama.com/download) to run for free (see below)

### Backend

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (defaults to the free Ollama provider)
cp .env.example .env

# 4. Run the API
uvicorn app.main:app --reload
```

API on <http://localhost:8000>; interactive docs at <http://localhost:8000/docs>;
status at <http://localhost:8000/health>.

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, expects the backend on :8000
```

Set `VITE_API_BASE_URL` (see [`frontend/.env.example`](frontend/.env.example)) if
the backend isn't on `localhost:8000`. CORS is pre-configured for the Vite dev
server.

### Tests

```bash
pytest                         # backend: 119 tests, all providers faked — no real calls, no network

cd frontend
npx tsc -b && npx eslint src   # frontend: typecheck + lint
npm run build                  # production build
```

---

## Run it free with Ollama

This is the default. No API key, no billing — everything runs locally and your
documents never leave your machine.

```bash
# 1. Install Ollama:  https://ollama.com/download

# 2. Pull an embedding model (required for upload/retrieval)
ollama pull nomic-embed-text

# 3. Pull a chat model (required only for generated answers; optional —
#    without it RAGBot runs in retrieval-only mode, see below)
ollama pull llama3.2

# 4. Make sure Ollama is running, then start the backend
ollama serve            # if not already running as a service
uvicorn app.main:app --reload
```

The defaults in `.env.example` already point at these models:

```
PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3.2
```

If Ollama isn't running yet, `/health` reports `reachable: false` — the API never
crashes, it just can't embed/generate until the backend is up.

## Switch to OpenAI

Edit `.env`:

```
PROVIDER=openai
OPENAI_API_KEY=sk-...                 # never commit a real key
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

No code changes are needed — the provider factory does the rest.

> **💸 Cost note.** OpenAI is a **paid, metered** API: every upload (embeddings)
> and every `/ask` (embeddings + chat completion) bills your account. **Set a hard
> spend limit** in your OpenAI dashboard (*Settings → Limits → Usage limits*)
> before enabling this provider. The free Ollama path has none of these concerns.

## Retrieval-only mode (graceful degradation)

RAGBot separates **retrieval** (embed + similarity search) from **generation** (LLM
answer). When **embeddings are configured but no chat model is** — e.g.
`PROVIDER=ollama` with `OLLAMA_CHAT_MODEL=` empty — RAGBot enters
**retrieval-only mode**:

- Upload, indexing, retrieval, multi-doc diversity, and grounding status all work.
- `POST /ask` returns the matching **source passages** with `answer: null` and
  `generation_enabled: false` — never an error. Query reformulation is skipped
  gracefully (it needs the LLM), falling back to the raw question.
- `/health` reports `generation_enabled: false` accurately; the frontend shows a
  "retrieval only" badge and renders the passages.

If embeddings themselves aren't configured (`PROVIDER=none`), upload and `/ask`
return `503` with a clear message — retrieval is impossible without embeddings.

---

## API reference

Base URL: `http://localhost:8000`

| Method   | Path               | Body / Params | Description |
| -------- | ------------------ | ------------- | ----------- |
| `GET`    | `/health`          | — | Provider config + capability flags + backend reachability + upload limits |
| `POST`   | `/documents`       | `multipart/form-data` `file` (PDF/TXT/MD) | Upload & index a document synchronously → `201` with `Document` |
| `GET`    | `/documents`       | — | List indexed documents |
| `DELETE` | `/documents/{id}`  | path `id` (UUID) | Remove a document and its chunks → `204` |
| `POST`   | `/ask`             | `AskRequest` (below) | Retrieve and answer → `AnswerResult` |
| `GET`    | `/`                | — | Service info + link to docs |

**`AskRequest`** (`POST /ask`):

```json
{
  "question": "What is the deadline of the second contract?",
  "document_ids": ["<uuid>", "..."],          // optional: restrict scope (all if omitted)
  "conversation_id": "<uuid>"                  // optional: continue a conversation
}
```

**`AnswerResult`**:

```json
{
  "answer": "The deadline is March 1st [1].",
  "status": "answered",                        // answered | insufficient_context | no_documents
  "coverage": 0.82,                            // best-source similarity, 0..1 (the confidence signal)
  "low_confidence": false,                     // true if an answer cites no valid source
  "cited": [1],                                // citation numbers actually referenced
  "sources": [
    {
      "citation": 1,
      "document_id": "<uuid>",
      "filename": "contracts.pdf",
      "chunk_index": 4,
      "page": 3,                               // 1-based page for PDFs; null for TXT/MD
      "char_start": 1820,
      "char_end": 2110,
      "text": "…the second contract's deadline is March 1st…",
      "score": 0.82
    }
  ],
  "generation_enabled": true,
  "conversation_id": "<uuid>",                 // echo back to continue the conversation
  "standalone_question": "What is the deadline of the second contract?"  // the rewritten follow-up, or null
}
```

- **`insufficient_context`** → `answer` explicitly states the documents don't cover
  the question; the weak `sources` are still returned for inspection.
- **`no_documents`** → nothing was retrieved (none indexed, or none in scope).
- **Retrieval-only** → `answer` is `null`, `generation_enabled` is `false`,
  `sources` still populated.

### Status codes

| Code  | When |
| ----- | ---- |
| `200` | Successful `/ask`, `/health`, `/documents` listing |
| `201` | Document uploaded & indexed |
| `204` | Document deleted |
| `413` | Upload exceeds `MAX_UPLOAD_SIZE_BYTES` |
| `415` | Unsupported file type (not PDF/TXT/MD) |
| `422` | Invalid request (e.g. empty question) |
| `502` | A configured provider is unreachable / errored |
| `503` | Embeddings are disabled (`PROVIDER=none`) |

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example)).

| Variable | Values / default | Notes |
| -------- | ---------------- | ----- |
| `PROVIDER` | `none` · `ollama` *(default)* · `openai` | Active backend |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embeddings |
| `OLLAMA_CHAT_MODEL` | `llama3.2` (empty → retrieval-only) | Generation |
| `OPENAI_API_KEY` | *(empty)* | Required for `openai` |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embeddings |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Generation |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking (characters) |
| `TOP_K` | `4` | Chunks retrieved per query |
| `GROUNDING_THRESHOLD` | `0.4` | Min top-source similarity to answer; below it → `insufficient_context` |
| `CHROMA_DIR` / `UPLOAD_DIR` | `chroma_db` / `uploads` | Local storage |
| `MAX_UPLOAD_SIZE_BYTES` | `20971520` (20 MB) | Per-file upload limit |
| `MAX_FILES_PER_REQUEST` | `20` | Cap on files accepted in one upload batch (enforced client-side via `/health`'s `limits`) |

## License

[MIT](LICENSE) © Bünyamin Aydeniz
