"""Core RAG orchestration: retrieve relevant chunks, then ground an answer in them.

Pure orchestration over the embedding, vector-store, and LLM services — no web
framework coupling, so it is equally usable from an API route or a script.
"""

from __future__ import annotations

import math
import re

from app.models.answer import AnswerResult, GroundingStatus, Source
from app.models.chunk import RetrievedChunk
from app.models.conversation import ConversationTurn
from app.services.base import EmbeddingService, EmbeddingsDisabledError, LLMService
from app.services.vector_store import VectorStore

#: Exact phrase the LLM is told to use when the context does not answer the
#: question. Kept as a constant so the empty-retrieval short-circuit returns
#: the identical wording the model is instructed to produce.
NO_ANSWER = "I don't know based on the provided documents."

#: Returned verbatim (instead of asking the LLM) when retrieval found
#: something but the best match is below ``grounding_threshold`` — the system
#: declines to answer rather than ground a reply on a weak match.
INSUFFICIENT_CONTEXT_ANSWER = (
    "The documents don't clearly cover this question. Below is the closest "
    "related material I found — please read it and judge for yourself rather "
    "than treating this as a confident answer."
)

#: Default minimum top-source similarity (0..1) for an answer to count as
#: well-grounded, used when no threshold is supplied (e.g. in unit tests).
#: Production wires ``Settings.grounding_threshold`` through instead.
DEFAULT_GROUNDING_THRESHOLD = 0.4

_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Do not use any outside "
    "knowledge. Cite every claim with the bracketed number of the passage it "
    "comes from, like [1] or [2] — even when passages come from different "
    "documents, cite each one under its own number. If the context does not "
    f'contain enough information to answer, reply exactly: "{NO_ANSWER}" '
    "Never invent facts, sources, or citations."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")

_REFORMULATE_SYSTEM_PROMPT = (
    "You rewrite a user's follow-up question into a standalone question, "
    "using the conversation history for context. Output ONLY the rewritten "
    "question — no explanation, no quotes, no preamble. If the follow-up is "
    "already standalone, return it unchanged."
)

#: How many extra candidates to pull per requested chunk so chunks can be
#: balanced across documents before the final cut (see ``_diversify``).
_POOL_MULTIPLIER = 4
#: Upper bound on the expanded candidate pool, so a broad "all documents"
#: query doesn't pull an unbounded number of chunks just to diversify.
_MAX_POOL_SIZE = 100


class RagService:
    """Answers questions by retrieving indexed chunks and grounding an LLM in them."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None,
        llm_service: LLMService,
        vector_store: VectorStore,
        top_k: int,
        grounding_threshold: float = DEFAULT_GROUNDING_THRESHOLD,
    ) -> None:
        self._embedding_service = embedding_service
        self._llm_service = llm_service
        self._vector_store = vector_store
        self._top_k = top_k
        self._grounding_threshold = grounding_threshold

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed ``question`` and return the most similar chunks, best first.

        When the search spans more than one document, pulls a larger
        candidate pool and caps chunks per document (see ``_diversify``) so a
        single large or highly-relevant document can't crowd out the others.

        Raises:
            EmbeddingsDisabledError: If no embedding provider is configured;
                retrieval is impossible without one.
        """
        if self._embedding_service is None:
            raise EmbeddingsDisabledError(
                "Cannot retrieve: no embedding provider is configured. "
                "Set PROVIDER=ollama or PROVIDER=openai in .env to enable it."
            )

        k = top_k if top_k is not None else self._top_k
        [query_embedding] = await self._embedding_service.embed_texts([question])
        candidates = self._vector_store.query(
            query_embedding=query_embedding,
            top_k=_candidate_pool_size(k, document_ids),
            document_ids=document_ids,
        )
        return _diversify(candidates, top_k=k)

    async def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        history: list[ConversationTurn] | None = None,
    ) -> AnswerResult:
        """Retrieve context and produce a grounded, cited answer.

        When ``history`` is non-empty, ``question`` is first rewritten into a
        standalone question (e.g. "what about the second one?" ->
        "what is the deadline of the second contract?") so retrieval and the
        final prompt both operate on something self-contained — the prior
        turns themselves are never sent to retrieval. The rewrite is
        reported back as ``AnswerResult.standalone_question`` for transparency.

        Every result carries an honest grounding signal (see
        :class:`GroundingStatus`) and a ``coverage`` score from retrieval, so
        the caller can tell a well-supported answer from a weak-match decline.
        The flow guards against hallucination at two points:

        * Nothing retrieved -> ``no_documents``: nothing to draw on.
        * Best match below ``grounding_threshold`` -> ``insufficient_context``:
          the LLM is *not* asked to answer; a canned honest message is returned
          with the weak sources still listed, so a low-relevance match can't be
          dressed up as a confident reply.

        Degrades gracefully when generation is disabled (e.g.
        ``NullLLMService``): the grounding status and sources are still
        computed and returned with ``answer=None`` — retrieval is fully usable
        on its own. Reformulation also needs the LLM, so it's skipped then and
        retrieval falls back to the raw follow-up text.
        """
        standalone_question = await self._reformulate(question, history or [])
        retrieval_question = standalone_question or question

        chunks = await self.retrieve(retrieval_question, document_ids=document_ids)
        sources = _to_sources(chunks)
        coverage = _coverage(sources)
        generation_enabled = self._llm_service.enabled

        if not sources:
            # Nothing to ground on at all.
            return AnswerResult(
                answer=NO_ANSWER if generation_enabled else None,
                sources=sources,
                generation_enabled=generation_enabled,
                status=GroundingStatus.NO_DOCUMENTS,
                coverage=coverage,
                standalone_question=standalone_question,
            )

        if coverage < self._grounding_threshold:
            # Retrieved something, but the best match is too weak to answer
            # from. Decline honestly instead of letting the LLM guess; still
            # surface the closest material for the user to judge.
            return AnswerResult(
                answer=INSUFFICIENT_CONTEXT_ANSWER if generation_enabled else None,
                sources=sources,
                generation_enabled=generation_enabled,
                status=GroundingStatus.INSUFFICIENT_CONTEXT,
                coverage=coverage,
                standalone_question=standalone_question,
            )

        if not generation_enabled:
            # Retrieval-only mode: well-grounded sources, no generated answer.
            return AnswerResult(
                answer=None,
                sources=sources,
                generation_enabled=False,
                status=GroundingStatus.ANSWERED,
                coverage=coverage,
                standalone_question=standalone_question,
            )

        prompt = build_prompt(retrieval_question, sources)
        answer_text = await self._llm_service.generate(
            prompt, system=_SYSTEM_PROMPT
        )
        cited = _extract_citations(answer_text, valid={s.citation for s in sources})

        return AnswerResult(
            answer=answer_text,
            sources=sources,
            generation_enabled=True,
            status=GroundingStatus.ANSWERED,
            coverage=coverage,
            # Best-effort hallucination signal: a well-grounded retrieval whose
            # answer cites nothing valid may have drifted beyond the context.
            low_confidence=len(cited) == 0,
            cited=cited,
            standalone_question=standalone_question,
        )

    async def _reformulate(
        self, question: str, history: list[ConversationTurn]
    ) -> str | None:
        """Rewrite a follow-up into a standalone question, or None to skip.

        Skips (returns None) when there's no prior context to rewrite against,
        when none of the prior turns produced an answer worth grounding on
        (e.g. they were all asked while generation was disabled), or when
        generation is currently disabled — reformulation needs the LLM just
        like answering does, so it degrades the same way: retrieval simply
        uses the raw follow-up text instead.
        """
        if not self._llm_service.enabled:
            return None
        usable_history = [turn for turn in history if turn.answer is not None]
        if not usable_history:
            return None

        prompt = build_reformulation_prompt(question, usable_history)
        rewritten = await self._llm_service.generate(
            prompt, system=_REFORMULATE_SYSTEM_PROMPT
        )
        rewritten = rewritten.strip()
        return rewritten or None


def build_prompt(question: str, sources: list[Source]) -> str:
    """Build the user prompt embedding numbered context passages and the question."""
    context_blocks = "\n\n".join(
        f"[{source.citation}] (from {source.filename})\n{source.text}"
        for source in sources
    )
    multi_doc_note = ""
    if len({source.document_id for source in sources}) > 1:
        multi_doc_note = (
            "Note: the passages above come from multiple different "
            "documents. Synthesize across them to answer fully, and cite "
            "each passage under its own [n] rather than attributing "
            "everything to a single source.\n\n"
        )
    return (
        "Context passages:\n"
        f"{context_blocks}\n\n"
        f"{multi_doc_note}"
        f"Question: {question}\n\n"
        "Answer using only the passages above, citing them with [n]:"
    )


def build_reformulation_prompt(question: str, history: list[ConversationTurn]) -> str:
    """Build the prompt that rewrites a follow-up into a standalone question.

    Assumes every turn in ``history`` has a non-None ``answer`` — callers
    filter out turns asked while generation was disabled before reaching here.
    """
    transcript = "\n".join(
        f"User: {turn.question}\nAssistant: {turn.answer}" for turn in history
    )
    return (
        f"Conversation so far:\n{transcript}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )


def _candidate_pool_size(top_k: int, document_ids: list[str] | None) -> int:
    """Expand the candidate pool so ``_diversify`` has room to balance documents.

    A query scoped to a single document has nothing to diversify against, so
    it skips the expansion and asks the vector store for exactly ``top_k``.
    """
    if document_ids is not None and len(document_ids) <= 1:
        return top_k
    return min(top_k * _POOL_MULTIPLIER, _MAX_POOL_SIZE)


def _diversify(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Cap chunks per document, then take the global top ``top_k`` by score.

    ``chunks`` arrives best-first from the vector store. Without this cap, a
    single large or highly-relevant document can fill every slot in a
    multi-document query, starving the others; each document is limited to
    a fair share of ``top_k`` before the final ranked cut.
    """
    distinct_docs = {chunk.document_id for chunk in chunks}
    if not distinct_docs:
        return []
    cap_per_doc = max(1, math.ceil(top_k / len(distinct_docs)))

    counts: dict[str, int] = {}
    capped: list[RetrievedChunk] = []
    for chunk in chunks:
        count = counts.get(chunk.document_id, 0)
        if count >= cap_per_doc:
            continue
        counts[chunk.document_id] = count + 1
        capped.append(chunk)

    return capped[:top_k]


def _coverage(sources: list[Source]) -> float:
    """How well the best source matches: top similarity, clamped to 0..1.

    Cosine similarity can dip slightly negative for unrelated vectors; clamping
    keeps coverage a clean 0..1 signal for both the threshold check and the UI
    confidence indicator. 0 when nothing was retrieved.
    """
    return max((min(max(s.score, 0.0), 1.0) for s in sources), default=0.0)


def _to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            citation=i + 1,
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            page=chunk.page,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            text=chunk.text,
            score=chunk.similarity,
        )
        for i, chunk in enumerate(chunks)
    ]


def _extract_citations(answer_text: str, valid: set[int]) -> list[int]:
    """Return citation numbers referenced in ``answer_text``, in first-seen order.

    Only numbers that map to a real source are kept, so a hallucinated ``[9]``
    pointing at a non-existent source is dropped.
    """
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer_text):
        n = int(match.group(1))
        if n in valid and n not in seen:
            seen.append(n)
    return seen
