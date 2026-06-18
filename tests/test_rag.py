"""Tests for the core RAG flow (retrieve + grounded, cited answer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.answer import GroundingStatus
from app.models.conversation import ConversationTurn
from app.services.base import EmbeddingService, EmbeddingsDisabledError, LLMService
from app.services.null import NullLLMService
from app.services.rag import (
    INSUFFICIENT_CONTEXT_ANSWER,
    NO_ANSWER,
    RagService,
    build_prompt,
)
from app.services.vector_store import VectorStore


class FakeEmbeddingService(EmbeddingService):
    """Deterministic (cat-count, rocket-count) vectors — no real model calls."""

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_keyword_vector(text) for text in texts]

    async def is_reachable(self) -> bool:
        return True


def _keyword_vector(text: str) -> list[float]:
    lower = text.lower()
    return [float(lower.count("cat")), float(lower.count("rocket"))]


class ConstantEmbeddingService(EmbeddingService):
    """Returns the same fixed vector for every text.

    Gives full control over similarity scores, for tests that need to
    engineer a specific ranking (e.g. one document with many top-scoring
    chunks vs. several with one each) rather than rely on keyword counting.
    """

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def model_name(self) -> str:
        return "fake-constant-embedder"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    async def is_reachable(self) -> bool:
        return True


class FakeLLMService(LLMService):
    """Records the prompt/system it was called with and returns a canned answer."""

    def __init__(self, response: str = "Cats are great companions [1].") -> None:
        self._response = response
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return "fake-llm"

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system = system
        return self._response

    async def is_reachable(self) -> bool:
        return True


class SequencedLLMService(LLMService):
    """Returns responses from a list, one per call, in order.

    Needed for flows that call ``generate()`` more than once per request
    (query reformulation, then answer generation) where each call must see a
    different canned response.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return "fake-llm-sequenced"

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self._responses.pop(0)

    async def is_reachable(self) -> bool:
        return True


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / "chroma_db")
    store.add_chunks(
        document_id="doc-cat",
        chunks=[_chunk("Cats are wonderful furry companions.")],
        embeddings=[_keyword_vector("Cats are wonderful furry companions.")],
        metadata={"filename": "cats.txt"},
    )
    store.add_chunks(
        document_id="doc-rocket",
        chunks=[_chunk("Rockets launch into orbit using powerful engines.")],
        embeddings=[_keyword_vector("Rockets launch into orbit using powerful engines.")],
        metadata={"filename": "rockets.txt"},
    )
    return store


def _chunk(text: str, index: int = 0):
    from app.models.chunk import Chunk

    return Chunk(text=text, index=index, char_start=0, char_end=len(text))


def _rag(vector_store: VectorStore, llm: LLMService) -> RagService:
    return RagService(
        embedding_service=FakeEmbeddingService(),
        llm_service=llm,
        vector_store=vector_store,
        top_k=4,
    )


# --- retrieval ranking -----------------------------------------------------


async def test_retrieve_ranks_relevant_chunk_first(vector_store: VectorStore) -> None:
    rag = _rag(vector_store, FakeLLMService())
    chunks = await rag.retrieve("Tell me about cats")

    assert len(chunks) == 2
    assert chunks[0].filename == "cats.txt"
    assert chunks[0].similarity > chunks[1].similarity


async def test_retrieve_raises_when_embeddings_disabled(
    vector_store: VectorStore,
) -> None:
    rag = RagService(
        embedding_service=None,
        llm_service=FakeLLMService(),
        vector_store=vector_store,
        top_k=4,
    )
    with pytest.raises(EmbeddingsDisabledError):
        await rag.retrieve("anything")


# --- diversity-aware retrieval across many documents ------------------------


async def test_retrieve_diversifies_across_many_documents(tmp_path: Path) -> None:
    """One large, uniformly top-scoring document shouldn't crowd out the rest."""
    store = VectorStore(tmp_path / "diverse_db")
    store.add_chunks(
        document_id="doc-a",
        chunks=[_chunk(f"cats chunk {i}", i) for i in range(6)],
        embeddings=[[1.0, 0.0] for _ in range(6)],
        metadata={"filename": "a.txt"},
    )
    for name in ("b", "c", "d"):
        store.add_chunks(
            document_id=f"doc-{name}",
            chunks=[_chunk(f"also about cats, from {name}")],
            embeddings=[[0.9, 0.0]],
            metadata={"filename": f"{name}.txt"},
        )

    rag = RagService(
        embedding_service=ConstantEmbeddingService([1.0, 0.0]),
        llm_service=FakeLLMService(),
        vector_store=store,
        top_k=4,
    )
    chunks = await rag.retrieve("Tell me about cats")

    assert len(chunks) == 4
    represented_docs = {c.document_id for c in chunks}
    assert represented_docs == {"doc-a", "doc-b", "doc-c", "doc-d"}


async def test_retrieve_single_document_scope_is_not_diversified(
    tmp_path: Path,
) -> None:
    """Scoping to one document should return its best chunks, uncapped."""
    store = VectorStore(tmp_path / "single_doc_db")
    store.add_chunks(
        document_id="doc-a",
        chunks=[_chunk(f"cats chunk {i}", i) for i in range(4)],
        embeddings=[[1.0, 0.0] for _ in range(4)],
        metadata={"filename": "a.txt"},
    )

    rag = RagService(
        embedding_service=ConstantEmbeddingService([1.0, 0.0]),
        llm_service=FakeLLMService(),
        vector_store=store,
        top_k=4,
    )
    chunks = await rag.retrieve("Tell me about cats", document_ids=["doc-a"])

    assert len(chunks) == 4
    assert all(c.document_id == "doc-a" for c in chunks)


# --- grounded answer + prompt ----------------------------------------------


async def test_answer_includes_context_in_prompt(vector_store: VectorStore) -> None:
    llm = FakeLLMService()
    rag = _rag(vector_store, llm)
    result = await rag.answer("Tell me about cats")

    assert result.generation_enabled is True
    assert result.answer == "Cats are great companions [1]."
    # The retrieved chunk text must be embedded in the prompt sent to the LLM.
    assert llm.last_prompt is not None
    assert "Cats are wonderful furry companions." in llm.last_prompt
    # The system prompt must enforce grounding + citation + the refusal phrase.
    assert llm.last_system is not None
    assert "ONLY" in llm.last_system
    assert NO_ANSWER in llm.last_system


async def test_answer_maps_citations_back_to_sources(
    vector_store: VectorStore,
) -> None:
    llm = FakeLLMService(response="Cats are great [1]. Also unrelated [9].")
    rag = _rag(vector_store, llm)
    result = await rag.answer("Tell me about cats")

    # [1] is valid and kept; [9] points at no source and is dropped.
    assert result.cited == [1]
    assert result.sources[0].citation == 1
    assert result.sources[0].filename == "cats.txt"


async def test_answer_with_no_citations_reports_empty_cited(
    vector_store: VectorStore,
) -> None:
    llm = FakeLLMService(response="Cats are nice animals but I cite nothing.")
    rag = _rag(vector_store, llm)
    result = await rag.answer("Tell me about cats")

    # An uncited answer still returns sources, but `cited` is empty so the UI
    # can flag it as unverified.
    assert result.cited == []
    assert len(result.sources) >= 1


# --- citation parsing (pure) -----------------------------------------------


def test_extract_citations_keeps_valid_in_first_seen_order_and_dedups() -> None:
    from app.services.rag import _extract_citations

    cited = _extract_citations(
        "First [2], then [1], again [2], and a dangling [9].", valid={1, 2, 3}
    )

    # [2] before [1] (first-seen order), [2] not repeated, [9] dropped (invalid).
    assert cited == [2, 1]


def test_extract_citations_empty_when_none_present() -> None:
    from app.services.rag import _extract_citations

    assert _extract_citations("No markers at all here.", valid={1, 2}) == []


# --- multi-turn: query reformulation ----------------------------------------


async def test_no_reformulation_without_history(vector_store: VectorStore) -> None:
    llm = SequencedLLMService(["Cats are great companions [1]."])
    rag = _rag(vector_store, llm)

    result = await rag.answer("Tell me about cats")

    assert result.standalone_question is None
    # Only the answer call happened — no reformulation call on the first turn.
    assert len(llm.prompts) == 1


async def test_answer_reformulates_followup_using_history(
    vector_store: VectorStore,
) -> None:
    llm = SequencedLLMService(
        ["What is special about cats?", "Cats are wonderful [1]."]
    )
    rag = _rag(vector_store, llm)
    history = [ConversationTurn(question="Tell me about cats", answer="Cats are great [1].")]

    result = await rag.answer("what about them?", history=history)

    assert result.standalone_question == "What is special about cats?"
    # The rewritten question, not the raw follow-up, is what built the final prompt.
    assert "What is special about cats?" in llm.prompts[1]
    assert result.answer == "Cats are wonderful [1]."


async def test_reformulated_query_changes_retrieval_results(
    vector_store: VectorStore,
) -> None:
    """The whole point: a follow-up with no keyword overlap with the right
    document still retrieves it, because reformulation supplies the missing
    context from history."""
    llm = SequencedLLMService(["Tell me about rockets", "Rockets launch into orbit [1]."])
    rag = _rag(vector_store, llm)
    history = [ConversationTurn(question="Tell me about cats", answer="Cats are great [1].")]

    result = await rag.answer("what about the second one?", history=history)

    assert result.sources[0].filename == "rockets.txt"


async def test_reformulation_skipped_when_generation_disabled(
    vector_store: VectorStore,
) -> None:
    rag = RagService(
        embedding_service=FakeEmbeddingService(),
        llm_service=NullLLMService(),
        vector_store=vector_store,
        top_k=4,
    )
    history = [ConversationTurn(question="Tell me about cats", answer=None)]

    result = await rag.answer("what about them?", history=history)

    assert result.standalone_question is None
    assert result.generation_enabled is False


async def test_reformulation_skips_turns_with_no_answer(
    vector_store: VectorStore,
) -> None:
    """A turn asked while generation was disabled has no answer to ground a
    rewrite on; it's excluded from the history handed to reformulation. If
    every turn is like that, reformulation is skipped entirely."""
    llm = SequencedLLMService(["Cats are great companions [1]."])
    rag = _rag(vector_store, llm)
    history = [ConversationTurn(question="Tell me about cats", answer=None)]

    # Follow-up retrieves on its own (mentions cats), so the only LLM call is
    # the answer — proving no separate reformulation call was made.
    result = await rag.answer("anything else about cats?", history=history)

    assert result.standalone_question is None
    assert len(llm.prompts) == 1


def test_build_reformulation_prompt_includes_history_and_followup() -> None:
    from app.services.rag import build_reformulation_prompt

    history = [ConversationTurn(question="Tell me about cats", answer="Cats are great [1].")]
    prompt = build_reformulation_prompt("what about them?", history)

    assert "Tell me about cats" in prompt
    assert "Cats are great [1]." in prompt
    assert "what about them?" in prompt


# --- source mapping: locators carried end to end ---------------------------


async def test_answer_sources_carry_full_text_and_locators(tmp_path: Path) -> None:
    """Each source maps back to the exact passage: page, char range, full text."""
    from app.models.chunk import Chunk

    passage = "Cats are wonderful furry companions that purr."
    store = VectorStore(tmp_path / "locator_db")
    store.add_chunks(
        document_id="doc-cat",
        chunks=[Chunk(text=passage, index=2, char_start=100, char_end=146, page=4)],
        embeddings=[_keyword_vector(passage)],
        metadata={"filename": "cats.pdf"},
    )

    rag = _rag(store, FakeLLMService(response="Cats purr [1]."))
    result = await rag.answer("Tell me about cats")

    [source] = result.sources
    assert source.citation == 1
    assert source.filename == "cats.pdf"
    assert source.page == 4
    assert source.char_start == 100
    assert source.char_end == 146
    assert source.chunk_index == 2
    # Full chunk text is carried verbatim, not truncated.
    assert source.text == passage


# --- grounding honesty signals ---------------------------------------------


async def test_answer_well_grounded_reports_answered_with_high_coverage(
    vector_store: VectorStore,
) -> None:
    llm = FakeLLMService(response="Cats are great companions [1].")
    rag = _rag(vector_store, llm)

    result = await rag.answer("Tell me about cats")

    assert result.status == GroundingStatus.ANSWERED
    assert result.coverage > 0.9  # query vector aligns with the cats chunk
    assert result.low_confidence is False


async def test_answer_insufficient_context_when_top_match_is_weak(
    tmp_path: Path,
) -> None:
    """Best match below the threshold -> decline honestly, don't ask the LLM."""
    store = VectorStore(tmp_path / "weak_db")
    store.add_chunks(
        document_id="doc-1",
        chunks=[_chunk("loosely related material")],
        # Cosine([1,0],[1,3]) ≈ 0.32, below the 0.4 threshold below.
        embeddings=[[1.0, 3.0]],
        metadata={"filename": "weak.txt"},
    )
    llm = FakeLLMService()
    rag = RagService(
        embedding_service=ConstantEmbeddingService([1.0, 0.0]),
        llm_service=llm,
        vector_store=store,
        top_k=4,
        grounding_threshold=0.4,
    )

    result = await rag.answer("a question the docs don't really cover")

    assert result.status == GroundingStatus.INSUFFICIENT_CONTEXT
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.coverage < 0.4
    # The weak source is still surfaced for the user to judge.
    assert len(result.sources) == 1
    assert result.sources[0].filename == "weak.txt"
    # Crucially, the LLM was never asked to answer a weak-match question.
    assert llm.last_prompt is None


async def test_answer_no_documents_status_when_store_empty(tmp_path: Path) -> None:
    empty_store = VectorStore(tmp_path / "empty_db")
    rag = _rag(empty_store, FakeLLMService())

    result = await rag.answer("Tell me about anything")

    assert result.status == GroundingStatus.NO_DOCUMENTS
    assert result.coverage == 0.0
    assert result.sources == []


async def test_answer_low_confidence_when_grounded_answer_cites_nothing(
    vector_store: VectorStore,
) -> None:
    """A well-grounded retrieval whose answer cites no source is flagged."""
    llm = FakeLLMService(response="Cats are wonderful, but I cite nothing here.")
    rag = _rag(vector_store, llm)

    result = await rag.answer("Tell me about cats")

    assert result.status == GroundingStatus.ANSWERED
    assert result.cited == []
    assert result.low_confidence is True


async def test_answer_not_low_confidence_when_citations_present(
    vector_store: VectorStore,
) -> None:
    llm = FakeLLMService(response="Cats are wonderful [1].")
    rag = _rag(vector_store, llm)

    result = await rag.answer("Tell me about cats")

    assert result.cited == [1]
    assert result.low_confidence is False


def test_coverage_clamps_negative_similarity_and_takes_the_best() -> None:
    from app.models.answer import Source
    from app.services.rag import _coverage

    sources = [
        Source(citation=1, document_id="d", filename="f", chunk_index=0,
               text="a", score=-0.2),
        Source(citation=2, document_id="d", filename="f", chunk_index=1,
               text="b", score=0.7),
    ]

    # Best (0.7) wins; the negative score is clamped, not allowed to dominate.
    assert _coverage(sources) == pytest.approx(0.7)


# --- retrieval-only mode (LLM disabled) ------------------------------------


async def test_answer_retrieval_only_when_generation_disabled(
    vector_store: VectorStore,
) -> None:
    rag = RagService(
        embedding_service=FakeEmbeddingService(),
        llm_service=NullLLMService(),
        vector_store=vector_store,
        top_k=4,
    )
    result = await rag.answer("Tell me about cats")

    assert result.generation_enabled is False
    assert result.answer is None
    # Retrieval still works for free: sources are returned.
    assert len(result.sources) == 2
    assert result.sources[0].filename == "cats.txt"


# --- empty retrieval -------------------------------------------------------


async def test_answer_says_dont_know_when_no_context(tmp_path: Path) -> None:
    empty_store = VectorStore(tmp_path / "empty_db")
    llm = FakeLLMService()
    rag = _rag(empty_store, llm)
    result = await rag.answer("Tell me about cats")

    assert result.generation_enabled is True
    assert result.answer == NO_ANSWER
    assert result.sources == []
    # The LLM should not be invoked when there is no context to ground on.
    assert llm.last_prompt is None


async def test_empty_retrieval_with_generation_disabled_is_clean(
    tmp_path: Path,
) -> None:
    empty_store = VectorStore(tmp_path / "empty_db")
    rag = RagService(
        embedding_service=FakeEmbeddingService(),
        llm_service=NullLLMService(),
        vector_store=empty_store,
        top_k=4,
    )
    result = await rag.answer("Tell me about cats")

    assert result.answer is None
    assert result.generation_enabled is False
    assert result.sources == []


# --- prompt builder (pure) -------------------------------------------------


def test_build_prompt_numbers_sources() -> None:
    from app.models.answer import Source

    sources = [
        Source(
            citation=1,
            document_id="doc-a",
            filename="a.txt",
            chunk_index=0,
            text="alpha",
            score=0.9,
        ),
        Source(
            citation=2,
            document_id="doc-a",
            filename="a.txt",
            chunk_index=1,
            text="beta",
            score=0.5,
        ),
    ]
    prompt = build_prompt("What?", sources)

    assert "[1]" in prompt and "alpha" in prompt
    assert "[2]" in prompt and "beta" in prompt
    assert "Question: What?" in prompt


def test_build_prompt_omits_multi_doc_note_for_single_document() -> None:
    from app.models.answer import Source

    sources = [
        Source(
            citation=1,
            document_id="doc-a",
            filename="a.txt",
            chunk_index=0,
            text="alpha",
            score=0.9,
        ),
        Source(
            citation=2,
            document_id="doc-a",
            filename="a.txt",
            chunk_index=1,
            text="beta",
            score=0.5,
        ),
    ]
    prompt = build_prompt("What?", sources)

    assert "multiple different documents" not in prompt


def test_build_prompt_notes_multiple_documents() -> None:
    from app.models.answer import Source

    sources = [
        Source(
            citation=1,
            document_id="doc-a",
            filename="a.txt",
            chunk_index=0,
            text="alpha",
            score=0.9,
        ),
        Source(
            citation=2,
            document_id="doc-b",
            filename="b.txt",
            chunk_index=0,
            text="beta",
            score=0.8,
        ),
    ]
    prompt = build_prompt("What?", sources)

    assert "multiple different documents" in prompt
