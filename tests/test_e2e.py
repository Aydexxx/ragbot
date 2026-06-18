"""End-to-end test through the public REST API.

Indexes known text, asks a question whose answer is in it, and asserts the
right chunk surfaces as the top source — exercising the full
ingest -> chunk -> embed -> store -> retrieve (-> generate) pipeline across the
HTTP boundary. Both paths are proven from the same fixture data:

* generation enabled (faked LLM) -> a grounded, cited answer, and
* retrieval-only (no LLM)         -> sources only, ``answer=None``.

No real model or network call is made: the embedding service is a deterministic
bag-of-words fake and the LLM is a canned-response fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_registry,
    get_embedding_service,
    get_llm_service,
    get_vector_store,
)
from app.main import app
from app.services.base import EmbeddingService, LLMService
from app.services.null import NullLLMService
from app.services.registry import JsonFileDocumentRegistry
from app.services.vector_store import VectorStore

# A tiny fixed vocabulary. Each document/question is embedded as the count of
# these words it contains, so a question lands closest to the fact that answers
# it under cosine distance — a faithful, fully deterministic stand-in for a
# real semantic embedder.
_VOCAB = (
    "capital france paris tallest mountain everest "
    "largest ocean pacific photosynthesis plants sunlight"
).split()

_FACTS = {
    "geography.txt": "The capital of France is Paris.",
    "mountains.txt": "The tallest mountain on Earth is Mount Everest.",
    "oceans.txt": "The largest ocean is the Pacific Ocean.",
    "biology.txt": "Photosynthesis lets plants turn sunlight into energy.",
}


def _vec(text: str) -> list[float]:
    low = text.lower()
    return [float(low.count(word)) for word in _VOCAB]


class VocabEmbeddingService(EmbeddingService):
    """Bag-of-words embedder over a fixed vocabulary; no real model calls."""

    @property
    def model_name(self) -> str:
        return "fake-vocab-embedder"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_vec(text) for text in texts]

    async def is_reachable(self) -> bool:
        return True


class GroundedFakeLLM(LLMService):
    """Echoes the top context passage with a [1] citation.

    Proves grounding: the answer it returns is built from the prompt it was
    handed, so a correct answer is only possible if the right passage was
    retrieved and placed into the prompt.
    """

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return "fake-grounded-llm"

    async def generate(self, prompt: str, system: str | None = None) -> str:
        # The first numbered passage is the top-ranked source.
        marker = "[1] (from "
        start = prompt.index(marker)
        passage_start = prompt.index("\n", start) + 1
        passage_end = prompt.index("\n\n", passage_start)
        passage = prompt[passage_start:passage_end].strip()
        return f"{passage} [1]"

    async def is_reachable(self) -> bool:
        return True


@pytest.fixture
def make_client(tmp_path: Path):
    def _make(llm_service: LLMService) -> TestClient:
        registry = JsonFileDocumentRegistry(tmp_path / "documents.json")
        vector_store = VectorStore(tmp_path / "chroma_db")

        app.dependency_overrides[get_document_registry] = lambda: registry
        app.dependency_overrides[get_vector_store] = lambda: vector_store
        app.dependency_overrides[get_embedding_service] = VocabEmbeddingService
        app.dependency_overrides[get_llm_service] = lambda: llm_service
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def _index_all_facts(client: TestClient) -> None:
    for filename, text in _FACTS.items():
        resp = client.post(
            "/documents", files={"file": (filename, text.encode(), "text/plain")}
        )
        assert resp.status_code == 201, resp.text


def test_e2e_generation_enabled_returns_grounded_cited_answer(
    make_client: Any,
) -> None:
    client = make_client(GroundedFakeLLM())
    _index_all_facts(client)

    resp = client.post("/ask", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()

    # The right chunk was retrieved: top source is the France fact.
    assert body["sources"][0]["filename"] == "geography.txt"
    assert "Paris" in body["sources"][0]["text"]
    # The answer is grounded in that chunk and cites it.
    assert body["generation_enabled"] is True
    assert "Paris" in body["answer"]
    assert body["cited"] == [1]


def test_e2e_retrieval_only_returns_right_chunk_without_answer(
    make_client: Any,
) -> None:
    client = make_client(NullLLMService())
    _index_all_facts(client)

    resp = client.post(
        "/ask", json={"question": "Which is the tallest mountain?"}
    )
    assert resp.status_code == 200
    body = resp.json()

    # Retrieval still works for free with no LLM configured.
    assert body["generation_enabled"] is False
    assert body["answer"] is None
    assert body["sources"][0]["filename"] == "mountains.txt"
    assert "Everest" in body["sources"][0]["text"]


def test_e2e_cross_document_question_draws_from_multiple_docs(
    make_client: Any,
) -> None:
    """Index several docs, two of which speak to the same topic, then ask one
    cross-cutting question: diversity-aware retrieval should surface the
    relevant passage from BOTH topical docs (not just one), and the answer is
    grounded and cited across them."""
    client = make_client(GroundedFakeLLM())
    docs = {
        "ocean_size.txt": "The largest ocean is the Pacific Ocean.",
        "ocean_facts.txt": "The Pacific Ocean is the largest ocean on Earth.",
        "mountains.txt": "The tallest mountain on Earth is Mount Everest.",
    }
    for filename, text in docs.items():
        resp = client.post(
            "/documents", files={"file": (filename, text.encode(), "text/plain")}
        )
        assert resp.status_code == 201, resp.text

    resp = client.post("/ask", json={"question": "What is the largest ocean?"})
    assert resp.status_code == 200
    body = resp.json()

    # The two best-matching sources span BOTH ocean documents — the cross-doc
    # question is grounded across files, not crowded out by a single one.
    top_two = {body["sources"][0]["filename"], body["sources"][1]["filename"]}
    assert top_two == {"ocean_size.txt", "ocean_facts.txt"}
    assert body["sources"][0]["score"] > body["sources"][-1]["score"]
    assert body["status"] == "answered"
    # Sources are contiguously numbered for citation/attribution.
    citations = [source["citation"] for source in body["sources"]]
    assert citations == list(range(1, len(citations) + 1))


def test_e2e_different_questions_retrieve_different_chunks(
    make_client: Any,
) -> None:
    client = make_client(NullLLMService())
    _index_all_facts(client)

    cases = {
        "How do plants make energy?": "biology.txt",
        "What is the largest ocean?": "oceans.txt",
    }
    for question, expected_file in cases.items():
        resp = client.post("/ask", json={"question": question})
        assert resp.status_code == 200
        assert resp.json()["sources"][0]["filename"] == expected_file
