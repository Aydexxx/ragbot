"""Tests for the REST API: upload, list, delete, ask, and error paths.

Uses FastAPI's ``TestClient`` with faked embedding/LLM services injected via
``app.dependency_overrides`` — no real model or network calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_conversation_store,
    get_document_registry,
    get_embedding_service,
    get_llm_service,
    get_vector_store,
)
from app.config import get_settings
from app.main import app
from app.services.base import EmbeddingService, LLMService
from app.services.conversation import ConversationStore
from app.services.null import NullLLMService
from app.services.rag import NO_ANSWER
from app.services.registry import JsonFileDocumentRegistry
from app.services.vector_store import VectorStore

_UNSET = object()


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


class FakeLLMService(LLMService):
    """Returns a canned response instead of calling a real chat model."""

    def __init__(self, response: str = "Cats are great companions [1].") -> None:
        self._response = response

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return "fake-llm"

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return self._response

    async def is_reachable(self) -> bool:
        return True


class SequencedLLMService(LLMService):
    """Returns responses from a list, one per call, in order — for flows
    (like query reformulation + answer generation) that call generate() more
    than once per request."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return "fake-llm-sequenced"

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return self._responses.pop(0)

    async def is_reachable(self) -> bool:
        return True


@pytest.fixture
def make_client(tmp_path: Path):
    """Build a TestClient with the document/vector-store/provider deps faked.

    Pass ``embedding_service=None`` or ``llm_service=NullLLMService()`` to
    simulate those backends being disabled.
    """

    def _make(
        embedding_service: Any = _UNSET,
        llm_service: Any = _UNSET,
        max_upload_size_bytes: int | None = None,
    ) -> TestClient:
        if embedding_service is _UNSET:
            embedding_service = FakeEmbeddingService()
        if llm_service is _UNSET:
            llm_service = FakeLLMService()

        registry = JsonFileDocumentRegistry(tmp_path / "documents.json")
        vector_store = VectorStore(tmp_path / "chroma_db")
        conversation_store = ConversationStore()

        app.dependency_overrides[get_document_registry] = lambda: registry
        app.dependency_overrides[get_vector_store] = lambda: vector_store
        app.dependency_overrides[get_embedding_service] = lambda: embedding_service
        app.dependency_overrides[get_llm_service] = lambda: llm_service
        app.dependency_overrides[get_conversation_store] = lambda: conversation_store

        if max_upload_size_bytes is not None:
            base_settings = get_settings()
            overridden = base_settings.model_copy(
                update={"max_upload_size_bytes": max_upload_size_bytes}
            )
            app.dependency_overrides[get_settings] = lambda: overridden

        return TestClient(app)

    yield _make

    app.dependency_overrides.clear()


def _upload(client: TestClient, filename: str, content: bytes) -> dict[str, Any]:
    resp = client.post(
        "/documents", files={"file": (filename, content, "text/plain")}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- POST /documents ---------------------------------------------------


def test_upload_document_returns_metadata(make_client: Any) -> None:
    client = make_client()
    document = _upload(client, "cats.txt", b"Cats are wonderful furry companions.")

    assert document["filename"] == "cats.txt"
    assert document["status"] == "ready"
    assert document["num_chunks"] == 1


def test_upload_unsupported_file_type_returns_415(make_client: Any) -> None:
    client = make_client()
    resp = client.post(
        "/documents", files={"file": ("archive.zip", b"binary junk", "application/zip")}
    )
    assert resp.status_code == 415


def test_upload_file_too_large_returns_413(make_client: Any) -> None:
    client = make_client(max_upload_size_bytes=10)
    resp = client.post(
        "/documents",
        files={
            "file": (
                "cats.txt",
                b"This text is definitely longer than ten bytes.",
                "text/plain",
            )
        },
    )
    assert resp.status_code == 413
    assert "MB" in resp.json()["detail"]


def test_upload_with_embeddings_disabled_returns_503(make_client: Any) -> None:
    client = make_client(embedding_service=None)
    resp = client.post(
        "/documents",
        files={"file": ("cats.txt", b"Cats are wonderful.", "text/plain")},
    )
    assert resp.status_code == 503


# --- GET /documents ------------------------------------------------------


def test_list_documents_returns_uploaded(make_client: Any) -> None:
    client = make_client()
    document = _upload(client, "cats.txt", b"Cats are wonderful.")

    resp = client.get("/documents")
    assert resp.status_code == 200
    ids = [d["id"] for d in resp.json()]
    assert document["id"] in ids


def test_list_documents_empty_when_none_uploaded(make_client: Any) -> None:
    client = make_client()
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == []


# --- DELETE /documents/{id} ----------------------------------------------


def test_delete_document_removes_it(make_client: Any) -> None:
    client = make_client()
    document = _upload(client, "cats.txt", b"Cats are wonderful.")

    resp = client.delete(f"/documents/{document['id']}")
    assert resp.status_code == 204

    remaining = client.get("/documents").json()
    assert document["id"] not in [d["id"] for d in remaining]


def test_delete_missing_document_returns_404(make_client: Any) -> None:
    client = make_client()
    resp = client.delete(f"/documents/{uuid4()}")
    assert resp.status_code == 404


# --- POST /ask -------------------------------------------------------------


def test_ask_returns_grounded_cited_answer(make_client: Any) -> None:
    llm = FakeLLMService(response="Cats are great companions [1].")
    client = make_client(llm_service=llm)
    _upload(client, "cats.txt", b"Cats are wonderful furry companions.")

    resp = client.post("/ask", json={"question": "Tell me about cats"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Cats are great companions [1]."
    assert body["generation_enabled"] is True
    assert body["sources"][0]["filename"] == "cats.txt"
    assert body["cited"] == [1]


def test_ask_retrieval_only_when_generation_disabled(make_client: Any) -> None:
    client = make_client(llm_service=NullLLMService())
    _upload(client, "cats.txt", b"Cats are wonderful furry companions.")

    resp = client.post("/ask", json={"question": "Tell me about cats"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] is None
    assert body["generation_enabled"] is False
    assert len(body["sources"]) == 1
    assert body["sources"][0]["filename"] == "cats.txt"


def test_ask_without_document_ids_searches_all_documents(make_client: Any) -> None:
    client = make_client()
    cats = _upload(client, "cats.txt", b"Cats are wonderful furry companions.")
    rockets = _upload(
        client, "rockets.txt", b"Rockets launch into orbit using powerful engines."
    )

    resp = client.post(
        "/ask", json={"question": "Tell me about cats and rockets"}
    )
    assert resp.status_code == 200
    body = resp.json()
    document_ids = {source["document_id"] for source in body["sources"]}
    assert document_ids == {cats["id"], rockets["id"]}


def test_ask_with_document_ids_restricts_retrieval(make_client: Any) -> None:
    client = make_client()
    cats = _upload(client, "cats.txt", b"Cats are wonderful furry companions.")
    _upload(client, "rockets.txt", b"Rockets launch into orbit using powerful engines.")

    resp = client.post(
        "/ask",
        json={
            "question": "Tell me about cats and rockets",
            "document_ids": [cats["id"]],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["document_id"] == cats["id"]


def test_ask_with_no_documents_indexed_returns_friendly_200(make_client: Any) -> None:
    client = make_client()
    resp = client.post("/ask", json={"question": "Tell me about cats"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == NO_ANSWER
    assert body["sources"] == []
    assert body["status"] == "no_documents"
    assert body["coverage"] == 0.0


def test_ask_well_grounded_answer_reports_status_and_coverage(make_client: Any) -> None:
    client = make_client()
    _upload(client, "cats.txt", b"Cats are wonderful furry companions.")

    resp = client.post("/ask", json={"question": "Tell me about cats"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["coverage"] > 0.9
    assert body["low_confidence"] is False


def test_ask_with_embeddings_disabled_returns_503(make_client: Any) -> None:
    client = make_client(embedding_service=None)
    resp = client.post("/ask", json={"question": "Tell me about cats"})
    assert resp.status_code == 503


def test_ask_rejects_empty_question(make_client: Any) -> None:
    client = make_client()
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


# --- POST /ask: multi-turn conversations ------------------------------------


def test_ask_without_conversation_id_starts_new_conversation(make_client: Any) -> None:
    client = make_client()
    _upload(client, "cats.txt", b"Cats are wonderful furry companions.")

    resp = client.post("/ask", json={"question": "Tell me about cats"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"]
    # First turn: nothing to rewrite, so no standalone question is reported.
    assert body["standalone_question"] is None


def test_ask_followup_reuses_conversation_id_and_reformulates(make_client: Any) -> None:
    llm = SequencedLLMService(
        [
            "Cats are great companions [1].",
            "What is the lifespan of cats?",
            "Cats typically live 12-18 years [1].",
        ]
    )
    client = make_client(llm_service=llm)
    _upload(client, "cats.txt", b"Cats are wonderful furry companions.")

    first = client.post("/ask", json={"question": "Tell me about cats"}).json()
    conversation_id = first["conversation_id"]

    second = client.post(
        "/ask",
        json={
            "question": "what is their lifespan?",
            "conversation_id": conversation_id,
        },
    ).json()

    assert second["conversation_id"] == conversation_id
    assert second["standalone_question"] == "What is the lifespan of cats?"
    assert second["answer"] == "Cats typically live 12-18 years [1]."


def test_ask_new_conversation_has_no_carried_over_history(make_client: Any) -> None:
    """Omitting conversation_id (the "New conversation" reset) starts fresh:
    no reformulation call happens because there's no history to rewrite from."""
    llm = SequencedLLMService(
        ["Cats are great companions [1].", "Rockets launch into orbit [1]."]
    )
    client = make_client(llm_service=llm)
    _upload(client, "cats.txt", b"Cats are wonderful furry companions.")
    _upload(
        client, "rockets.txt", b"Rockets launch into orbit using powerful engines."
    )

    first = client.post("/ask", json={"question": "Tell me about cats"}).json()
    second = client.post("/ask", json={"question": "Tell me about rockets"}).json()

    assert first["conversation_id"] != second["conversation_id"]
    assert second["standalone_question"] is None
    assert second["sources"][0]["filename"] == "rockets.txt"


# --- CORS --------------------------------------------------------------


def test_cors_allows_react_dev_server_origin(make_client: Any) -> None:
    client = make_client()
    resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
