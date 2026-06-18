"""Tests for the document indexing pipeline (ingest -> chunk -> embed -> store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.document import DocumentStatus
from app.services.base import EmbeddingService, EmbeddingsDisabledError
from app.services.indexer import DocumentIndexer
from app.services.ingestion import UnsupportedFileTypeError
from app.services.registry import JsonFileDocumentRegistry
from app.services.vector_store import VectorStore


class FakeEmbeddingService(EmbeddingService):
    """Deterministic, model-free stand-in for a real embedding backend.

    Maps text to a 2-D bag-of-keywords vector (cat-count, rocket-count) so
    semantically distinct inputs land far apart and similar inputs land
    close together under cosine distance, without calling any real model.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_keyword_vector(text) for text in texts]

    async def is_reachable(self) -> bool:
        return True


def _keyword_vector(text: str) -> list[float]:
    lower = text.lower()
    return [float(lower.count("cat")), float(lower.count("rocket"))]


@pytest.fixture
def registry(tmp_path: Path) -> JsonFileDocumentRegistry:
    return JsonFileDocumentRegistry(tmp_path / "documents.json")


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma_db")


def _indexer(
    registry: JsonFileDocumentRegistry,
    vector_store: VectorStore,
    embedding_service: EmbeddingService | None,
) -> DocumentIndexer:
    return DocumentIndexer(
        registry=registry,
        vector_store=vector_store,
        embedding_service=embedding_service,
        chunk_size=1000,
        chunk_overlap=0,
    )


async def test_index_document_marks_ready_with_chunk_count(
    registry: JsonFileDocumentRegistry, vector_store: VectorStore
) -> None:
    indexer = _indexer(registry, vector_store, FakeEmbeddingService())
    document = await indexer.index_document(
        b"Cats are wonderful furry companions.", "cats.txt"
    )

    assert document.status == DocumentStatus.READY
    assert document.num_chunks == 1
    assert registry.get(document.id) == document


async def test_index_document_batches_embedding_calls(
    registry: JsonFileDocumentRegistry, vector_store: VectorStore
) -> None:
    embedder = FakeEmbeddingService()
    indexer = DocumentIndexer(
        registry=registry,
        vector_store=vector_store,
        embedding_service=embedder,
        chunk_size=100,
        chunk_overlap=20,
    )

    text = "\n\n".join(
        f"Paragraph number {i} about cats and rockets." for i in range(20)
    )
    document = await indexer.index_document(text.encode("utf-8"), "many.txt")

    assert document.num_chunks > 1
    assert len(embedder.calls) == 1  # one batched call, not one per chunk
    assert len(embedder.calls[0]) == document.num_chunks


async def test_index_document_raises_when_embeddings_disabled(
    registry: JsonFileDocumentRegistry, vector_store: VectorStore
) -> None:
    indexer = _indexer(registry, vector_store, None)
    with pytest.raises(EmbeddingsDisabledError):
        await indexer.index_document(b"hello", "notes.txt")


async def test_index_document_marks_failed_on_extraction_error(
    registry: JsonFileDocumentRegistry, vector_store: VectorStore
) -> None:
    indexer = _indexer(registry, vector_store, FakeEmbeddingService())
    with pytest.raises(UnsupportedFileTypeError):
        await indexer.index_document(b"binary", "archive.zip")

    [document] = registry.list_all()
    assert document.status == DocumentStatus.FAILED


async def test_index_then_query_returns_most_similar_chunk_first(
    registry: JsonFileDocumentRegistry, vector_store: VectorStore
) -> None:
    embedder = FakeEmbeddingService()
    indexer = _indexer(registry, vector_store, embedder)

    cat_doc = await indexer.index_document(
        b"Cats are wonderful furry companions.", "cats.txt"
    )
    rocket_doc = await indexer.index_document(
        b"Rockets launch into orbit using powerful engines.", "rockets.txt"
    )

    [query_vector] = await embedder.embed_texts(["Tell me about cats"])
    results = vector_store.query(query_vector, top_k=2)

    assert len(results) == 2
    assert results[0].document_id == str(cat_doc.id)
    assert results[0].filename == "cats.txt"
    assert results[1].document_id == str(rocket_doc.id)


async def test_delete_document_removes_its_chunks(
    registry: JsonFileDocumentRegistry, vector_store: VectorStore
) -> None:
    embedder = FakeEmbeddingService()
    indexer = _indexer(registry, vector_store, embedder)

    cat_doc = await indexer.index_document(
        b"Cats are wonderful furry companions.", "cats.txt"
    )
    rocket_doc = await indexer.index_document(
        b"Rockets launch into orbit using powerful engines.", "rockets.txt"
    )

    vector_store.delete_document(cat_doc.id)

    [query_vector] = await embedder.embed_texts(["Tell me about cats"])
    results = vector_store.query(query_vector, top_k=10)

    assert len(results) == 1
    assert results[0].document_id == str(rocket_doc.id)
