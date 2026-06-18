"""Tests for the ChromaDB-backed vector store."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.chunk import Chunk
from app.services.vector_store import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma_db")


def _chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, index=index, char_start=0, char_end=len(text))


def test_get_or_create_collection_is_idempotent(store: VectorStore) -> None:
    first = store.get_or_create_collection()
    second = store.get_or_create_collection()
    assert first is second


def test_add_chunks_rejects_mismatched_lengths(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.add_chunks(
            document_id="doc-1",
            chunks=[_chunk("a"), _chunk("b", 1)],
            embeddings=[[1.0, 0.0]],
            metadata={"filename": "f.txt"},
        )


def test_add_chunks_empty_list_is_a_noop(store: VectorStore) -> None:
    store.add_chunks(
        document_id="doc-1", chunks=[], embeddings=[], metadata={"filename": "f.txt"}
    )
    assert store.get_or_create_collection().count() == 0


def test_query_returns_most_similar_chunk_first(store: VectorStore) -> None:
    store.add_chunks(
        document_id="doc-1",
        chunks=[_chunk("about cats", 0)],
        embeddings=[[1.0, 0.0]],
        metadata={"filename": "cats.txt"},
    )
    store.add_chunks(
        document_id="doc-2",
        chunks=[_chunk("about rockets", 0)],
        embeddings=[[0.0, 1.0]],
        metadata={"filename": "rockets.txt"},
    )

    results = store.query(query_embedding=[1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].text == "about cats"
    assert results[0].document_id == "doc-1"
    assert results[0].filename == "cats.txt"
    assert results[0].chunk_index == 0
    assert results[0].similarity > results[1].similarity


def test_query_filters_by_document_ids(store: VectorStore) -> None:
    store.add_chunks(
        document_id="doc-1",
        chunks=[_chunk("about cats", 0)],
        embeddings=[[1.0, 0.0]],
        metadata={"filename": "cats.txt"},
    )
    store.add_chunks(
        document_id="doc-2",
        chunks=[_chunk("about rockets", 0)],
        embeddings=[[0.0, 1.0]],
        metadata={"filename": "rockets.txt"},
    )

    results = store.query(query_embedding=[0.5, 0.5], top_k=5, document_ids=["doc-2"])

    assert len(results) == 1
    assert results[0].document_id == "doc-2"


def test_query_on_empty_store_returns_empty_list(store: VectorStore) -> None:
    assert store.query(query_embedding=[1.0, 0.0], top_k=5) == []


def test_query_non_positive_top_k_returns_empty_list(store: VectorStore) -> None:
    store.add_chunks(
        document_id="doc-1",
        chunks=[_chunk("about cats", 0)],
        embeddings=[[1.0, 0.0]],
        metadata={"filename": "cats.txt"},
    )
    assert store.query(query_embedding=[1.0, 0.0], top_k=0) == []


def test_query_roundtrips_page_and_char_range(store: VectorStore) -> None:
    store.add_chunks(
        document_id="doc-1",
        chunks=[
            Chunk(text="about cats", index=2, char_start=40, char_end=50, page=3)
        ],
        embeddings=[[1.0, 0.0]],
        metadata={"filename": "cats.pdf"},
    )

    [result] = store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert result.page == 3
    assert result.char_start == 40
    assert result.char_end == 50
    assert result.chunk_index == 2


def test_query_returns_none_page_for_unpaginated_chunk(store: VectorStore) -> None:
    # page defaults to None (e.g. a .txt chunk); ChromaDB can't store None, so
    # the key is omitted on write and must read back as None, not a crash.
    store.add_chunks(
        document_id="doc-1",
        chunks=[Chunk(text="about cats", index=0, char_start=0, char_end=10)],
        embeddings=[[1.0, 0.0]],
        metadata={"filename": "cats.txt"},
    )

    [result] = store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert result.page is None
    assert result.char_start == 0
    assert result.char_end == 10


def test_delete_document_removes_its_chunks(store: VectorStore) -> None:
    store.add_chunks(
        document_id="doc-1",
        chunks=[_chunk("about cats", 0), _chunk("more cats", 1)],
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
        metadata={"filename": "cats.txt"},
    )
    store.add_chunks(
        document_id="doc-2",
        chunks=[_chunk("about rockets", 0)],
        embeddings=[[0.0, 1.0]],
        metadata={"filename": "rockets.txt"},
    )

    store.delete_document("doc-1")

    remaining = store.query(query_embedding=[1.0, 0.0], top_k=10)
    assert [r.document_id for r in remaining] == ["doc-2"]
