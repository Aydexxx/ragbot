"""Vector storage backed by a local persistent ChromaDB collection.

Wraps the chromadb client so the rest of the app depends on a small, typed
interface instead of the client API directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import chromadb
from chromadb.api.models.Collection import Collection

from app.models.chunk import Chunk, RetrievedChunk

_COLLECTION_NAME = "chunks"


class VectorStore:
    """A ChromaDB collection of document chunks, keyed by ``document_id``.

    The collection is configured for cosine distance, so
    ``similarity = 1 - distance`` is a stable score independent of which
    embedding backend produced the vectors.
    """

    def __init__(self, persist_dir: str | Path) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection: Collection | None = None

    def get_or_create_collection(self) -> Collection:
        """Return the chunks collection, creating it on first use."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                _COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def add_chunks(
        self,
        document_id: UUID | str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> None:
        """Store chunk text, embeddings, and metadata for one document.

        ``metadata`` (e.g. ``{"filename": "report.pdf"}``) is merged into
        every chunk's record alongside the computed ``document_id`` and
        ``chunk_index``.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return

        doc_id = str(document_id)
        collection = self.get_or_create_collection()
        collection.add(
            ids=[_chunk_id(doc_id, chunk.index) for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_chunk_metadata(metadata, doc_id, chunk) for chunk in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[UUID | str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best match first.

        Restricts the search to ``document_ids`` when given. Returns fewer
        than ``top_k`` results if the (filtered) collection is smaller, and
        an empty list for a non-positive ``top_k``.
        """
        if top_k <= 0:
            return []

        collection = self.get_or_create_collection()
        where = (
            {"document_id": {"$in": [str(d) for d in document_ids]}}
            if document_ids
            else None
        )
        result = collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            RetrievedChunk(
                text=documents[i],
                document_id=str(metadatas[i]["document_id"]),
                filename=str(metadatas[i]["filename"]),
                chunk_index=int(metadatas[i]["chunk_index"]),
                similarity=1.0 - distances[i],
                page=_optional_int(metadatas[i].get("page")),
                char_start=int(metadatas[i].get("char_start", 0)),
                char_end=int(metadatas[i].get("char_end", 0)),
            )
            for i in range(len(ids))
        ]

    def delete_document(self, document_id: UUID | str) -> None:
        """Remove all chunks belonging to ``document_id``."""
        collection = self.get_or_create_collection()
        collection.delete(where={"document_id": str(document_id)})


def _chunk_metadata(
    base: dict[str, Any], document_id: str, chunk: Chunk
) -> dict[str, Any]:
    """Per-chunk metadata: caller-supplied fields plus locators.

    ChromaDB rejects ``None`` metadata values, so ``page`` is omitted entirely
    for unpaginated documents (read back as ``None`` via ``.get``).
    """
    record: dict[str, Any] = {
        **base,
        "document_id": document_id,
        "chunk_index": chunk.index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }
    if chunk.page is not None:
        record["page"] = chunk.page
    return record


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"
