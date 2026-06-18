"""Orchestrates the full ingest -> chunk -> embed -> store pipeline."""

from __future__ import annotations

from app.models.document import Document, DocumentStatus
from app.services.base import EmbeddingsDisabledError, EmbeddingService
from app.services.ingestion import assign_pages, chunk_text, extract
from app.services.registry import DocumentRegistry
from app.services.vector_store import VectorStore


class DocumentIndexer:
    """Runs ingestion, embedding, and storage, recording status in the registry."""

    def __init__(
        self,
        registry: DocumentRegistry,
        vector_store: VectorStore,
        embedding_service: EmbeddingService | None,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._registry = registry
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def index_document(self, file_bytes: bytes, filename: str) -> Document:
        """Extract, chunk, embed, and store ``file_bytes``, returning its record.

        The document is registered as ``PROCESSING`` immediately, then moves
        to ``READY`` on success or ``FAILED`` if any step raises.

        Raises:
            EmbeddingsDisabledError: If no embedding provider is configured.
        """
        if self._embedding_service is None:
            raise EmbeddingsDisabledError(
                "Cannot index a document: no embedding provider is configured. "
                "Set PROVIDER=ollama or PROVIDER=openai in .env to enable it."
            )

        document = Document(filename=filename, status=DocumentStatus.PROCESSING)
        self._registry.add(document)

        try:
            extraction = extract(file_bytes, filename)
            chunks = chunk_text(
                extraction.text, self._chunk_size, self._chunk_overlap
            )
            assign_pages(chunks, extraction.page_starts)

            if chunks:
                # One batched call instead of one request per chunk.
                embeddings = await self._embedding_service.embed_texts(
                    [chunk.text for chunk in chunks]
                )
                self._vector_store.add_chunks(
                    document_id=document.id,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata={"filename": filename},
                )

            document.num_chunks = len(chunks)
            document.status = DocumentStatus.READY
        except Exception:
            document.status = DocumentStatus.FAILED
            self._registry.update(document)
            raise
        else:
            self._registry.update(document)

        return document
