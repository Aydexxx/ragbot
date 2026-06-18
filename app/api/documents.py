"""Document upload, listing, and deletion endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.dependencies import (
    DocumentIndexerDep,
    DocumentRegistryDep,
    SettingsDep,
    VectorStoreDep,
)
from app.models.document import Document

router = APIRouter(prefix="/documents", tags=["documents"])

# Read in bounded pieces while enforcing the size limit, instead of buffering
# an attacker-controlled body fully into memory before checking it.
_READ_CHUNK_BYTES = 1024 * 1024


@router.post("", response_model=Document, status_code=status.HTTP_201_CREATED)
async def upload_document(
    indexer: DocumentIndexerDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
) -> Document:
    """Index an uploaded file (PDF/TXT/MD): extract, chunk, embed, and store it.

    Indexing runs synchronously, inline, on this request — acceptable at
    prototype scale. At scale, swap this for a background task/queue (FastAPI
    ``BackgroundTasks``, Celery, RQ, ...): return 202 Accepted with a
    ``PENDING`` :class:`Document` immediately, and let the client poll
    ``GET /documents`` (or a future ``GET /documents/{id}``) for status.
    """
    file_bytes = await _read_limited(file, settings.max_upload_size_bytes)
    return await indexer.index_document(file_bytes, file.filename or "upload")


@router.get("", response_model=list[Document])
async def list_documents(registry: DocumentRegistryDep) -> list[Document]:
    """List all indexed documents."""
    return registry.list_all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    registry: DocumentRegistryDep,
    vector_store: VectorStoreDep,
) -> None:
    """Remove a document's metadata and its chunks from the vector store."""
    if registry.get(document_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )
    vector_store.delete_document(document_id)
    registry.delete(document_id)


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read ``file`` up to ``max_bytes``, raising 413 instead of over-buffering."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"File exceeds the {_human_size(max_bytes)} upload limit."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _human_size(num_bytes: int) -> str:
    """Render a byte count as a friendly MB figure (e.g. ``"20.0 MB"``)."""
    return f"{num_bytes / (1024 * 1024):.1f} MB"
