"""Schema for a single text chunk produced during ingestion."""

from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    """A contiguous slice of a document's extracted text."""

    text: str
    index: int
    char_start: int
    char_end: int
    #: 1-based page the chunk starts on, for paginated formats (PDF). ``None``
    #: for formats without pages (TXT/MD), where char offsets are the only
    #: locator.
    page: int | None = None


class RetrievedChunk(BaseModel):
    """A chunk returned from a vector store similarity query."""

    text: str
    document_id: str
    filename: str
    chunk_index: int
    similarity: float
    #: 1-based page, or ``None`` for unpaginated documents. See ``Chunk.page``.
    page: int | None = None
    #: Character offsets of the chunk within the document's extracted text,
    #: so a reader can locate the exact passage. Default 0/0 for chunks stored
    #: before these were tracked.
    char_start: int = 0
    char_end: int = 0
