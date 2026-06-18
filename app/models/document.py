"""Schema for an ingested document's metadata record."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle status of an ingested document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(BaseModel):
    """Metadata record for an uploaded document."""

    id: UUID = Field(default_factory=uuid4)
    filename: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    num_chunks: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
