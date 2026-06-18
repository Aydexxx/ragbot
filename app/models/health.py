"""Schemas for the health / status endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderStatus(BaseModel):
    """Live status of the configured embedding/LLM backend."""

    provider: str = Field(description="Active provider: none | ollama | openai")
    embeddings_enabled: bool = Field(
        description="Whether an embedding model is configured."
    )
    generation_enabled: bool = Field(
        description="Whether a chat model is configured for answer generation."
    )
    embed_model: str | None = Field(
        default=None, description="Name of the active embedding model, if any."
    )
    chat_model: str | None = Field(
        default=None, description="Name of the active chat model, if any."
    )
    reachable: bool = Field(
        description="Whether the configured backend responded to a probe."
    )


class UploadLimits(BaseModel):
    """Configured upload constraints, so the frontend can validate client-side."""

    max_upload_size_bytes: int = Field(
        description="Largest accepted file size, in bytes."
    )
    max_files_per_request: int = Field(
        description="Largest number of files accepted in one upload batch."
    )
    allowed_extensions: list[str] = Field(
        description="File extensions accepted by the upload endpoint."
    )


class HealthResponse(BaseModel):
    """Top-level response for ``GET /health``."""

    status: str = Field(default="ok", description="Overall service status.")
    version: str = Field(description="RAGBot version.")
    providers: ProviderStatus
    limits: UploadLimits
