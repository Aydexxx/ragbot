"""Health / status endpoint reporting provider configuration and reachability."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.dependencies import EmbeddingServiceDep, LLMServiceDep, SettingsDep
from app.models.health import HealthResponse, ProviderStatus, UploadLimits
from app.services.ingestion import SUPPORTED_EXTENSIONS

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: SettingsDep,
    embedding_service: EmbeddingServiceDep,
    llm_service: LLMServiceDep,
) -> HealthResponse:
    """Report provider status without crashing when the backend is offline.

    ``reachable`` is determined by a best-effort probe; an unreachable backend
    reports ``reachable=false`` rather than raising.
    """
    # Probe whichever services exist. A missing embedding service (disabled)
    # counts as not reachable, but never errors.
    embed_reachable = (
        await embedding_service.is_reachable()
        if embedding_service is not None
        else False
    )
    llm_reachable = await llm_service.is_reachable() if llm_service.enabled else False
    reachable = embed_reachable or llm_reachable

    status = ProviderStatus(
        provider=settings.provider.value,
        embeddings_enabled=settings.embeddings_enabled,
        generation_enabled=settings.generation_enabled,
        embed_model=settings.embed_model_name,
        chat_model=settings.chat_model_name,
        reachable=reachable,
    )
    limits = UploadLimits(
        max_upload_size_bytes=settings.max_upload_size_bytes,
        max_files_per_request=settings.max_files_per_request,
        allowed_extensions=SUPPORTED_EXTENSIONS,
    )
    return HealthResponse(version=__version__, providers=status, limits=limits)
