"""Factory that builds concrete services from configuration.

The rest of the app depends only on the abstract interfaces in
``app.services.base``; this module is the single place that knows about
concrete providers. Switching providers is therefore a config-only change.
"""

from __future__ import annotations

from app.config import Provider, Settings
from app.services.base import EmbeddingService, LLMService
from app.services.null import NullLLMService
from app.services.ollama import OllamaEmbeddingService, OllamaLLMService
from app.services.openai import OpenAIEmbeddingService, OpenAILLMService


def create_embedding_service(settings: Settings) -> EmbeddingService | None:
    """Return the configured embedding service, or None when unavailable.

    None signals that embeddings are disabled (``PROVIDER=none`` or a misconfig
    such as ``openai`` without an API key). Callers should check before use.
    """
    if not settings.embeddings_enabled:
        return None

    if settings.provider is Provider.OLLAMA:
        return OllamaEmbeddingService(
            base_url=settings.ollama_url,
            model=settings.ollama_embed_model,
        )
    if settings.provider is Provider.OPENAI:
        return OpenAIEmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embed_model,
        )
    return None


def create_llm_service(settings: Settings) -> LLMService:
    """Return the configured LLM service.

    Always returns a usable instance: a :class:`NullLLMService` when no chat
    model is configured, so generation degrades gracefully instead of crashing.
    """
    if not settings.generation_enabled:
        return NullLLMService()

    if settings.provider is Provider.OLLAMA:
        return OllamaLLMService(
            base_url=settings.ollama_url,
            model=settings.ollama_chat_model,
        )
    if settings.provider is Provider.OPENAI:
        return OpenAILLMService(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
        )
    return NullLLMService()
