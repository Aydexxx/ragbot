"""Tests for the /health endpoint and provider factory wiring.

These verify graceful behaviour when no backend is reachable — the API must
report ``reachable=false`` rather than crashing.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.dependencies import get_embedding_service, get_llm_service
from app.config import Provider, Settings, get_settings
from app.core.factory import create_embedding_service, create_llm_service
from app.main import app
from app.services.base import EmbeddingService, LLMService
from app.services.null import NullLLMService

client = TestClient(app)


class _UnreachableEmbedding(EmbeddingService):
    """Configured embedder whose backend is offline — never makes a real call."""

    @property
    def model_name(self) -> str:
        return "offline-embed"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("health check must not embed")

    async def is_reachable(self) -> bool:
        return False


class _UnreachableLLM(LLMService):
    """Configured (enabled) chat model whose backend is offline."""

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return "offline-chat"

    async def generate(self, prompt: str, system: str | None = None) -> str:
        raise AssertionError("health check must not generate")

    async def is_reachable(self) -> bool:
        return False


def test_health_ok_when_backend_offline() -> None:
    """Default (ollama) config returns 200 with accurate flags even when the
    backend is offline.

    Settings and provider services are overridden so the test is hermetic —
    independent of the developer's local ``.env`` and making no network call.
    """
    ollama = Settings(provider=Provider.OLLAMA)
    app.dependency_overrides[get_settings] = lambda: ollama
    app.dependency_overrides[get_embedding_service] = _UnreachableEmbedding
    app.dependency_overrides[get_llm_service] = _UnreachableLLM
    try:
        resp = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    providers = body["providers"]
    assert providers["provider"] == "ollama"
    assert providers["embeddings_enabled"] is True
    assert providers["generation_enabled"] is True
    # Backend offline → reported gracefully as unreachable, never a crash.
    assert providers["reachable"] is False


def test_health_reports_default_upload_limits() -> None:
    """``limits`` mirrors the configured upload constraints for the frontend."""
    settings = Settings(provider=Provider.NONE)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        body = TestClient(app).get("/health").json()
    finally:
        app.dependency_overrides.clear()

    limits = body["limits"]
    assert limits["max_upload_size_bytes"] == settings.max_upload_size_bytes
    assert limits["max_files_per_request"] == 20
    assert limits["allowed_extensions"] == [".markdown", ".md", ".pdf", ".txt"]


def test_health_reports_overridden_upload_limits() -> None:
    settings = Settings(
        provider=Provider.NONE, max_upload_size_bytes=1024, max_files_per_request=5
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        body = TestClient(app).get("/health").json()
    finally:
        app.dependency_overrides.clear()

    limits = body["limits"]
    assert limits["max_upload_size_bytes"] == 1024
    assert limits["max_files_per_request"] == 5


def test_factory_none_provider_disables_everything() -> None:
    settings = Settings(provider=Provider.NONE)
    assert create_embedding_service(settings) is None
    assert isinstance(create_llm_service(settings), NullLLMService)
    assert settings.embeddings_enabled is False
    assert settings.generation_enabled is False


def test_factory_openai_without_key_is_disabled() -> None:
    settings = Settings(provider=Provider.OPENAI, openai_api_key="")
    assert settings.embeddings_enabled is False
    assert settings.generation_enabled is False
    assert create_embedding_service(settings) is None
    assert isinstance(create_llm_service(settings), NullLLMService)


def test_factory_ollama_builds_services() -> None:
    settings = Settings(provider=Provider.OLLAMA)
    embed = create_embedding_service(settings)
    llm = create_llm_service(settings)
    assert embed is not None
    assert embed.model_name == settings.ollama_embed_model
    assert llm.enabled is True
    assert llm.model_name == settings.ollama_chat_model


def test_ollama_without_chat_model_is_retrieval_only() -> None:
    """The headline free-tier scenario: Ollama for embeddings, no chat model.

    Embeddings (and therefore upload/index/retrieval) stay enabled while answer
    generation is disabled — exactly the retrieval-only mode the API serves
    when ``OLLAMA_CHAT_MODEL`` is left empty.
    """
    settings = Settings(provider=Provider.OLLAMA, ollama_chat_model="")

    assert settings.embeddings_enabled is True
    assert settings.generation_enabled is False
    assert settings.embed_model_name == settings.ollama_embed_model
    assert settings.chat_model_name is None

    assert create_embedding_service(settings) is not None
    assert isinstance(create_llm_service(settings), NullLLMService)


def test_health_reports_retrieval_only_flags_for_ollama_without_chat_model() -> None:
    """/health reflects the retrieval-only capability flags accurately.

    The provider services are overridden with disabled/unreachable fakes so the
    probe never touches the network — the flags come straight from settings.
    """
    from app.config import get_settings
    from app.api.dependencies import get_embedding_service, get_llm_service

    retrieval_only = Settings(provider=Provider.OLLAMA, ollama_chat_model="")
    app.dependency_overrides[get_settings] = lambda: retrieval_only
    app.dependency_overrides[get_embedding_service] = lambda: None
    app.dependency_overrides[get_llm_service] = NullLLMService
    try:
        body = TestClient(app).get("/health").json()
    finally:
        app.dependency_overrides.clear()

    providers = body["providers"]
    assert providers["provider"] == "ollama"
    assert providers["embeddings_enabled"] is True
    assert providers["generation_enabled"] is False
    assert providers["chat_model"] is None
    assert providers["reachable"] is False
