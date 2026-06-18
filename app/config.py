"""Application configuration loaded from environment / `.env`.

All settings are provider-agnostic. Switching the active backend is a single
change to ``PROVIDER`` in ``.env`` — no code changes required.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """Supported backends for embeddings and answer generation."""

    NONE = "none"
    OLLAMA = "ollama"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Provider selection -----------------------------------------------------
    provider: Provider = Provider.OLLAMA

    # Ollama -----------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3.2"

    # OpenAI -----------------------------------------------------------------
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Retrieval / chunking ---------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    #: Minimum top-source similarity (0..1) for an answer to count as
    #: well-grounded. When the best retrieved chunk scores below this, ``/ask``
    #: returns ``status=insufficient_context`` and declines to answer rather
    #: than grounding a reply on a weak match. Tune to the embedding model.
    grounding_threshold: float = 0.4

    # Storage ----------------------------------------------------------------
    chroma_dir: str = "chroma_db"
    upload_dir: str = "uploads"
    max_upload_size_bytes: int = 20 * 1024 * 1024  # 20 MB
    #: Cap on how many files a single upload batch (e.g. a multi-file drag and
    #: drop in the UI) may submit. The API itself only ever accepts one file
    #: per HTTP request; this bounds client-driven batches of those requests.
    max_files_per_request: int = 20

    # Computed capabilities --------------------------------------------------
    @property
    def embeddings_enabled(self) -> bool:
        """True when a provider capable of producing embeddings is configured.

        This reflects *configuration*, not live reachability. The configured
        backend may still be unreachable at request time; callers should handle
        connection errors gracefully.
        """
        if self.provider is Provider.OLLAMA:
            return bool(self.ollama_embed_model)
        if self.provider is Provider.OPENAI:
            return bool(self.openai_api_key and self.openai_embed_model)
        return False

    @property
    def generation_enabled(self) -> bool:
        """True when a chat-capable LLM is configured for answer generation."""
        if self.provider is Provider.OLLAMA:
            return bool(self.ollama_chat_model)
        if self.provider is Provider.OPENAI:
            return bool(self.openai_api_key and self.openai_chat_model)
        return False

    @property
    def embed_model_name(self) -> str | None:
        """Human-readable name of the active embedding model, if any."""
        if self.provider is Provider.OLLAMA:
            return self.ollama_embed_model
        if self.provider is Provider.OPENAI:
            return self.openai_embed_model
        return None

    @property
    def chat_model_name(self) -> str | None:
        """Human-readable name of the active chat model, if any."""
        if not self.generation_enabled:
            return None
        if self.provider is Provider.OLLAMA:
            return self.ollama_chat_model
        if self.provider is Provider.OPENAI:
            return self.openai_chat_model
        return None


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the `.env` file is parsed once per process. Use this as the
    single source of truth for configuration throughout the app.
    """
    return Settings()
