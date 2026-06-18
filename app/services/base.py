"""Abstract interfaces for the embedding and LLM service layers.

Concrete providers (Ollama, OpenAI, …) implement these contracts so that the
rest of the application never depends on a specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class GenerationDisabledError(RuntimeError):
    """Raised when answer generation is requested but no LLM is configured."""


class EmbeddingsDisabledError(RuntimeError):
    """Raised when embedding is requested but no embedding provider is configured."""


class ProviderUnreachableError(RuntimeError):
    """Raised when a configured provider's backend can't be reached or errors.

    Distinct from :class:`EmbeddingsDisabledError`/:class:`GenerationDisabledError`,
    which mean nothing is *configured*. This means something is configured but
    the network call to it failed (offline server, timeout, bad credentials).
    """


class EmbeddingService(ABC):
    """Turns text into dense vector representations."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Input strings to embed.

        Returns:
            One embedding vector per input text, in the same order.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_reachable(self) -> bool:
        """Return True if the backend currently responds to a lightweight probe."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        raise NotImplementedError


class LLMService(ABC):
    """Generates a natural-language answer from a prompt."""

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Produce a completion for ``prompt``.

        Args:
            prompt: The user/content prompt.
            system: Optional system instruction steering the model.

        Returns:
            The generated text.

        Raises:
            GenerationDisabledError: If generation is not available.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_reachable(self) -> bool:
        """Return True if the backend currently responds to a lightweight probe."""
        raise NotImplementedError

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this service can actually generate answers."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str | None:
        """Name of the underlying chat model, or None when disabled."""
        raise NotImplementedError
