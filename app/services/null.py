"""No-op LLM service used when no chat model is configured.

Lets the API boot and serve retrieval even when answer generation is
unavailable (``PROVIDER=none`` or a provider without a chat model). Calling
:meth:`generate` raises a clear, typed error so callers can degrade gracefully.
"""

from __future__ import annotations

from app.services.base import GenerationDisabledError, LLMService


class NullLLMService(LLMService):
    """An LLM service that is always disabled."""

    @property
    def enabled(self) -> bool:
        return False

    @property
    def model_name(self) -> str | None:
        return None

    async def generate(self, prompt: str, system: str | None = None) -> str:
        raise GenerationDisabledError(
            "Answer generation is disabled: no chat LLM is configured. "
            "Set PROVIDER=ollama or PROVIDER=openai in .env to enable it."
        )

    async def is_reachable(self) -> bool:
        return False
