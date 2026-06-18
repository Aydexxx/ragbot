"""OpenAI-backed embedding and LLM services.

Uses the official ``openai`` async SDK. Selected by setting ``PROVIDER=openai``
and supplying ``OPENAI_API_KEY`` in ``.env``.
"""

from __future__ import annotations

from openai import APIError, AsyncOpenAI

from app.services.base import EmbeddingService, LLMService, ProviderUnreachableError


class OpenAIEmbeddingService(EmbeddingService):
    """Embeddings via the OpenAI embeddings API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.embeddings.create(
                model=self._model, input=texts
            )
        except APIError as exc:
            raise ProviderUnreachableError(
                f"Could not reach OpenAI for embeddings: {exc}"
            ) from exc
        # The API preserves input order; sort defensively just in case.
        ordered = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def is_reachable(self) -> bool:
        try:
            await self._client.models.retrieve(self._model)
            return True
        except APIError:
            return False
        except Exception:  # noqa: BLE001 - network/auth issues are non-fatal here
            return False


class OpenAILLMService(LLMService):
    """Answer generation via the OpenAI chat completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return self._model

    async def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
            )
        except APIError as exc:
            raise ProviderUnreachableError(
                f"Could not reach OpenAI for generation: {exc}"
            ) from exc
        return resp.choices[0].message.content or ""

    async def is_reachable(self) -> bool:
        try:
            await self._client.models.retrieve(self._model)
            return True
        except APIError:
            return False
        except Exception:  # noqa: BLE001 - network/auth issues are non-fatal here
            return False
