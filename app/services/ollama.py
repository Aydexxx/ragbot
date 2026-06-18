"""Ollama-backed embedding and LLM services.

Talks to a local Ollama instance over HTTP via ``httpx``. This is the default,
no-API-key path that lets the full pipeline run for free.
"""

from __future__ import annotations

import httpx

from app.services.base import EmbeddingService, LLMService, ProviderUnreachableError

# Ollama is local; generation can be slow, embeddings are quick.
_PROBE_TIMEOUT = 2.0
_EMBED_TIMEOUT = 60.0
_GENERATE_TIMEOUT = 120.0


class OllamaEmbeddingService(EmbeddingService):
    """Embeddings via Ollama's ``/api/embeddings`` endpoint."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Ollama's embeddings endpoint accepts a single prompt per request, so
        # we issue one request per text over a shared client/connection pool.
        try:
            async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
                vectors: list[list[float]] = []
                for text in texts:
                    resp = await client.post(
                        f"{self._base_url}/api/embeddings",
                        json={"model": self._model, "prompt": text},
                    )
                    resp.raise_for_status()
                    vectors.append(resp.json()["embedding"])
                return vectors
        except httpx.HTTPError as exc:
            raise ProviderUnreachableError(
                f"Could not reach Ollama at {self._base_url} for embeddings: {exc}"
            ) from exc

    async def is_reachable(self) -> bool:
        return await _probe(self._base_url)


class OllamaLLMService(LLMService):
    """Answer generation via Ollama's ``/api/generate`` endpoint."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str | None:
        return self._model

    async def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=_GENERATE_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
                return resp.json()["response"]
        except httpx.HTTPError as exc:
            raise ProviderUnreachableError(
                f"Could not reach Ollama at {self._base_url} for generation: {exc}"
            ) from exc

    async def is_reachable(self) -> bool:
        return await _probe(self._base_url)


async def _probe(base_url: str) -> bool:
    """Best-effort liveness check against the Ollama root endpoint."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
