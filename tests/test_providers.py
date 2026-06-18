"""Tests for the concrete Ollama and OpenAI provider services.

These are the only modules that touch the network, so they are the riskiest to
leave untested: response parsing and error mapping must be exactly right. Every
call here is faked — no real Ollama instance and no real OpenAI API are
contacted. The HTTP client (``httpx.AsyncClient``) and the OpenAI SDK client
(``AsyncOpenAI``) are monkeypatched with in-memory fakes.
"""

from __future__ import annotations

import httpx
import pytest

from app.services import ollama as ollama_mod
from app.services import openai as openai_mod
from app.services.base import ProviderUnreachableError
from app.services.ollama import OllamaEmbeddingService, OllamaLLMService
from app.services.openai import OpenAIEmbeddingService, OpenAILLMService

# ---------------------------------------------------------------------------
# Ollama — faked httpx.AsyncClient
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("POST", "http://x"), response=None  # type: ignore[arg-type]
            )


class _FakeOllamaClient:
    """Stands in for ``httpx.AsyncClient`` as used by the Ollama services.

    ``fail=True`` makes every call raise an ``httpx.ConnectError``, simulating
    Ollama not running — the realistic "unreachable" path.
    """

    def __init__(
        self,
        *,
        embedding: list[float] | None = None,
        response: str | None = None,
        fail: bool = False,
        tags_status: int = 200,
    ) -> None:
        self._embedding = embedding
        self._response = response
        self._fail = fail
        self._tags_status = tags_status

    async def __aenter__(self) -> "_FakeOllamaClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, json: dict | None = None) -> _FakeResponse:
        if self._fail:
            raise httpx.ConnectError("connection refused")
        if url.endswith("/api/embeddings"):
            return _FakeResponse({"embedding": self._embedding})
        if url.endswith("/api/generate"):
            return _FakeResponse({"response": self._response})
        raise AssertionError(f"unexpected POST {url}")

    async def get(self, url: str) -> _FakeResponse:
        if self._fail:
            raise httpx.ConnectError("connection refused")
        return _FakeResponse({}, status_code=self._tags_status)


def _patch_ollama(monkeypatch: pytest.MonkeyPatch, client: _FakeOllamaClient) -> None:
    monkeypatch.setattr(
        ollama_mod.httpx, "AsyncClient", lambda *a, **k: client
    )


async def test_ollama_embed_texts_parses_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _FakeOllamaClient(embedding=[0.1, 0.2, 0.3]))
    service = OllamaEmbeddingService(base_url="http://localhost:11434", model="nomic")

    vectors = await service.embed_texts(["one", "two"])

    # One request per text, each returning the parsed embedding vector.
    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert service.model_name == "nomic"


async def test_ollama_embed_texts_unreachable_maps_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _FakeOllamaClient(fail=True))
    service = OllamaEmbeddingService(base_url="http://localhost:11434", model="nomic")

    with pytest.raises(ProviderUnreachableError):
        await service.embed_texts(["hello"])


async def test_ollama_generate_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _FakeOllamaClient(response="Generated answer [1]."))
    service = OllamaLLMService(base_url="http://localhost:11434", model="llama3.2")

    assert service.enabled is True
    assert service.model_name == "llama3.2"
    answer = await service.generate("prompt", system="be precise")
    assert answer == "Generated answer [1]."


async def test_ollama_generate_unreachable_maps_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _FakeOllamaClient(fail=True))
    service = OllamaLLMService(base_url="http://localhost:11434", model="llama3.2")

    with pytest.raises(ProviderUnreachableError):
        await service.generate("prompt")


async def test_ollama_is_reachable_true_when_tags_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _FakeOllamaClient(tags_status=200))
    service = OllamaEmbeddingService(base_url="http://localhost:11434", model="nomic")
    assert await service.is_reachable() is True


async def test_ollama_is_reachable_false_when_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _FakeOllamaClient(fail=True))
    service = OllamaLLMService(base_url="http://localhost:11434", model="llama3.2")
    assert await service.is_reachable() is False


# ---------------------------------------------------------------------------
# OpenAI — faked AsyncOpenAI SDK client
# ---------------------------------------------------------------------------


def _api_error(message: str = "boom") -> openai_mod.APIError:
    return openai_mod.APIError(
        message, request=httpx.Request("POST", "https://api.openai.com"), body=None
    )


class _EmbItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _EmbResponse:
    def __init__(self, data: list[_EmbItem]) -> None:
        self.data = data


class _FakeEmbeddings:
    def __init__(self, *, fail: bool, scrambled: bool) -> None:
        self._fail = fail
        self._scrambled = scrambled

    async def create(self, model: str, input: list[str]) -> _EmbResponse:
        if self._fail:
            raise _api_error()
        items = [_EmbItem(i, [float(i)]) for i in range(len(input))]
        if self._scrambled:
            # API is documented to preserve order, but the service sorts by
            # index defensively; hand back reversed to prove the sort works.
            items = list(reversed(items))
        return _EmbResponse(items)


class _ChatMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _ChatChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _ChatMessage(content)


class _ChatResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_ChatChoice(content)]


class _FakeChatCompletions:
    def __init__(self, *, fail: bool, content: str | None) -> None:
        self._fail = fail
        self._content = content

    async def create(self, model: str, messages: list[dict]) -> _ChatResponse:
        if self._fail:
            raise _api_error()
        return _ChatResponse(self._content)


class _FakeModels:
    def __init__(self, *, fail: bool) -> None:
        self._fail = fail

    async def retrieve(self, model: str) -> object:
        if self._fail:
            raise _api_error()
        return object()


class _FakeAsyncOpenAI:
    def __init__(
        self,
        *,
        fail: bool = False,
        scrambled: bool = False,
        content: str | None = "Answer [1].",
    ) -> None:
        self.embeddings = _FakeEmbeddings(fail=fail, scrambled=scrambled)
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeChatCompletions(fail=fail, content=content)
        self.models = _FakeModels(fail=fail)


def _patch_openai(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> None:
    monkeypatch.setattr(
        openai_mod, "AsyncOpenAI", lambda api_key: _FakeAsyncOpenAI(**kwargs)
    )


async def test_openai_embed_texts_orders_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai(monkeypatch, scrambled=True)
    service = OpenAIEmbeddingService(api_key="sk-test", model="text-embedding-3-small")

    vectors = await service.embed_texts(["a", "b", "c"])

    # Sorted back into input order regardless of the scrambled response.
    assert vectors == [[0.0], [1.0], [2.0]]


async def test_openai_embed_texts_error_maps_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai(monkeypatch, fail=True)
    service = OpenAIEmbeddingService(api_key="sk-test", model="text-embedding-3-small")

    with pytest.raises(ProviderUnreachableError):
        await service.embed_texts(["a"])


async def test_openai_generate_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai(monkeypatch, content="Paris is the capital [1].")
    service = OpenAILLMService(api_key="sk-test", model="gpt-4o-mini")

    assert service.enabled is True
    assert service.model_name == "gpt-4o-mini"
    answer = await service.generate("prompt", system="be precise")
    assert answer == "Paris is the capital [1]."


async def test_openai_generate_handles_null_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai(monkeypatch, content=None)
    service = OpenAILLMService(api_key="sk-test", model="gpt-4o-mini")
    assert await service.generate("prompt") == ""


async def test_openai_generate_error_maps_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai(monkeypatch, fail=True)
    service = OpenAILLMService(api_key="sk-test", model="gpt-4o-mini")

    with pytest.raises(ProviderUnreachableError):
        await service.generate("prompt")


async def test_openai_is_reachable_true_then_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai(monkeypatch, fail=False)
    ok = OpenAILLMService(api_key="sk-test", model="gpt-4o-mini")
    assert await ok.is_reachable() is True

    _patch_openai(monkeypatch, fail=True)
    down = OpenAIEmbeddingService(api_key="sk-test", model="text-embedding-3-small")
    assert await down.is_reachable() is False
