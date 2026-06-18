"""Tests for FastAPI dependency providers — focused on thread-safety.

FastAPI runs sync dependencies (like these) in a thread pool, so two
concurrent requests can race to construct a lazily-built singleton on first
use. A bare ``@lru_cache`` doesn't prevent that race; only locking does. This
was caught empirically: concurrent first-construction of ChromaDB's
``PersistentClient`` against the same not-yet-initialized directory corrupts
its on-disk tenant metadata (``ValueError: Could not connect to tenant
default_tenant``).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.api import dependencies
from app.config import Settings


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Each test gets a clean slate so prior tests don't leak instances."""
    dependencies._document_registry = None
    dependencies._vector_store = None
    yield
    dependencies._document_registry = None
    dependencies._vector_store = None


def _run_concurrently(fn, count: int) -> tuple[list, list[BaseException]]:
    results: list = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            results.append(fn())
        except BaseException as exc:  # noqa: BLE001 - capturing for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def test_get_vector_store_is_safe_under_concurrent_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(chroma_dir=str(tmp_path / "chroma_db"))
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)

    results, errors = _run_concurrently(dependencies.get_vector_store, count=8)

    assert errors == []
    assert len(results) == 8
    assert len({id(r) for r in results}) == 1


def test_get_document_registry_is_safe_under_concurrent_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(upload_dir=str(tmp_path / "uploads"))
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)

    results, errors = _run_concurrently(dependencies.get_document_registry, count=8)

    assert errors == []
    assert len(results) == 8
    assert len({id(r) for r in results}) == 1
