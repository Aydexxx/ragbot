"""Tests for the JSON-file document registry."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.models.document import Document, DocumentStatus
from app.services.registry import JsonFileDocumentRegistry


@pytest.fixture
def registry(tmp_path: Path) -> JsonFileDocumentRegistry:
    return JsonFileDocumentRegistry(tmp_path / "documents.json")


def test_registry_starts_empty(registry: JsonFileDocumentRegistry) -> None:
    assert registry.list_all() == []


def test_registry_add_and_get(registry: JsonFileDocumentRegistry) -> None:
    document = Document(filename="report.pdf")
    registry.add(document)

    fetched = registry.get(document.id)
    assert fetched is not None
    assert fetched.filename == "report.pdf"
    assert fetched.status == DocumentStatus.PENDING


def test_registry_get_missing_returns_none(
    registry: JsonFileDocumentRegistry,
) -> None:
    assert registry.get(uuid4()) is None


def test_registry_update(registry: JsonFileDocumentRegistry) -> None:
    document = Document(filename="report.pdf")
    registry.add(document)

    document.status = DocumentStatus.READY
    document.num_chunks = 12
    registry.update(document)

    fetched = registry.get(document.id)
    assert fetched is not None
    assert fetched.status == DocumentStatus.READY
    assert fetched.num_chunks == 12


def test_registry_update_missing_raises(
    registry: JsonFileDocumentRegistry,
) -> None:
    with pytest.raises(KeyError):
        registry.update(Document(filename="ghost.txt"))


def test_registry_delete(registry: JsonFileDocumentRegistry) -> None:
    document = Document(filename="report.pdf")
    registry.add(document)
    registry.delete(document.id)
    assert registry.get(document.id) is None


def test_registry_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    document = Document(filename="report.pdf")
    JsonFileDocumentRegistry(path).add(document)

    reloaded = JsonFileDocumentRegistry(path)
    assert len(reloaded.list_all()) == 1
    assert reloaded.get(document.id).filename == "report.pdf"
