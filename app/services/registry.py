"""Document registry: tracks uploaded documents and their metadata.

Defined behind an abstract interface so the storage backend (a JSON file
today, a real database later) can change without touching callers.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

from app.models.document import Document


class DocumentRegistry(ABC):
    """Stores and retrieves document metadata records."""

    @abstractmethod
    def add(self, document: Document) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, document_id: UUID) -> Document | None:
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def update(self, document: Document) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: UUID) -> None:
        raise NotImplementedError


class JsonFileDocumentRegistry(DocumentRegistry):
    """Persists document metadata as a JSON array on disk.

    Suitable for a single process at prototype scale: the whole file is read
    and rewritten on each mutation, so it isn't safe for concurrent writers.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_all([])

    def add(self, document: Document) -> None:
        documents = self._read_all()
        documents.append(document)
        self._write_all(documents)

    def get(self, document_id: UUID) -> Document | None:
        for document in self._read_all():
            if document.id == document_id:
                return document
        return None

    def list_all(self) -> list[Document]:
        return self._read_all()

    def update(self, document: Document) -> None:
        documents = self._read_all()
        for i, existing in enumerate(documents):
            if existing.id == document.id:
                documents[i] = document
                self._write_all(documents)
                return
        raise KeyError(f"Document {document.id} not found")

    def delete(self, document_id: UUID) -> None:
        documents = [d for d in self._read_all() if d.id != document_id]
        self._write_all(documents)

    def _read_all(self) -> list[Document]:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return [Document.model_validate(item) for item in raw]

    def _write_all(self, documents: list[Document]) -> None:
        payload = [d.model_dump(mode="json") for d in documents]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
