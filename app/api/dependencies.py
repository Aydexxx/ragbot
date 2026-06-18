"""FastAPI dependency providers.

These wire configuration and the concrete service implementations into request
handlers using ``Depends``, keeping routers free of construction logic.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.core.factory import create_embedding_service, create_llm_service
from app.services.base import EmbeddingService, LLMService
from app.services.conversation import ConversationStore
from app.services.indexer import DocumentIndexer
from app.services.rag import RagService
from app.services.registry import DocumentRegistry, JsonFileDocumentRegistry
from app.services.vector_store import VectorStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_embedding_service(settings: SettingsDep) -> EmbeddingService | None:
    """Provide the configured embedding service (or None when disabled)."""
    return create_embedding_service(settings)


EmbeddingServiceDep = Annotated[
    "EmbeddingService | None", Depends(get_embedding_service)
]


def get_llm_service(settings: SettingsDep) -> LLMService:
    """Provide the configured LLM service (NullLLMService when disabled)."""
    return create_llm_service(settings)


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]


_document_registry: DocumentRegistry | None = None
_document_registry_lock = threading.Lock()


def get_document_registry() -> DocumentRegistry:
    """Provide the document registry as a process-wide singleton.

    Built lazily, on first use, behind a lock — not ``@lru_cache``. FastAPI
    runs sync dependencies like this one in a thread pool, so two requests
    can race to construct it concurrently on first use; ``lru_cache`` alone
    doesn't prevent that race, only double-checked locking does. Tests
    override this dependency directly via ``app.dependency_overrides``,
    never reaching this function at all.
    """
    global _document_registry
    if _document_registry is None:
        with _document_registry_lock:
            if _document_registry is None:
                settings = get_settings()
                _document_registry = JsonFileDocumentRegistry(
                    Path(settings.upload_dir) / "documents.json"
                )
    return _document_registry


DocumentRegistryDep = Annotated[DocumentRegistry, Depends(get_document_registry)]


_vector_store: VectorStore | None = None
_vector_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """Provide the vector store as a process-wide singleton.

    Lock-guarded for the same reason as :func:`get_document_registry`: two
    requests racing to construct ChromaDB's ``PersistentClient`` for the same
    not-yet-initialized directory at the same time corrupts its on-disk
    tenant metadata (observed directly — see the regression test for this).
    """
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                settings = get_settings()
                _vector_store = VectorStore(settings.chroma_dir)
    return _vector_store


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]


def get_document_indexer(
    settings: SettingsDep,
    registry: DocumentRegistryDep,
    vector_store: VectorStoreDep,
    embedding_service: EmbeddingServiceDep,
) -> DocumentIndexer:
    """Provide a :class:`DocumentIndexer` wired to the configured services."""
    return DocumentIndexer(
        registry=registry,
        vector_store=vector_store,
        embedding_service=embedding_service,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


DocumentIndexerDep = Annotated[DocumentIndexer, Depends(get_document_indexer)]


def get_rag_service(
    settings: SettingsDep,
    vector_store: VectorStoreDep,
    embedding_service: EmbeddingServiceDep,
    llm_service: LLMServiceDep,
) -> RagService:
    """Provide a :class:`RagService` wired to the configured services."""
    return RagService(
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
        top_k=settings.top_k,
        grounding_threshold=settings.grounding_threshold,
    )


RagServiceDep = Annotated[RagService, Depends(get_rag_service)]


_conversation_store: ConversationStore | None = None
_conversation_store_lock = threading.Lock()


def get_conversation_store() -> ConversationStore:
    """Provide the conversation history store as a process-wide singleton.

    Must be a singleton (not built fresh per request) so history actually
    accumulates across turns. Lock-guarded for the same first-use race as the
    other singletons here, even though a stray duplicate instance here would
    only cost a dropped turn rather than on-disk corruption.
    """
    global _conversation_store
    if _conversation_store is None:
        with _conversation_store_lock:
            if _conversation_store is None:
                _conversation_store = ConversationStore()
    return _conversation_store


ConversationStoreDep = Annotated[ConversationStore, Depends(get_conversation_store)]
