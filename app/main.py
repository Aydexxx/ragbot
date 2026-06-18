"""FastAPI application entry point.

Run locally with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import ask, documents, health
from app.api.errors import UnhandledErrorMiddleware, register_exception_handlers

# The React dev server's origin (Vite default port). Add production origins
# here once the frontend has a deployed URL.
_DEV_FRONTEND_ORIGIN = "http://localhost:5173"

app = FastAPI(
    title="RAGBot",
    version=__version__,
    summary="A document Q&A system built on Retrieval-Augmented Generation.",
)

# Order matters: added before CORSMiddleware, so it ends up *inside* it (closer
# to the routes). That way a response built here from a caught exception still
# passes back out through CORSMiddleware and gets its headers. The reverse
# order would let the browser block the response as a bare CORS failure.
app.add_middleware(UnhandledErrorMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_DEV_FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ask.router)


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """Minimal landing endpoint pointing to the docs."""
    return {"name": "RAGBot", "version": __version__, "docs": "/docs"}
