"""Maps domain errors raised by the service layer to HTTP responses.

Registered once on the app so routers can let these exceptions propagate
instead of repeating try/except blocks in every handler.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.services.base import EmbeddingsDisabledError, ProviderUnreachableError
from app.services.ingestion import UnsupportedFileTypeError

logger = logging.getLogger(__name__)


class UnhandledErrorMiddleware(BaseHTTPMiddleware):
    """Safety net for exceptions with no registered handler (e.g. a bare bug).

    A plain ``@app.exception_handler(Exception)`` would *not* fix this: Starlette
    special-cases that key to run outside the CORS middleware, so the browser
    would still block the response and the frontend would see a confusing CORS
    failure instead of the actual error. A regular middleware positioned inside
    CORS middleware (added before it — see ``main.py``) doesn't have that
    problem, since its response still passes back out through CORS.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)
        except Exception:
            logger.exception("Unhandled exception while processing request")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error. Check the backend logs for details."
                },
            )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach domain-error -> HTTP-response handlers to ``app``.

    Registering handlers here (rather than letting these propagate as
    unhandled exceptions) matters beyond just status codes: Starlette's
    catch-all for truly unhandled exceptions runs *outside* the CORS
    middleware, so that response would be missing CORS headers and the
    browser would block it entirely, leaving the frontend unable to show
    any error at all. A registered handler's response still passes back
    through CORS middleware.
    """

    @app.exception_handler(UnsupportedFileTypeError)
    async def _unsupported_file_type(
        request: Request, exc: UnsupportedFileTypeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(EmbeddingsDisabledError)
    async def _embeddings_disabled(
        request: Request, exc: EmbeddingsDisabledError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ProviderUnreachableError)
    async def _provider_unreachable(
        request: Request, exc: ProviderUnreachableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": str(exc)},
        )
