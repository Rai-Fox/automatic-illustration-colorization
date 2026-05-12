from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request

from services.api.app.container import AppContainer
from services.api.app.core.config import ApiSettings, load_settings
from services.api.app.presentation.routes import router

LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    settings: ApiSettings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()
    resolved_container = container or AppContainer.from_settings(resolved_settings)
    app = FastAPI(title="Illustration Colorization API")
    app.state.container = resolved_container

    @app.middleware("http")
    async def log_http_request(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        client = request.client.host if request.client else "-"
        content_length = request.headers.get("content-length", "-")
        started_at = time.perf_counter()
        LOGGER.info(
            "api request started request_id=%s method=%s path=%s client=%s "
            "content_length=%s",
            request_id,
            request.method,
            request.url.path,
            client,
            content_length,
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            LOGGER.exception(
                "api request failed request_id=%s method=%s path=%s "
                "duration_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "api request completed request_id=%s method=%s path=%s "
            "status_code=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    app.include_router(router)
    return app
