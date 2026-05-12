from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)


class ColorizationApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def colorize(
        self,
        image_bytes: bytes,
        *,
        model_id: str | None = None,
        reference_bytes: bytes | None = None,
    ) -> bytes:
        started_at = time.perf_counter()
        LOGGER.info(
            "api client colorize request base_url=%s model_id=%s "
            "image_size_bytes=%s reference_sent=%s",
            self.base_url,
            model_id,
            len(image_bytes),
            reference_bytes is not None,
        )
        files = {"file": ("image.png", image_bytes, "image/png")}
        if reference_bytes is not None:
            files["reference"] = ("reference.png", reference_bytes, "image/png")
        data = {"model_id": model_id} if model_id else None
        response = await self._client.post(
            f"{self.base_url}/colorize",
            data=data,
            files=files,
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()
        result = base64.b64decode(payload["image_base64"])
        LOGGER.info(
            "api client colorize response status_code=%s duration_ms=%.2f "
            "result_size_bytes=%s",
            getattr(response, "status_code", None),
            (time.perf_counter() - started_at) * 1000,
            len(result),
        )
        return result

    async def list_models(self) -> list[dict[str, Any]]:
        started_at = time.perf_counter()
        LOGGER.info("api client list models request base_url=%s", self.base_url)
        response = await self._client.get(f"{self.base_url}/models", timeout=20.0)
        response.raise_for_status()
        payload = response.json()
        LOGGER.info(
            "api client list models response status_code=%s duration_ms=%.2f "
            "models_count=%s",
            getattr(response, "status_code", None),
            (time.perf_counter() - started_at) * 1000,
            len(payload),
        )
        return payload

    async def create_colorization_job(
        self,
        image_bytes: bytes,
        *,
        model_id: str,
        chat_id: int | None = None,
        reference_bytes: bytes | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        LOGGER.info(
            "api client create job request base_url=%s model_id=%s chat_id=%s "
            "image_size_bytes=%s reference_sent=%s seed_set=%s options_keys=%s",
            self.base_url,
            model_id,
            chat_id,
            len(image_bytes),
            reference_bytes is not None,
            seed is not None,
            sorted(options or {}),
        )
        files = {"file": ("image.png", image_bytes, "image/png")}
        if reference_bytes is not None:
            files["reference"] = ("reference.png", reference_bytes, "image/png")
        data: dict[str, Any] = {"model_id": model_id}
        if chat_id is not None:
            data["chat_id"] = chat_id
        if seed is not None:
            data["seed"] = seed
        if options:
            data["options"] = json.dumps(options)
        response = await self._client.post(
            f"{self.base_url}/jobs",
            data=data,
            files=files,
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        LOGGER.info(
            "api client create job response status_code=%s duration_ms=%.2f "
            "job_id=%s status=%s",
            getattr(response, "status_code", None),
            (time.perf_counter() - started_at) * 1000,
            payload.get("job_id"),
            payload.get("status"),
        )
        return payload

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        LOGGER.info("api client get job status request job_id=%s", job_id)
        response = await self._client.get(
            f"{self.base_url}/jobs/{job_id}",
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        LOGGER.info(
            "api client get job status response status_code=%s duration_ms=%.2f "
            "job_id=%s status=%s",
            getattr(response, "status_code", None),
            (time.perf_counter() - started_at) * 1000,
            job_id,
            payload.get("status"),
        )
        return payload

    async def get_job_result(self, job_id: str) -> bytes:
        started_at = time.perf_counter()
        LOGGER.info("api client get job result request job_id=%s", job_id)
        response = await self._client.get(
            f"{self.base_url}/jobs/{job_id}/result",
            timeout=60.0,
        )
        response.raise_for_status()
        LOGGER.info(
            "api client get job result response status_code=%s duration_ms=%.2f "
            "job_id=%s result_size_bytes=%s",
            getattr(response, "status_code", None),
            (time.perf_counter() - started_at) * 1000,
            job_id,
            len(response.content),
        )
        return response.content
