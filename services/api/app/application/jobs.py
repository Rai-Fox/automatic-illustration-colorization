from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.api.app.infrastructure.jobs import (
    JobStore,
    build_job_record,
    new_job_id,
    utc_now,
)
from services.api.app.infrastructure.storage import JobFileStorage

LOGGER = logging.getLogger(__name__)


class JobService:
    def __init__(
        self,
        *,
        store: JobStore,
        file_storage: JobFileStorage,
    ) -> None:
        self.store = store
        self.file_storage = file_storage

    async def create_job(
        self,
        *,
        model_id: str,
        image_bytes: bytes,
        reference_image_bytes: bytes | None,
        reference_images_bytes: list[bytes],
        seed: int | None,
        options: dict[str, Any],
        chat_id: int | None,
    ) -> dict[str, Any]:
        job_id = new_job_id()
        references_count = int(reference_image_bytes is not None) + len(
            reference_images_bytes
        )
        LOGGER.info(
            "creating job job_id=%s model_id=%s chat_id=%s input_size_bytes=%s "
            "references_count=%s seed_set=%s options_keys=%s",
            job_id,
            model_id,
            chat_id,
            len(image_bytes),
            references_count,
            seed is not None,
            sorted(options),
        )
        files = self.file_storage.save_job_inputs(
            job_id=job_id,
            image_bytes=image_bytes,
            reference_image_bytes=reference_image_bytes,
            reference_images_bytes=reference_images_bytes,
        )
        record = build_job_record(
            job_id=job_id,
            model_id=model_id,
            input_path=files.input_path,
            reference_paths=files.reference_paths,
            seed=seed,
            options=options,
            chat_id=chat_id,
        )
        await self.store.create(record)
        await self.store.enqueue(job_id)
        LOGGER.info(
            "job queued job_id=%s model_id=%s input_path=%s references_count=%s",
            job_id,
            model_id,
            files.input_path,
            references_count,
        )
        return record

    async def dequeue(self, *, timeout_seconds: int = 5) -> str | None:
        job_id = await self.store.dequeue(timeout_seconds=timeout_seconds)
        if job_id is not None:
            LOGGER.info("job dequeued job_id=%s", job_id)
        return job_id

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await self.store.get(job_id)

    def get_result_path(self, record: dict[str, Any]) -> Path | None:
        return self.file_storage.get_result_path(record)

    async def mark_running(self, job_id: str) -> None:
        LOGGER.info("marking job running job_id=%s", job_id)
        await self.store.update(
            job_id,
            {
                "status": "running",
                "started_at": utc_now(),
                "error": None,
            },
        )

    async def mark_succeeded(
        self,
        *,
        job_id: str,
        record: dict[str, Any],
        result_path: Path,
        warnings: list[str],
    ) -> None:
        finished_at = utc_now()
        LOGGER.info(
            "marking job succeeded job_id=%s result_path=%s warnings_count=%s",
            job_id,
            result_path,
            len(warnings),
        )
        await self.store.update(
            job_id,
            {
                "status": "succeeded",
                "finished_at": finished_at,
                "result_path": str(result_path),
                "warnings": warnings,
                "error": None,
            },
        )
        await self.store.publish_event(
            {
                "job_id": job_id,
                "status": "succeeded",
                "chat_id": record.get("chat_id"),
                "model_id": record.get("model_id"),
                "finished_at": finished_at,
            }
        )
        LOGGER.info("published job succeeded event job_id=%s", job_id)

    async def mark_failed(
        self,
        *,
        job_id: str,
        record: dict[str, Any],
        error: str,
    ) -> None:
        finished_at = utc_now()
        LOGGER.info("marking job failed job_id=%s error=%s", job_id, error)
        await self.store.update(
            job_id,
            {
                "status": "failed",
                "finished_at": finished_at,
                "error": error,
            },
        )
        await self.store.publish_event(
            {
                "job_id": job_id,
                "status": "failed",
                "chat_id": record.get("chat_id"),
                "model_id": record.get("model_id"),
                "finished_at": finished_at,
                "error": error,
            }
        )
        LOGGER.info("published job failed event job_id=%s", job_id)
