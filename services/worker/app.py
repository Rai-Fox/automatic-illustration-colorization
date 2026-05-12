from __future__ import annotations

import asyncio
import logging

from illustration_colorizer.models import ColorizationModelError, ColorizationResult
from services.api.app.application.colorization import (
    ColorizationService,
    encode_png,
)
from services.api.app.application.jobs import JobService
from services.api.app.container import AppContainer
from services.api.app.core.config import ApiSettings, load_settings
from services.api.app.infrastructure.jobs import JobStore
from services.api.app.infrastructure.storage import JobFileStorage

LOGGER = logging.getLogger(__name__)


class ColorizationWorker:
    def __init__(
        self,
        *,
        job_service: JobService,
        colorization_service: ColorizationService,
        file_storage: JobFileStorage,
    ) -> None:
        self.job_service = job_service
        self.colorization_service = colorization_service
        self.file_storage = file_storage

    @classmethod
    def from_settings(
        cls,
        settings: ApiSettings | None = None,
        *,
        job_store: JobStore | None = None,
    ) -> ColorizationWorker:
        resolved_settings = settings or load_settings()
        LOGGER.info(
            "building worker dependencies device=%s enabled_models=%s storage_dir=%s",
            resolved_settings.device,
            resolved_settings.enabled_models,
            resolved_settings.service_storage_dir,
        )
        container = AppContainer.from_settings(
            resolved_settings,
            job_store=job_store,
        )
        return cls(
            job_service=container.job_service,
            colorization_service=container.colorization_service,
            file_storage=container.file_storage,
        )

    def _colorize_record(self, record: dict[str, object]) -> ColorizationResult:
        LOGGER.info(
            "worker running colorization job_id=%s model_id=%s references_count=%s",
            record.get("job_id"),
            record.get("model_id"),
            len(record.get("reference_paths", [])),  # type: ignore[arg-type]
        )
        return self.colorization_service.colorize(
            self.file_storage.read_input(record),
            model_id=str(record["model_id"]),
            reference_images_bytes=self.file_storage.read_references(record),
            seed=record.get("seed"),
            options=record.get("options") or {},
        )

    async def process_job(self, job_id: str) -> None:
        LOGGER.info("worker processing job job_id=%s", job_id)
        record = await self.job_service.get_job(job_id)
        if record is None:
            LOGGER.warning("Skipping missing job %s", job_id)
            return

        await self.job_service.mark_running(job_id)
        LOGGER.info(
            "worker marked job running job_id=%s model_id=%s",
            job_id,
            record.get("model_id"),
        )

        try:
            result = await asyncio.to_thread(self._colorize_record, record)
            result_path = self.file_storage.write_result(
                record,
                encode_png(result.image),
            )
            LOGGER.info(
                "worker colorization finished job_id=%s result_path=%s",
                job_id,
                result_path,
            )
            await self.job_service.mark_succeeded(
                job_id=job_id,
                record=record,
                result_path=result_path,
                warnings=result.warnings,
            )
        except (
            ColorizationModelError,
            FileNotFoundError,
            KeyError,
            OSError,
            ValueError,
        ) as exc:
            LOGGER.exception("Colorization job %s failed", job_id)
            await self.job_service.mark_failed(
                job_id=job_id,
                record=record,
                error=str(exc),
            )
        except Exception as exc:
            LOGGER.exception("Unexpected colorization job %s failure", job_id)
            await self.job_service.mark_failed(
                job_id=job_id,
                record=record,
                error=str(exc),
            )

    async def run_once(self, *, timeout_seconds: int = 5) -> bool:
        job_id = await self.job_service.dequeue(timeout_seconds=timeout_seconds)
        if job_id is None:
            return False
        LOGGER.info("worker dequeued job job_id=%s", job_id)
        await self.process_job(job_id)
        return True

    async def run_forever(self) -> None:
        LOGGER.info("worker started polling for colorization jobs")
        while True:
            await self.run_once(timeout_seconds=5)
