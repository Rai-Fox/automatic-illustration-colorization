from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from services.api.app.application.colorization import ColorizationService
from services.api.app.application.jobs import JobService
from services.api.app.core.config import ApiSettings
from services.api.app.infrastructure.jobs import JobStore, RedisJobStore
from services.api.app.infrastructure.models import ModelManager
from services.api.app.infrastructure.storage import JobFileStorage
from shared.paths import get_project_root


@dataclass(frozen=True)
class AppContainer:
    settings: ApiSettings
    model_manager: ModelManager
    colorization_service: ColorizationService
    job_store: JobStore
    file_storage: JobFileStorage
    job_service: JobService

    @classmethod
    def from_settings(
        cls,
        settings: ApiSettings,
        *,
        job_store: JobStore | None = None,
    ) -> AppContainer:
        project_root = get_project_root(Path(__file__), levels_up=3)
        model_manager = ModelManager(
            project_root=project_root,
            model_path=settings.model_path,
            device=settings.device,
        )
        file_storage = JobFileStorage(settings.service_storage_dir)
        resolved_job_store = job_store or RedisJobStore(
            settings.redis_url,
            ttl_seconds=settings.job_ttl_seconds,
        )
        colorization_service = ColorizationService(
            model_manager=model_manager,
            enabled_models=settings.enabled_models,
            max_image_side=settings.max_image_size,
        )
        job_service = JobService(
            store=resolved_job_store,
            file_storage=file_storage,
        )
        return cls(
            settings=settings,
            model_manager=model_manager,
            colorization_service=colorization_service,
            job_store=resolved_job_store,
            file_storage=file_storage,
            job_service=job_service,
        )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_settings(request: Request) -> ApiSettings:
    return get_container(request).settings


def get_colorization_service(request: Request) -> ColorizationService:
    return get_container(request).colorization_service


def get_job_service(request: Request) -> JobService:
    return get_container(request).job_service


def get_job_store(request: Request) -> JobStore:
    return get_container(request).job_store


def get_file_storage(request: Request) -> JobFileStorage:
    return get_container(request).file_storage
