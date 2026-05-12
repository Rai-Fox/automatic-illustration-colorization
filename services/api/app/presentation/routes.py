from __future__ import annotations

import base64
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from illustration_colorizer.models import (
    ColorizationModelError,
    MissingReferenceImageError,
    ModelBackendUnavailableError,
)
from services.api.app.application.colorization import (
    ColorizationService,
    encode_png,
    parse_options,
)
from services.api.app.application.jobs import JobService
from services.api.app.container import (
    get_colorization_service,
    get_job_service,
    get_settings,
)
from services.api.app.core.config import ApiSettings
from services.api.app.schemas import (
    ColorizeResponse,
    JobCreateResponse,
    JobStatusResponse,
    ModelInfo,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)

SettingsDep = Annotated[ApiSettings, Depends(get_settings)]
ColorizationServiceDep = Annotated[
    ColorizationService,
    Depends(get_colorization_service),
]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models", response_model=list[ModelInfo])
def models(colorization_service: ColorizationServiceDep) -> list[ModelInfo]:
    model_infos = colorization_service.list_models()
    enabled_count = sum(1 for model in model_infos if model.get("enabled"))
    LOGGER.info(
        "listed models total=%s enabled=%s",
        len(model_infos),
        enabled_count,
    )
    return [ModelInfo(**model) for model in model_infos]


async def _read_upload(
    upload: UploadFile,
    *,
    field_name: str,
    settings: ApiSettings,
    colorization_service: ColorizationService,
) -> bytes:
    content = await upload.read()
    LOGGER.info(
        "read upload field=%s filename=%s content_type=%s size_bytes=%s",
        field_name,
        upload.filename,
        upload.content_type,
        len(content),
    )
    if not content:
        LOGGER.warning("empty upload field=%s filename=%s", field_name, upload.filename)
        raise HTTPException(status_code=400, detail=f"Empty {field_name}")
    if len(content) > settings.max_upload_bytes:
        LOGGER.warning(
            "upload rejected field=%s reason=too_large size_bytes=%s max_bytes=%s",
            field_name,
            len(content),
            settings.max_upload_bytes,
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"{field_name} is too large. Maximum size is "
                f"{settings.max_upload_bytes} bytes."
            ),
        )
    try:
        colorization_service.validate_image_bytes(content)
    except (OSError, ValueError) as exc:
        LOGGER.warning(
            "upload rejected field=%s reason=invalid_image error=%s",
            field_name,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return content


def _handle_colorization_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MissingReferenceImageError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (OSError, ValueError, FileNotFoundError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ModelBackendUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ColorizationModelError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/colorize", response_model=ColorizeResponse)
async def colorize(
    file: Annotated[UploadFile, File(...)],
    settings: SettingsDep,
    colorization_service: ColorizationServiceDep,
    model_id: Annotated[str | None, Form()] = None,
    reference: Annotated[UploadFile | None, File()] = None,
    references: Annotated[list[UploadFile] | None, File()] = None,
    seed: Annotated[int | None, Form()] = None,
    options: Annotated[str | None, Form()] = None,
) -> ColorizeResponse:
    selected_model_id = model_id or settings.model_id
    LOGGER.info(
        "sync colorize requested model_id=%s seed_set=%s has_reference=%s "
        "references_count=%s options_provided=%s",
        selected_model_id,
        seed is not None,
        reference is not None,
        len(references or []),
        bool(options),
    )
    try:
        content = await _read_upload(
            file,
            field_name="file",
            settings=settings,
            colorization_service=colorization_service,
        )
        reference_content = (
            await _read_upload(
                reference,
                field_name="reference",
                settings=settings,
                colorization_service=colorization_service,
            )
            if reference is not None
            else None
        )
        reference_contents = [
            await _read_upload(
                reference_file,
                field_name="references",
                settings=settings,
                colorization_service=colorization_service,
            )
            for reference_file in references or []
        ]
        result = await run_in_threadpool(
            colorization_service.colorize,
            content,
            model_id=selected_model_id,
            reference_image_bytes=reference_content,
            reference_images_bytes=reference_contents,
            seed=seed,
            options=options,
        )
    except Exception as exc:
        LOGGER.warning(
            "sync colorize failed model_id=%s error=%s",
            selected_model_id,
            exc,
        )
        raise _handle_colorization_error(exc) from exc

    output_b64 = base64.b64encode(encode_png(result.image)).decode("ascii")
    LOGGER.info(
        "sync colorize completed model_id=%s warnings_count=%s metadata_keys=%s",
        result.model_id,
        len(result.warnings),
        sorted(result.metadata),
    )
    return ColorizeResponse(
        image_base64=output_b64,
        model=result.model_id,
        warnings=result.warnings,
        metadata=result.metadata,
    )


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    file: Annotated[UploadFile, File(...)],
    settings: SettingsDep,
    colorization_service: ColorizationServiceDep,
    job_service: JobServiceDep,
    model_id: Annotated[str | None, Form()] = None,
    chat_id: Annotated[int | None, Form()] = None,
    reference: Annotated[UploadFile | None, File()] = None,
    references: Annotated[list[UploadFile] | None, File()] = None,
    seed: Annotated[int | None, Form()] = None,
    options: Annotated[str | None, Form()] = None,
) -> JobCreateResponse:
    selected_model_id = model_id or settings.model_id
    LOGGER.info(
        "job create requested model_id=%s chat_id=%s seed_set=%s "
        "has_reference=%s references_count=%s options_provided=%s",
        selected_model_id,
        chat_id,
        seed is not None,
        reference is not None,
        len(references or []),
        bool(options),
    )
    try:
        content = await _read_upload(
            file,
            field_name="file",
            settings=settings,
            colorization_service=colorization_service,
        )
        reference_content = (
            await _read_upload(
                reference,
                field_name="reference",
                settings=settings,
                colorization_service=colorization_service,
            )
            if reference is not None
            else None
        )
        reference_contents = [
            await _read_upload(
                reference_file,
                field_name="references",
                settings=settings,
                colorization_service=colorization_service,
            )
            for reference_file in references or []
        ]
        parsed_options = parse_options(options)
        colorization_service.ensure_model_allowed(selected_model_id)
        colorization_service.validate_model_inputs(
            model_id=selected_model_id,
            reference_image_bytes=reference_content,
            reference_images_bytes=reference_contents,
        )
        record = await job_service.create_job(
            model_id=selected_model_id,
            image_bytes=content,
            reference_image_bytes=reference_content,
            reference_images_bytes=reference_contents,
            seed=seed,
            options=parsed_options,
            chat_id=chat_id,
        )
    except Exception as exc:
        LOGGER.warning(
            "job create failed model_id=%s chat_id=%s error=%s",
            selected_model_id,
            chat_id,
            exc,
        )
        raise _handle_colorization_error(exc) from exc

    LOGGER.info(
        "job created job_id=%s status=%s model_id=%s chat_id=%s",
        record["job_id"],
        record["status"],
        record["model_id"],
        record.get("chat_id"),
    )
    return JobCreateResponse(job_id=str(record["job_id"]), status=str(record["status"]))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, job_service: JobServiceDep) -> JobStatusResponse:
    record = await job_service.get_job(job_id)
    if record is None:
        LOGGER.warning("job status requested for missing job job_id=%s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    LOGGER.info(
        "job status returned job_id=%s status=%s model_id=%s",
        job_id,
        record.get("status"),
        record.get("model_id"),
    )
    return JobStatusResponse(**record)


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str, job_service: JobServiceDep) -> FileResponse:
    record = await job_service.get_job(job_id)
    if record is None:
        LOGGER.warning("job result requested for missing job job_id=%s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    if record.get("status") != "succeeded":
        LOGGER.warning(
            "job result requested before completion job_id=%s status=%s",
            job_id,
            record.get("status"),
        )
        raise HTTPException(status_code=409, detail="Job is not completed")
    result_path = job_service.get_result_path(record)
    if result_path is None or not result_path.exists():
        LOGGER.warning("job result missing job_id=%s path=%s", job_id, result_path)
        raise HTTPException(status_code=404, detail="Result not found")
    LOGGER.info("job result returned job_id=%s path=%s", job_id, result_path)
    return FileResponse(result_path, media_type="image/png", filename="colorized.png")
