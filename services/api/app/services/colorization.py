from __future__ import annotations

from pathlib import Path
from typing import Any

from services.api.app.application.colorization import (
    ColorizationService,
    encode_png,
    parse_options,
)
from services.api.app.infrastructure.models import ModelManager
from shared.paths import get_project_root

__all__ = [
    "ColorizationService",
    "colorize_image",
    "encode_png",
    "ensure_model_allowed",
    "list_model_infos",
    "parse_options",
    "run_colorization",
    "validate_image_bytes",
    "validate_model_inputs",
]


def _default_service(
    *,
    model_path: str,
    device: str,
    max_image_side: int,
    enabled_models: tuple[str, ...],
) -> ColorizationService:
    project_root = get_project_root(Path(__file__), levels_up=4)
    model_manager = ModelManager(
        project_root=project_root,
        model_path=model_path,
        device=device,
    )
    return ColorizationService(
        model_manager=model_manager,
        enabled_models=enabled_models,
        max_image_side=max_image_side,
    )


def ensure_model_allowed(model_id: str, enabled_models: tuple[str, ...] = ()) -> None:
    service = _default_service(
        model_path="",
        device="cpu",
        max_image_side=4096,
        enabled_models=enabled_models,
    )
    service.ensure_model_allowed(model_id)


def validate_image_bytes(
    image_bytes: bytes,
    *,
    max_image_side: int | None = None,
) -> None:
    service = _default_service(
        model_path="",
        device="cpu",
        max_image_side=max_image_side or 4096,
        enabled_models=(),
    )
    service.validate_image_bytes(image_bytes)


def validate_model_inputs(
    *,
    model_id: str,
    reference_image_bytes: bytes | None = None,
    reference_images_bytes: list[bytes] | None = None,
) -> None:
    service = _default_service(
        model_path="",
        device="cpu",
        max_image_side=4096,
        enabled_models=(),
    )
    service.validate_model_inputs(
        model_id=model_id,
        reference_image_bytes=reference_image_bytes,
        reference_images_bytes=reference_images_bytes,
    )


def list_model_infos(enabled_models: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    service = _default_service(
        model_path="",
        device="cpu",
        max_image_side=4096,
        enabled_models=enabled_models,
    )
    return service.list_models()


def run_colorization(
    image_bytes: bytes,
    *,
    model_id: str,
    model_path: str,
    device: str,
    reference_image_bytes: bytes | None = None,
    reference_images_bytes: list[bytes] | None = None,
    seed: int | None = None,
    options: str | dict[str, Any] | None = None,
    max_image_side: int | None = None,
) -> Any:
    service = _default_service(
        model_path=model_path,
        device=device,
        max_image_side=max_image_side or 4096,
        enabled_models=(),
    )
    return service.colorize(
        image_bytes,
        model_id=model_id,
        reference_image_bytes=reference_image_bytes,
        reference_images_bytes=reference_images_bytes,
        seed=seed,
        options=options,
    )


def colorize_image(
    image_bytes: bytes,
    *,
    model_id: str,
    model_path: str,
    device: str,
    reference_image_bytes: bytes | None = None,
    reference_images_bytes: list[bytes] | None = None,
    seed: int | None = None,
    options: str | dict[str, Any] | None = None,
    max_image_side: int | None = None,
) -> bytes:
    service = _default_service(
        model_path=model_path,
        device=device,
        max_image_side=max_image_side or 4096,
        enabled_models=(),
    )
    return service.colorize_to_png(
        image_bytes,
        model_id=model_id,
        reference_image_bytes=reference_image_bytes,
        reference_images_bytes=reference_images_bytes,
        seed=seed,
        options=options,
    )
