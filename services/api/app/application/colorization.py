from __future__ import annotations

import json
import logging
import time
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from illustration_colorizer.models import (
    ColorizationRequest,
    ColorizationResult,
    MissingReferenceImageError,
)
from services.api.app.infrastructure.models import ModelManager

LOGGER = logging.getLogger(__name__)


def encode_png(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def parse_options(raw_options: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw_options is None or raw_options == "":
        return {}
    if isinstance(raw_options, dict):
        return dict(raw_options)
    parsed = json.loads(raw_options)
    if not isinstance(parsed, dict):
        raise ValueError("options must be a JSON object.")
    return parsed


class ColorizationService:
    def __init__(
        self,
        *,
        model_manager: ModelManager,
        enabled_models: tuple[str, ...],
        max_image_side: int,
    ) -> None:
        self.model_manager = model_manager
        self.enabled_models = enabled_models
        self.max_image_side = max_image_side

    def list_models(self) -> list[dict[str, Any]]:
        LOGGER.info(
            "listing colorization models enabled_models=%s",
            self.enabled_models,
        )
        return self.model_manager.list_model_infos(enabled_models=self.enabled_models)

    def ensure_model_allowed(self, model_id: str) -> None:
        LOGGER.info("validating model availability model_id=%s", model_id)
        self.model_manager.ensure_model_allowed(
            model_id,
            enabled_models=self.enabled_models,
        )

    def validate_image_bytes(self, image_bytes: bytes) -> None:
        LOGGER.info("validating image bytes size_bytes=%s", len(image_bytes))
        self._decode_image(image_bytes)

    def validate_model_inputs(
        self,
        *,
        model_id: str,
        reference_image_bytes: bytes | None = None,
        reference_images_bytes: list[bytes] | None = None,
    ) -> None:
        references_count = int(reference_image_bytes is not None) + len(
            reference_images_bytes or []
        )
        LOGGER.info(
            "validating model inputs model_id=%s references_count=%s",
            model_id,
            references_count,
        )
        model_config = self.model_manager.load_model_config(model_id)
        if bool(model_config.get("requires_reference", False)):
            if references_count == 0:
                raise MissingReferenceImageError(
                    f"{model_id} requires reference_image or reference_images."
                )

    def colorize(
        self,
        image_bytes: bytes,
        *,
        model_id: str,
        reference_image_bytes: bytes | None = None,
        reference_images_bytes: list[bytes] | None = None,
        seed: int | None = None,
        options: str | dict[str, Any] | None = None,
    ) -> ColorizationResult:
        started_at = time.perf_counter()
        parsed_options = parse_options(options)
        LOGGER.info(
            "colorization started model_id=%s input_size_bytes=%s "
            "references_count=%s seed_set=%s options_keys=%s",
            model_id,
            len(image_bytes),
            int(reference_image_bytes is not None) + len(reference_images_bytes or []),
            seed is not None,
            sorted(parsed_options),
        )
        self.ensure_model_allowed(model_id)
        self.validate_model_inputs(
            model_id=model_id,
            reference_image_bytes=reference_image_bytes,
            reference_images_bytes=reference_images_bytes,
        )

        input_image = self._decode_image(image_bytes)
        reference_image = (
            self._decode_image(reference_image_bytes)
            if reference_image_bytes
            else None
        )
        reference_images = [
            self._decode_image(reference_bytes)
            for reference_bytes in reference_images_bytes or []
        ]
        model = self.model_manager.get_model(model_id)
        result = model.colorize(
            ColorizationRequest(
                input_image=input_image,
                reference_image=reference_image,
                reference_images=reference_images,
                seed=seed,
                options=parsed_options,
            )
        )
        duration_ms = (time.perf_counter() - started_at) * 1000
        LOGGER.info(
            "colorization completed model_id=%s duration_ms=%.2f warnings_count=%s",
            result.model_id,
            duration_ms,
            len(result.warnings),
        )
        return result

    def colorize_to_png(
        self,
        image_bytes: bytes,
        *,
        model_id: str,
        reference_image_bytes: bytes | None = None,
        reference_images_bytes: list[bytes] | None = None,
        seed: int | None = None,
        options: str | dict[str, Any] | None = None,
    ) -> bytes:
        result = self.colorize(
            image_bytes,
            model_id=model_id,
            reference_image_bytes=reference_image_bytes,
            reference_images_bytes=reference_images_bytes,
            seed=seed,
            options=options,
        )
        return encode_png(result.image)

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            if max(rgb_image.size) > self.max_image_side:
                raise ValueError(
                    f"Image side is too large: {rgb_image.size}. "
                    f"Maximum side is {self.max_image_side}."
                )
            return np.asarray(rgb_image)
