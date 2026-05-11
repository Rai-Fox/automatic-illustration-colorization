from __future__ import annotations

import numpy as np
import pytest

from illustration_colorizer.models import (
    ColorizationRequest,
    MissingReferenceImageError,
)
from illustration_colorizer.models.passthrough import PassthroughColorizationModel


def test_colorization_request_accepts_reference_list_and_options() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    reference = np.ones((2, 3, 3), dtype=np.uint8)

    request = ColorizationRequest(
        input_image=image,
        reference_images=[reference],
        sample_id="sample",
        seed=123,
        options={"size": 256},
    )

    assert request.reference_images == [reference]
    assert request.seed == 123
    assert request.options["size"] == 256


def test_require_reference_raises_custom_error() -> None:
    model = PassthroughColorizationModel({"model_id": "passthrough"})

    with pytest.raises(MissingReferenceImageError):
        model.require_reference(ColorizationRequest(input_image=np.zeros((2, 2, 3))))


def test_passthrough_returns_rgb_uint8_result() -> None:
    model = PassthroughColorizationModel({"model_id": "passthrough"})
    result = model.colorize(
        ColorizationRequest(input_image=np.ones((2, 2), dtype=np.float32))
    )

    assert result.image.dtype == np.uint8
    assert result.image.shape == (2, 2, 3)
    assert result.model_id == "passthrough"
    assert result.warnings


def test_default_colorize_batch_uses_single_request_colorize() -> None:
    model = PassthroughColorizationModel({"model_id": "passthrough"})

    results = model.colorize_batch(
        [
            ColorizationRequest(input_image=np.zeros((2, 2), dtype=np.float32)),
            ColorizationRequest(input_image=np.ones((2, 2), dtype=np.float32)),
        ]
    )

    assert len(results) == 2
    assert all(result.model_id == "passthrough" for result in results)
