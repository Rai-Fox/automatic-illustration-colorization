from __future__ import annotations

import numpy as np
import pytest

from illustration_colorizer.models import (
    ColorizationRequest,
    MissingReferenceImageError,
)
from illustration_colorizer.models.cgan import ExampleBasedMangaCganModel
from illustration_colorizer.models.cobra import CobraModel
from illustration_colorizer.models.colorcomic import ColorComicReferenceModel


@pytest.mark.parametrize(
    "model",
    [
        ColorComicReferenceModel({"model_id": "colorcomic_reference"}),
        ExampleBasedMangaCganModel({"model_id": "cgan_reference"}),
        CobraModel({"model_id": "cobra"}),
    ],
)
def test_reference_models_require_reference(model) -> None:
    request = ColorizationRequest(input_image=np.zeros((4, 4, 3), dtype=np.uint8))

    with pytest.raises(MissingReferenceImageError):
        model.require_reference(request)


def test_cobra_requires_multiple_reference_list() -> None:
    model = CobraModel({"model_id": "cobra"})

    with pytest.raises(MissingReferenceImageError):
        model.get_references(
            ColorizationRequest(input_image=np.zeros((4, 4, 3), dtype=np.uint8))
        )
