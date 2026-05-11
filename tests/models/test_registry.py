from __future__ import annotations

from pathlib import Path

import pytest

from illustration_colorizer.models import MODEL_REGISTRY, create_model_from_config
from illustration_colorizer.models.base import ColorizationModel


@pytest.mark.parametrize(
    "model_id",
    [
        "ddcolor",
        "deoldify",
        "colorcomic_auto",
        "colorcomic_reference",
        "cgan_reference",
        "cobra",
        "passthrough",
    ],
)
def test_registry_contains_supported_models(model_id: str) -> None:
    assert model_id in MODEL_REGISTRY


def test_create_passthrough_model_from_config() -> None:
    model = create_model_from_config(
        {"model_id": "passthrough"},
        project_root=Path.cwd(),
    )

    assert isinstance(model, ColorizationModel)
    assert model.model_id == "passthrough"


def test_unknown_model_id_raises_key_error() -> None:
    with pytest.raises(KeyError):
        create_model_from_config({"model_id": "missing"}, project_root=Path.cwd())
