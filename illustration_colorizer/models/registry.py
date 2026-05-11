from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from illustration_colorizer.models.base import ColorizationModel
from illustration_colorizer.models.passthrough import PassthroughColorizationModel
from shared.omegaconf import to_plain_mapping

ModelRegistryEntry = type[ColorizationModel] | str

MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "cgan_reference": "illustration_colorizer.models.cgan:ExampleBasedMangaCganModel",
    "cobra": "illustration_colorizer.models.cobra:CobraModel",
    "colorcomic_auto": "illustration_colorizer.models.colorcomic:ColorComicAutoModel",
    "colorcomic_reference": (
        "illustration_colorizer.models.colorcomic:ColorComicReferenceModel"
    ),
    "ddcolor": "illustration_colorizer.models.ddcolor:DDColorModel",
    "deoldify": "illustration_colorizer.models.deoldify:DeOldifyModel",
    "passthrough": PassthroughColorizationModel,
}


def _resolve_model_class(entry: ModelRegistryEntry) -> type[ColorizationModel]:
    if isinstance(entry, str):
        module_name, class_name = entry.split(":", maxsplit=1)
        module = import_module(module_name)
        return getattr(module, class_name)

    return entry


def create_model_from_config(
    config: Mapping[str, Any] | DictConfig,
    *,
    project_root: Path,
) -> ColorizationModel:
    data = to_plain_mapping(config)
    model_id = str(data["model_id"])
    model_class = MODEL_REGISTRY.get(model_id)
    if model_class is None:
        raise KeyError(f"Unknown model_id: {model_id}")

    data["project_root"] = str(project_root)
    return _resolve_model_class(model_class)(data)


def resolve_model_configs(
    models_config: Mapping[str, Any] | DictConfig,
    *,
    selected_models: list[str] | None,
) -> list[dict[str, Any]]:
    data = to_plain_mapping(models_config)

    if selected_models:
        resolved = []
        for model_name in selected_models:
            if model_name not in data:
                raise KeyError(f"Unknown configured model: {model_name}")
            resolved.append(to_plain_mapping(data[model_name]))
        return resolved

    return [
        to_plain_mapping(model_cfg)
        for model_cfg in data.values()
        if bool(to_plain_mapping(model_cfg).get("enabled", False))
    ]
