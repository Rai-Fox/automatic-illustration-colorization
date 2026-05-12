from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from illustration_colorizer.models import (
    MODEL_REGISTRY,
    ColorizationModel,
    create_model_from_config,
)
from shared.hydra import load_config
from shared.omegaconf import to_plain_mapping

LOGGER = logging.getLogger(__name__)


class ModelManager:
    def __init__(
        self,
        *,
        project_root: Path,
        model_path: str,
        device: str,
    ) -> None:
        self.project_root = project_root
        self.model_path = model_path
        self.device = device
        self._cache: dict[tuple[str, str, str], ColorizationModel] = {}

    def load_model_config(self, model_id: str) -> dict[str, Any]:
        LOGGER.debug("loading model config model_id=%s", model_id)
        config = load_config(self.project_root / "illustration_colorizer" / "conf")
        models = to_plain_mapping(config.models)
        if model_id not in models:
            return {"model_id": model_id}
        return to_plain_mapping(models[model_id])

    def ensure_model_allowed(
        self,
        model_id: str,
        *,
        enabled_models: tuple[str, ...] = (),
    ) -> None:
        if model_id not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_id: {model_id}")
        if enabled_models and model_id not in enabled_models:
            raise KeyError(f"Model is not enabled: {model_id}")

    def list_model_infos(
        self,
        *,
        enabled_models: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        model_infos: list[dict[str, Any]] = []
        for model_id in sorted(MODEL_REGISTRY):
            model_config = self.load_model_config(model_id)
            model_infos.append(
                {
                    "model_id": model_id,
                    "enabled": not enabled_models or model_id in enabled_models,
                    "requires_reference": bool(
                        model_config.get("requires_reference", False)
                    ),
                    "supports_multiple_references": bool(
                        model_config.get("supports_multiple_references", False)
                    ),
                    "supports_cpu": bool(model_config.get("supports_cpu", True)),
                }
            )
        return model_infos

    def get_model(
        self,
        model_id: str,
        *,
        config_overrides: Mapping[str, Any] | None = None,
        ) -> ColorizationModel:
        cache_key = (model_id, self.model_path, self.device)
        cached_model = self._cache.get(cache_key)
        if cached_model is not None:
            LOGGER.info(
                "model cache hit model_id=%s device=%s model_path=%s",
                model_id,
                self.device,
                self.model_path or "<config>",
            )
            return cached_model

        started_at = time.perf_counter()
        LOGGER.info(
            "loading model model_id=%s device=%s model_path=%s",
            model_id,
            self.device,
            self.model_path or "<config>",
        )
        model_config = self.load_model_config(model_id)
        model_config["device"] = self.device
        if self.model_path:
            model_config["checkpoint_path"] = self.model_path
        if config_overrides:
            model_config.update(config_overrides)

        model = create_model_from_config(model_config, project_root=self.project_root)
        model.load()
        self._cache[cache_key] = model
        duration_ms = (time.perf_counter() - started_at) * 1000
        LOGGER.info(
            "model loaded model_id=%s device=%s duration_ms=%.2f",
            model_id,
            self.device,
            duration_ms,
        )
        return model
