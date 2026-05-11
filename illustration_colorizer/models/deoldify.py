from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from illustration_colorizer.models.base import (
    ColorizationModel,
    ColorizationRequest,
    ColorizationResult,
)
from illustration_colorizer.models.local_assets import ensure_hf_snapshot_dir
from illustration_colorizer.models.runtime import require_loaded, result
from shared.images import pil_from_numpy
from shared.paths import ensure_on_sys_path, resolve_from_root

LOGGER = logging.getLogger(__name__)


class DeOldifyModel(ColorizationModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._colorizer: Any | None = None
        self._project_root = Path(self.config["project_root"])
        self._previous_cuda_visible_devices: str | None = None

    def load(self) -> None:
        if self._colorizer is not None:
            return

        import torch

        repo_path = resolve_from_root(self._project_root, self.config.get("repo_path"))
        if repo_path is None or not repo_path.exists():
            raise FileNotFoundError(f"DeOldify repository not found: {repo_path}")
        ensure_on_sys_path(repo_path)
        self._previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        from deoldify.filters import ColorizerFilter, MasterFilter
        from deoldify.generators import gen_inference_deep, gen_inference_wide
        from fastai.torch_core import defaults

        root_folder_raw = self.config.get("root_folder")
        if root_folder_raw:
            root_folder = ensure_hf_snapshot_dir(
                project_root=self._project_root,
                raw_path=str(root_folder_raw),
                repo_id=str(self.config.get("hf_repo_id", "leonelhs/deoldify")),
                allow_download=bool(self.config.get("allow_download", True)),
            )
        else:
            raise ValueError("DeOldify requires a configured root_folder.")

        if root_folder is None or not root_folder.exists():
            raise FileNotFoundError(
                f"DeOldify root folder not found: {root_folder_raw}"
            )

        requested_device = str(self.config.get("device", "cpu")).strip().lower()
        effective_device = (
            "cuda"
            if requested_device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        if requested_device == "cuda" and effective_device != "cuda":
            LOGGER.warning(
                "Requested device=%s for DeOldify, but CUDA is unavailable. "
                "Falling back to cpu.",
                requested_device,
            )
        torch_device = (
            torch.device("cuda:0")
            if effective_device == "cuda"
            else torch.device("cpu")
        )
        defaults.device = torch_device
        if effective_device == "cuda":
            torch.cuda.set_device(0)

        original_torch_load = torch.load

        def compat_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            kwargs.setdefault("map_location", torch_device)
            return original_torch_load(*args, **kwargs)

        torch.load = compat_torch_load
        try:
            artistic = bool(self.config.get("artistic", True))
            weights_name = (
                str(self.config.get("weights_name", "ColorizeArtistic_gen"))
                if artistic
                else str(self.config.get("weights_name", "ColorizeStable_gen"))
            )
            learn = (
                gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
                if artistic
                else gen_inference_wide(
                    root_folder=root_folder, weights_name=weights_name
                )
            )
            self._colorizer = MasterFilter(
                [ColorizerFilter(learn=learn)],
                render_factor=int(self.config.get("render_factor", 25)),
            )
        finally:
            torch.load = original_torch_load
            if self._previous_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = (
                    self._previous_cuda_visible_devices
                )

    def unload(self) -> None:
        self._colorizer = None
        self._previous_cuda_visible_devices = None

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        colorizer = require_loaded(self._colorizer, self.model_id)

        start_time = time.perf_counter()
        source = pil_from_numpy(request.input_image)
        output = colorizer.filter(
            source,
            source,
            render_factor=int(self.config.get("render_factor", 25)),
            post_process=True,
        )

        return result(
            image=np.asarray(output.convert("RGB")),
            model_id=self.model_id,
            start_time=start_time,
        )
