from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from illustration_colorizer.models.base import (
    ColorizationModel,
    ColorizationRequest,
    ColorizationResult,
)
from illustration_colorizer.models.local_assets import ensure_hf_snapshot_dir
from illustration_colorizer.models.runtime import (
    bgr_to_rgb,
    bgr_uint8,
    require_loaded,
    result,
)
from shared.paths import ensure_on_sys_path, resolve_from_root

LOGGER = logging.getLogger(__name__)


class DDColorModel(ColorizationModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._pipeline: Any | None = None
        self._project_root = Path(self.config["project_root"])

    def load(self) -> None:
        if self._pipeline is not None:
            return

        repo_path = resolve_from_root(self._project_root, self.config.get("repo_path"))
        if repo_path is None or not repo_path.exists():
            raise FileNotFoundError(f"DDColor repository not found: {repo_path}")

        ensure_on_sys_path(repo_path)

        import torch
        from ddcolor import ColorizationPipeline, DDColor
        from ddcolor.pipeline import build_ddcolor_model

        device_name = str(self.config.get("device", "cpu"))
        device = torch.device(
            "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
        )

        checkpoint_path_raw = self.config.get("checkpoint_path")
        if checkpoint_path_raw:
            checkpoint_path = resolve_from_root(self._project_root, checkpoint_path_raw)
            if checkpoint_path is None or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"DDColor checkpoint not found: {checkpoint_path_raw}"
                )
            model = build_ddcolor_model(
                DDColor,
                model_path=str(checkpoint_path),
                input_size=int(self.config.get("input_size", 512)),
                model_size=str(self.config.get("model_size", "tiny")),
                decoder_type=str(
                    self.config.get("decoder_type", "MultiScaleColorDecoder")
                ),
                device=device,
            )
        else:
            pretrained_id = self.config.get("pretrained_id")
            if not pretrained_id:
                raise ValueError(
                    "DDColor requires either checkpoint_path or pretrained_id."
                )
            pretrained_local_dir = self.config.get("pretrained_local_dir")
            if not pretrained_local_dir:
                raise ValueError(
                    "DDColor requires pretrained_local_dir when checkpoint_path "
                    "is not set."
                )
            local_snapshot_dir = ensure_hf_snapshot_dir(
                project_root=self._project_root,
                raw_path=str(pretrained_local_dir),
                repo_id=str(pretrained_id),
                allow_download=bool(self.config.get("allow_download", True)),
            )
            bin_checkpoint_path = local_snapshot_dir / "pytorch_model.bin"
            safetensors_checkpoint_path = local_snapshot_dir / "model.safetensors"

            if bin_checkpoint_path.exists():
                LOGGER.info(
                    "Loading DDColor weights from local checkpoint %s",
                    bin_checkpoint_path,
                )
                model = build_ddcolor_model(
                    DDColor,
                    model_path=str(bin_checkpoint_path),
                    input_size=int(self.config.get("input_size", 512)),
                    model_size=str(self.config.get("model_size", "tiny")),
                    decoder_type=str(
                        self.config.get("decoder_type", "MultiScaleColorDecoder")
                    ),
                    device=device,
                )
            elif safetensors_checkpoint_path.exists():
                raise FileNotFoundError(
                    "DDColor local snapshot contains model.safetensors, but the "
                    "current wrapper expects pytorch_model.bin. Re-download the "
                    "snapshot or add a compatible loader."
                )
            else:
                raise FileNotFoundError(
                    "DDColor local snapshot does not contain pytorch_model.bin. "
                    f"Checked: {local_snapshot_dir}"
                )

        self._pipeline = ColorizationPipeline(
            model,
            input_size=int(self.config.get("input_size", 512)),
            device=device,
        )

    def unload(self) -> None:
        self._pipeline = None

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        pipeline = require_loaded(self._pipeline, self.model_id)

        start_time = time.perf_counter()
        output_bgr = pipeline.process(bgr_uint8(request.input_image))
        return result(
            image=bgr_to_rgb(output_bgr),
            model_id=self.model_id,
            start_time=start_time,
        )
