from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from illustration_colorizer.models.base import (
    ColorizationModel,
    ColorizationRequest,
    ColorizationResult,
    ModelBackendUnavailableError,
)
from illustration_colorizer.models.runtime import (
    isolated_vendor_imports,
    project_path,
    request_option,
    require_loaded,
    result,
    rgb_uint8,
)


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ExampleBasedMangaCganModel(ColorizationModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._color_encoder: Any | None = None
        self._color_unet: Any | None = None
        self._torch: Any | None = None
        self._functional: Any | None = None
        self._color: Any | None = None
        self._device: Any | None = None

    @property
    def requires_reference(self) -> bool:
        return True

    def load(self) -> None:
        if self._color_encoder is not None and self._color_unet is not None:
            return

        repo_path = project_path(self.config, "repo_path")
        checkpoint_path = project_path(self.config, "checkpoint_path")
        assert repo_path is not None and checkpoint_path is not None

        try:
            import torch
            import torch.nn.functional as functional
            from skimage import color
        except ImportError as exc:
            raise ModelBackendUnavailableError(
                f"cGAN dependencies are unavailable: {exc}"
            ) from exc

        device_name = str(self.config.get("device", "cpu"))
        device = torch.device(
            "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
        )

        try:
            with isolated_vendor_imports(repo_path):
                from models import ColorEncoder, ColorUNet
        except ImportError as exc:
            raise ModelBackendUnavailableError(
                f"cGAN backend is unavailable: {exc}"
            ) from exc

        original_torch_load = torch.load

        def compat_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            kwargs.setdefault("map_location", "cpu")
            return original_torch_load(*args, **kwargs)

        with _working_directory(repo_path):
            torch.load = compat_torch_load
            try:
                checkpoint = torch.load(
                    str(checkpoint_path),
                    map_location=device,
                )
                color_encoder = ColorEncoder().to(device)
                color_encoder.load_state_dict(checkpoint["colorEncoder"])
                color_encoder.eval()

                color_unet = ColorUNet().to(device)
                color_unet.load_state_dict(checkpoint["colorUNet"])
                color_unet.eval()
            finally:
                torch.load = original_torch_load

        self._color_encoder = color_encoder
        self._color_unet = color_unet
        self._torch = torch
        self._functional = functional
        self._color = color
        self._device = device

    def unload(self) -> None:
        self._color_encoder = None
        self._color_unet = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _preprocess(self, image: np.ndarray) -> tuple[Any, Any]:
        torch = require_loaded(self._torch, self.model_id)
        color = require_loaded(self._color, self.model_id)
        rgb = rgb_uint8(image)
        lab = color.rgb2lab(rgb)
        lab[:, :, 0:1] = lab[:, :, 0:1] - 50.0
        img_tensor = torch.from_numpy(
            rgb.astype("float32").transpose(2, 0, 1)
        ).unsqueeze(0)
        lab_tensor = torch.from_numpy(
            lab.astype("float32").transpose(2, 0, 1)
        ).unsqueeze(0)
        return img_tensor.to(self._device), lab_tensor.to(self._device)

    def _lab_to_rgb(self, image_lab: Any) -> np.ndarray:
        color = require_loaded(self._color, self.model_id)
        lab = image_lab.detach().cpu()
        lab_l = lab[:, :1, :, :] + 50.0
        pred_lab = self._torch.cat((lab_l, lab[:, 1:, :, :]), 1)[0].numpy()
        return (
            np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0.0, 1.0) * 255.0
        ).astype("uint8")

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        color_encoder = require_loaded(self._color_encoder, self.model_id)
        color_unet = require_loaded(self._color_unet, self.model_id)
        torch = require_loaded(self._torch, self.model_id)
        functional = require_loaded(self._functional, self.model_id)
        reference = self.require_reference(request)
        img_size = int(request_option(request, "size", self.config.get("size", 256)))

        input_rgb = rgb_uint8(request.input_image)
        height, width = input_rgb.shape[:2]
        input_tensor, input_lab = self._preprocess(input_rgb)
        reference_tensor, _ = self._preprocess(reference)

        start_time = time.perf_counter()
        with torch.no_grad():
            ref_resized = functional.interpolate(
                reference_tensor / 255.0,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
            input_l_resized = functional.interpolate(
                input_lab[:, :1, :, :] / 50.0,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
            color_vector = color_encoder(ref_resized)
            fake_ab = color_unet((input_l_resized, color_vector))
            fake_ab = functional.interpolate(
                fake_ab * 110.0,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            fake_lab = torch.cat((input_lab[:, :1, :, :], fake_ab), 1)
            output = self._lab_to_rgb(fake_lab)

        return result(
            image=output,
            model_id=self.model_id,
            start_time=start_time,
            metadata={"size": img_size},
        )
