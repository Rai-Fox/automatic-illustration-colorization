from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from illustration_colorizer.models.base import (
    ColorizationModel,
    ColorizationRequest,
    ColorizationResult,
    ModelBackendUnavailableError,
)
from illustration_colorizer.models.runtime import (
    bgr_to_rgb,
    isolated_vendor_imports,
    project_path,
    require_loaded,
    result,
    rgb_to_bgr,
)

LOGGER = logging.getLogger(__name__)


def _resolve_effective_device(requested_device: str) -> str:
    import torch

    normalized = requested_device.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning(
            "Requested device=%s for ColorComic, but CUDA is unavailable. "
            "Falling back to cpu.",
            requested_device,
        )
        return "cpu"
    return requested_device


class ColorComicAutoModel(ColorizationModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._colorizer: Any | None = None
        self._repo_path: Path | None = None

    def load(self) -> None:
        if self._colorizer is not None:
            return

        repo_path = project_path(self.config, "repo_path")
        assert repo_path is not None
        self._repo_path = repo_path
        weights_dir = repo_path / str(self.config.get("weights_dir", "models/weights"))
        generator_path = repo_path / str(
            self.config.get("generator_path", "models/weights/generator.zip")
        )
        extractor_path = repo_path / str(
            self.config.get("extractor_path", "models/weights/extractor.pth")
        )
        denoiser_dir = repo_path / str(
            self.config.get("denoiser_weights_dir", "models/weights/denoiser")
        )

        if bool(self.config.get("allow_download", False)):
            try:
                with isolated_vendor_imports(repo_path):
                    from core.model_downloader import ensure_models_downloaded

                    ensure_models_downloaded(str(weights_dir))
            except ImportError as exc:
                raise ModelBackendUnavailableError(
                    f"ColorComic downloader dependencies are unavailable: {exc}"
                ) from exc

        if not generator_path.exists():
            raise FileNotFoundError(
                f"ColorComic generator weights not found: {generator_path}"
            )

        effective_device = _resolve_effective_device(
            str(self.config.get("device", "auto"))
        )
        try:
            with isolated_vendor_imports(repo_path):
                from core.ml_colorizer import MangaColorizer

                self._colorizer = MangaColorizer(
                    device=effective_device,
                    generator_path=str(generator_path),
                    extractor_path=(
                        str(extractor_path) if extractor_path.exists() else ""
                    ),
                    denoiser_weights_dir=str(denoiser_dir),
                )
        except ImportError as exc:
            raise ModelBackendUnavailableError(
                f"ColorComic auto backend is unavailable: {exc}"
            ) from exc

    def unload(self) -> None:
        if self._colorizer is not None and hasattr(self._colorizer, "unload"):
            self._colorizer.unload()
        self._colorizer = None

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        colorizer = require_loaded(self._colorizer, self.model_id)
        size = int(request.options.get("size", self.config.get("size", 576)))
        start_time = time.perf_counter()
        output_bgr = colorizer.colorize(rgb_to_bgr(request.input_image), size=size)
        return result(
            image=bgr_to_rgb(output_bgr),
            model_id=self.model_id,
            start_time=start_time,
            metadata={"size": size},
        )


class ColorComicReferenceModel(ColorizationModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._colorizer: Any | None = None

    @property
    def requires_reference(self) -> bool:
        return True

    def load(self) -> None:
        if self._colorizer is not None:
            return

        repo_path = project_path(self.config, "repo_path")
        assert repo_path is not None
        weights_dir = repo_path / str(self.config.get("weights_dir", "models/weights"))
        manganinja_dir = weights_dir / "manganinja"

        cfg = SimpleNamespace(
            MANGANINJA_WEIGHTS_DIR=str(manganinja_dir),
            MANGANINJA_DENOISING_UNET=str(manganinja_dir / "denoising_unet.pth"),
            MANGANINJA_REFERENCE_UNET=str(manganinja_dir / "reference_unet.pth"),
            MANGANINJA_POINTNET=str(manganinja_dir / "point_net.pth"),
            MANGANINJA_CONTROLNET=str(manganinja_dir / "controlnet.pth"),
            MANGANINJA_HF_REPO=str(
                self.config.get("manganinja_hf_repo", "Johanan0528/MangaNinja")
            ),
            SD15_MODEL_PATH=str(
                self.config.get(
                    "sd15_model_path",
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                )
            ),
            CLIP_VISION_PATH=str(
                self.config.get("clip_vision_path", "openai/clip-vit-large-patch14")
            ),
            CONTROLNET_LINEART_PATH=str(
                self.config.get(
                    "controlnet_lineart_path", "lllyasviel/control_v11p_sd15_lineart"
                )
            ),
            LINEART_ANNOTATOR_PATH=str(manganinja_dir / "annotators"),
            MANGANINJA_DENOISE_STEPS=int(self.config.get("denoise_steps", 30)),
        )

        if bool(self.config.get("allow_download", False)):
            try:
                with isolated_vendor_imports(repo_path):
                    from core.model_downloader import ensure_manganinja_downloaded

                    ensure_manganinja_downloaded(cfg)
            except ImportError as exc:
                raise ModelBackendUnavailableError(
                    "ColorComic reference downloader dependencies are "
                    f"unavailable: {exc}"
                ) from exc

        missing = [
            path
            for path in (
                cfg.MANGANINJA_DENOISING_UNET,
                cfg.MANGANINJA_REFERENCE_UNET,
                cfg.MANGANINJA_POINTNET,
                cfg.MANGANINJA_CONTROLNET,
            )
            if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "ColorComic reference weights are missing: "
                + ", ".join(str(path) for path in missing)
            )

        effective_device = _resolve_effective_device(
            str(self.config.get("device", "auto"))
        )
        try:
            with isolated_vendor_imports(repo_path):
                from core.manga_ninja_colorizer import MangaNinjaColorizer

                self._colorizer = MangaNinjaColorizer(
                    device=effective_device,
                    config=cfg,
                )
        except ImportError as exc:
            raise ModelBackendUnavailableError(
                f"ColorComic reference backend is unavailable: {exc}"
            ) from exc

    def unload(self) -> None:
        if self._colorizer is not None and hasattr(self._colorizer, "unload"):
            self._colorizer.unload()
        self._colorizer = None

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        colorizer = require_loaded(self._colorizer, self.model_id)
        reference = self.get_reference(request)
        size = int(request.options.get("size", self.config.get("size", 512)))
        start_time = time.perf_counter()
        output_bgr = colorizer.colorize(
            rgb_to_bgr(request.input_image),
            reference_image=rgb_to_bgr(reference),
            size=size,
        )
        return result(
            image=bgr_to_rgb(output_bgr),
            model_id=self.model_id,
            start_time=start_time,
            metadata={"size": size},
        )
