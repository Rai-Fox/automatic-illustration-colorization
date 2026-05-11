from __future__ import annotations

import contextlib
import importlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from illustration_colorizer.models.base import (
    ColorizationRequest,
    ColorizationResult,
    ModelBackendUnavailableError,
    ModelNotLoadedError,
)
from shared.images import to_bgr_uint8, to_rgb_uint8
from shared.paths import resolve_from_root

TOP_LEVEL_VENDOR_MODULES = (
    "config",
    "core",
    "models",
    "utils",
    "vendor",
    "vgg_model",
    "extractor",
)


def project_path(
    config: dict[str, Any], key: str, *, required: bool = True
) -> Path | None:
    project_root = Path(config["project_root"])
    path = resolve_from_root(project_root, config.get(key))
    if required and (path is None or not path.exists()):
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def require_loaded(value: Any, model_id: str) -> Any:
    if value is None:
        raise ModelNotLoadedError(f"{model_id} model is not loaded.")
    return value


def rgb_uint8(image: np.ndarray) -> np.ndarray:
    return to_rgb_uint8(image)


def bgr_uint8(image: np.ndarray) -> np.ndarray:
    return to_bgr_uint8(image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(rgb_uint8(image), cv2.COLOR_RGB2BGR)


def result(
    *,
    image: np.ndarray,
    model_id: str,
    start_time: float | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ColorizationResult:
    return ColorizationResult(
        image=rgb_uint8(image),
        model_id=model_id,
        latency_seconds=(
            time.perf_counter() - start_time if start_time is not None else None
        ),
        warnings=list(warnings or []),
        metadata=dict(metadata or {}),
    )


def request_seed(request: ColorizationRequest, default: int) -> int:
    if request.seed is not None:
        return int(request.seed)
    if "seed" in request.options:
        return int(request.options["seed"])
    return default


def request_option(request: ColorizationRequest, key: str, default: Any) -> Any:
    return request.options.get(key, default)


@contextlib.contextmanager
def prepended_sys_path(path: Path) -> Iterator[None]:
    resolved = str(path.resolve())
    added = resolved not in sys.path
    if added:
        sys.path.insert(0, resolved)
    try:
        yield
    finally:
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(resolved)


@contextlib.contextmanager
def isolated_vendor_imports(path: Path) -> Iterator[None]:
    previous = {name: sys.modules.get(name) for name in TOP_LEVEL_VENDOR_MODULES}
    for name in TOP_LEVEL_VENDOR_MODULES:
        sys.modules.pop(name, None)
    with prepended_sys_path(path):
        try:
            yield
        finally:
            for name in TOP_LEVEL_VENDOR_MODULES:
                sys.modules.pop(name, None)
                module = previous[name]
                if module is not None:
                    sys.modules[name] = module


def import_from_path(path: Path, module_name: str) -> ModuleType:
    try:
        with prepended_sys_path(path):
            return importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelBackendUnavailableError(
            f"Could not import {module_name!r} from {path}: {exc}"
        ) from exc
