from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ColorizationRequest:
    input_image: np.ndarray
    reference_image: np.ndarray | None = None
    reference_images: list[np.ndarray] = field(default_factory=list)
    sample_id: str | None = None
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColorizationResult:
    image: np.ndarray
    model_id: str
    latency_seconds: float | None = None
    resource_usage: dict[str, float | int | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ColorizationModelError(RuntimeError):
    """Base exception for model wrapper failures."""


class ModelNotLoadedError(ColorizationModelError):
    """Raised when colorize is called before load."""


class MissingReferenceImageError(ColorizationModelError):
    """Raised when a reference-based model receives no reference image."""


class ModelBackendUnavailableError(ColorizationModelError):
    """Raised when a backend cannot be imported or initialized."""


class ColorizationModel(ABC):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @property
    def model_id(self) -> str:
        return str(self.config["model_id"])

    @property
    def supports_conditioning(self) -> bool:
        return bool(self.config.get("supports_conditioning", False))

    @property
    def requires_reference(self) -> bool:
        return bool(self.config.get("requires_reference", False))

    @property
    def supports_multiple_references(self) -> bool:
        return bool(self.config.get("supports_multiple_references", False))

    @property
    def supports_cpu(self) -> bool:
        return bool(self.config.get("supports_cpu", True))

    def get_reference(self, request: ColorizationRequest) -> np.ndarray:
        if request.reference_image is not None:
            return request.reference_image
        if request.reference_images:
            return request.reference_images[0]
        raise MissingReferenceImageError(
            f"{self.model_id} requires reference_image or reference_images."
        )

    def get_references(self, request: ColorizationRequest) -> list[np.ndarray]:
        references = list(request.reference_images)
        if request.reference_image is not None:
            references.insert(0, request.reference_image)
        if not references:
            raise MissingReferenceImageError(
                f"{self.model_id} requires at least one reference image."
            )
        return references

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize the inference pipeline."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @abstractmethod
    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        """Run image colorization for one request."""

    def colorize_batch(
        self, requests: list[ColorizationRequest]
    ) -> list[ColorizationResult]:
        """Run image colorization for a batch of requests."""
        return [self.colorize(request) for request in requests]
