import logging
from dataclasses import dataclass

import numpy as np
import torch
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)
from torchmetrics.image.kid import KernelInceptionDistance

from illustration_colorizer.benchmark.image_utils import (
    numpy_images_to_torch_batch,
    resolve_torch_device,
    torch_batch_to_uint8,
)
from illustration_colorizer.benchmark.metrics.base import BenchmarkMetric

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LpipsMetric(BenchmarkMetric):
    image_size: tuple[int, int] | None = None
    device: str | torch.device = "cpu"
    lpips_net: str = "alex"
    batch_size: int = 8
    name: str = "lpips"
    requires_ground_truth = True

    def compute(
        self,
        *,
        x_images: list[np.ndarray],
        y_images: list[np.ndarray],
        g_images: list[np.ndarray] | None = None,
    ) -> float:
        if g_images is None:
            raise ValueError("g_images is required for LPIPS.")
        if len(y_images) != len(g_images):
            raise ValueError("y_images and g_images must have the same length.")

        effective_device = resolve_torch_device(
            self.device,
            logger=LOGGER,
            context=self.name,
        )
        scores = []
        batch_size = max(1, int(self.batch_size))

        with torch.no_grad():
            for start in range(0, len(y_images), batch_size):
                end = start + batch_size
                y_batch = numpy_images_to_torch_batch(
                    y_images[start:end],
                    image_size=self.image_size,
                    device=effective_device,
                )
                g_batch = numpy_images_to_torch_batch(
                    g_images[start:end],
                    image_size=self.image_size,
                    device=effective_device,
                )
                y_batch = y_batch * 2.0 - 1.0
                g_batch = g_batch * 2.0 - 1.0
                score_batch = learned_perceptual_image_patch_similarity(
                    y_batch,
                    g_batch,
                    net_type=self.lpips_net,
                    reduction="none",
                    normalize=False,
                )
                scores.append(score_batch.detach().cpu().numpy().reshape(-1))
                del y_batch, g_batch, score_batch

        return float(np.mean(np.concatenate(scores, axis=0)))


@dataclass(frozen=True)
class KidMetric(BenchmarkMetric):
    image_size: tuple[int, int] | None = None
    device: str | torch.device = "cpu"
    kid_subset_size: int = 50
    batch_size: int = 8
    name: str = "kid"
    requires_ground_truth = True

    def compute(
        self,
        *,
        x_images: list[np.ndarray],
        y_images: list[np.ndarray],
        g_images: list[np.ndarray] | None = None,
    ) -> dict[str, float]:
        if g_images is None:
            raise ValueError("g_images is required for KID.")
        if len(y_images) < 2 or len(g_images) < 2:
            raise ValueError(
                "KID requires at least 2 generated and 2 reference images."
            )

        max_sample_count = self.effective_sample_count(len(y_images))
        kid = self.create_state(expected_sample_count=len(y_images))
        self.update_state(
            kid,
            y_images=y_images[:max_sample_count],
            g_images=g_images[:max_sample_count],
        )
        kid_mean, kid_std = kid.compute()
        return {
            "kid_mean": float(kid_mean.item()),
            "kid_std": float(kid_std.item()),
        }

    def effective_sample_count(self, expected_sample_count: int) -> int:
        return min(int(self.kid_subset_size), int(expected_sample_count))

    def create_state(self, *, expected_sample_count: int) -> KernelInceptionDistance:
        effective_device = resolve_torch_device(
            self.device,
            logger=LOGGER,
            context=self.name,
        )
        subset_size = self.effective_sample_count(expected_sample_count)
        if subset_size < 2:
            raise ValueError("kid_subset_size must be at least 2 after clipping.")
        return KernelInceptionDistance(subset_size=subset_size, normalize=False).to(
            effective_device
        )

    def update_state(
        self,
        kid: KernelInceptionDistance,
        *,
        y_images: list[np.ndarray],
        g_images: list[np.ndarray],
    ) -> None:
        if len(y_images) != len(g_images):
            raise ValueError("y_images and g_images must have the same length.")

        batch_size = max(1, int(self.batch_size))
        effective_device = resolve_torch_device(
            self.device,
            logger=LOGGER,
            context=self.name,
        )
        for start in range(0, len(y_images), batch_size):
            end = start + batch_size
            y_batch = numpy_images_to_torch_batch(
                y_images[start:end],
                image_size=self.image_size,
                device=effective_device,
            )
            g_batch = numpy_images_to_torch_batch(
                g_images[start:end],
                image_size=self.image_size,
                device=effective_device,
            )
            y_uint8 = torch_batch_to_uint8(y_batch)
            g_uint8 = torch_batch_to_uint8(g_batch)
            kid.update(g_uint8, real=True)
            kid.update(y_uint8, real=False)
            del y_batch, g_batch, y_uint8, g_uint8

    def compute_state(self, kid: KernelInceptionDistance) -> dict[str, float]:
        kid_mean, kid_std = kid.compute()
        return {
            "kid_mean": float(kid_mean.item()),
            "kid_std": float(kid_std.item()),
        }
