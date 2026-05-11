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
        y_batch = numpy_images_to_torch_batch(
            y_images,
            image_size=self.image_size,
            device=effective_device,
        )
        g_batch = numpy_images_to_torch_batch(
            g_images,
            image_size=self.image_size,
            device=effective_device,
        )

        y_batch = y_batch * 2.0 - 1.0
        g_batch = g_batch * 2.0 - 1.0
        scores = []

        with torch.no_grad():
            for start in range(0, len(y_batch), self.batch_size):
                end = start + self.batch_size
                score_batch = learned_perceptual_image_patch_similarity(
                    y_batch[start:end],
                    g_batch[start:end],
                    net_type=self.lpips_net,
                    reduction="none",
                    normalize=False,
                )
                scores.append(score_batch.detach().cpu().numpy().reshape(-1))

        return float(np.mean(np.concatenate(scores, axis=0)))


@dataclass(frozen=True)
class KidMetric(BenchmarkMetric):
    image_size: tuple[int, int] | None = None
    device: str | torch.device = "cpu"
    kid_subset_size: int = 50
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

        effective_device = resolve_torch_device(
            self.device,
            logger=LOGGER,
            context=self.name,
        )
        y_batch = numpy_images_to_torch_batch(
            y_images,
            image_size=self.image_size,
            device=effective_device,
        )
        g_batch = numpy_images_to_torch_batch(
            g_images,
            image_size=self.image_size,
            device=effective_device,
        )
        y_uint8 = torch_batch_to_uint8(y_batch)
        g_uint8 = torch_batch_to_uint8(g_batch)

        subset_size = min(self.kid_subset_size, y_uint8.shape[0], g_uint8.shape[0])
        if subset_size < 2:
            raise ValueError("kid_subset_size must be at least 2 after clipping.")

        kid = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(
            effective_device
        )
        kid.update(g_uint8, real=True)
        kid.update(y_uint8, real=False)

        kid_mean, kid_std = kid.compute()
        return {
            "kid_mean": float(kid_mean.item()),
            "kid_std": float(kid_std.item()),
        }
