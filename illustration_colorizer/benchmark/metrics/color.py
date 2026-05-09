from dataclasses import dataclass

import numpy as np

from illustration_colorizer.benchmark.image_utils import ensure_rgb
from illustration_colorizer.benchmark.metrics.base import BenchmarkMetric


@dataclass(frozen=True)
class ColorfulnessMetric(BenchmarkMetric):
    name: str = "colorfulness"

    def compute(
        self,
        *,
        x_images: list[np.ndarray],
        y_images: list[np.ndarray],
        g_images: list[np.ndarray] | None = None,
    ) -> float:
        scores = [self.compute_single(image) for image in y_images]
        return float(np.mean(scores))

    def compute_single(self, image_rgb: np.ndarray) -> float:
        img = ensure_rgb(image_rgb)

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg = np.std(rg)
        std_yb = np.std(yb)

        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)

        colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        return float(colorfulness)
