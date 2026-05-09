from dataclasses import dataclass

import cv2
import numpy as np

from illustration_colorizer.benchmark.image_utils import (
    ensure_rgb,
    rgb_to_luminance,
    to_grayscale,
)
from illustration_colorizer.benchmark.metrics.base import BenchmarkMetric


@dataclass(frozen=True)
class LinePreservationMetric(BenchmarkMetric):
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    name: str = "line_preservation_score"

    def compute(
        self,
        *,
        x_images: list[np.ndarray],
        y_images: list[np.ndarray],
        g_images: list[np.ndarray] | None = None,
    ) -> float:
        if len(x_images) != len(y_images):
            raise ValueError("x_images and y_images must have the same length.")

        scores = [
            self.compute_single(x_image=x_image, y_image=y_image) for x_image, y_image in zip(x_images, y_images)
        ]
        return float(np.mean(scores))

    def compute_single(self, x_image: np.ndarray, y_image: np.ndarray) -> float:
        x_gray = to_grayscale(x_image)
        y_lum = rgb_to_luminance(ensure_rgb(y_image))

        x_gray_u8 = np.clip(x_gray, 0, 255).astype(np.uint8)
        y_lum_u8 = np.clip(y_lum, 0, 255).astype(np.uint8)

        edge_x = (
            cv2.Canny(
                x_gray_u8,
                self.canny_low_threshold,
                self.canny_high_threshold,
            ).astype(np.float32)
            / 255.0
        )
        edge_y = (
            cv2.Canny(
                y_lum_u8,
                self.canny_low_threshold,
                self.canny_high_threshold,
            ).astype(np.float32)
            / 255.0
        )

        numerator = np.sum(np.abs(edge_x - edge_y))
        denominator = np.sum(edge_x) + 1e-8
        score = 1.0 - numerator / denominator

        return float(np.clip(score, 0.0, 1.0))


@dataclass(frozen=True)
class InkPreservationMetric(BenchmarkMetric):
    ink_threshold: int = 60
    name: str = "ink_preservation_score"

    def compute(
        self,
        *,
        x_images: list[np.ndarray],
        y_images: list[np.ndarray],
        g_images: list[np.ndarray] | None = None,
    ) -> float:
        if len(x_images) != len(y_images):
            raise ValueError("x_images and y_images must have the same length.")

        scores = [
            self.compute_single(x_image=x_image, y_image=y_image) for x_image, y_image in zip(x_images, y_images)
        ]
        return float(np.mean(scores))

    def compute_single(self, x_image: np.ndarray, y_image: np.ndarray) -> float:
        x_gray = to_grayscale(x_image)
        y_lum = rgb_to_luminance(ensure_rgb(y_image))
        ink_mask = x_gray < self.ink_threshold

        if ink_mask.sum() == 0:
            return 1.0

        error = np.mean(np.abs(y_lum[ink_mask] - x_gray[ink_mask]) / 255.0)
        score = 1.0 - error

        return float(np.clip(score, 0.0, 1.0))
