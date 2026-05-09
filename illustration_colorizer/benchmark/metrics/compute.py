from typing import Any

import numpy as np

from illustration_colorizer.benchmark.metrics.base import BenchmarkMetric


def compute_metrics(
    x_images: list[np.ndarray],
    y_images: list[np.ndarray],
    g_images: list[np.ndarray] | None = None,
    metrics: list[BenchmarkMetric] | None = None,
) -> dict[str, Any]:
    if len(x_images) != len(y_images):
        raise ValueError("x_images and y_images must have the same length.")
    if g_images is not None and len(y_images) != len(g_images):
        raise ValueError("y_images and g_images must have the same length.")

    selected_metrics = [] if metrics is None else metrics
    names = [metric.name for metric in selected_metrics]
    if len(names) != len(set(names)):
        raise ValueError(f"Metric names must be unique, got duplicates: {names}")

    results = {}
    for metric in selected_metrics:
        results[metric.name] = metric.compute(
            x_images=x_images,
            y_images=y_images,
            g_images=g_images,
        )

    return results
