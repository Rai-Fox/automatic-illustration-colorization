from illustration_colorizer.benchmark.metrics.base import BenchmarkMetric
from illustration_colorizer.benchmark.metrics.color import ColorfulnessMetric
from illustration_colorizer.benchmark.metrics.compute import compute_metrics
from illustration_colorizer.benchmark.metrics.perceptual import KidMetric, LpipsMetric
from illustration_colorizer.benchmark.metrics.preservation import (
    InkPreservationMetric,
    LinePreservationMetric,
)
