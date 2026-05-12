from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from illustration_colorizer.benchmark.aggregate import aggregate_generated_panels
from illustration_colorizer.benchmark.datasets import BenchmarkSample
from illustration_colorizer.benchmark.runner import _save_generated_images


def test_aggregate_generated_panels_creates_comparison_panel(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "outputs" / "benchmark"
    generated_root = benchmark_root / "generated"
    sample = BenchmarkSample(
        sample_id="sample_0",
        input_image=np.zeros((8, 10, 3), dtype=np.uint8),
        target_image=np.ones((8, 10, 3), dtype=np.uint8) * 127,
    )

    _save_generated_images(
        output_dir=generated_root,
        model_id="ddcolor",
        run_id="run_a",
        generated_images=[(sample, np.ones((8, 10, 3), dtype=np.uint8) * 64)],
        max_saved_images=10,
    )
    _save_generated_images(
        output_dir=generated_root,
        model_id="deoldify",
        run_id="run_b",
        generated_images=[(sample, np.ones((8, 10, 3), dtype=np.uint8) * 255)],
        max_saved_images=10,
    )

    result = aggregate_generated_panels(
        project_root=tmp_path,
        models=["ddcolor", "deoldify"],
        benchmark_output_dir="outputs/benchmark",
        max_images=10,
        samples=[sample],
    )

    assert result["sample_count"] == 1
    output_path = benchmark_root / "comparisons" / "sample_0.png"
    assert output_path.exists()

    panel = np.asarray(Image.open(output_path).convert("RGB"))
    assert panel.shape == (8, 40, 3)


def test_aggregate_generated_panels_can_compare_same_model_runs(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "outputs" / "benchmark"
    generated_root = benchmark_root / "generated"
    sample = BenchmarkSample(
        sample_id="sample_0",
        input_image=np.zeros((8, 10, 3), dtype=np.uint8),
        target_image=np.ones((8, 10, 3), dtype=np.uint8) * 127,
    )

    _save_generated_images(
        output_dir=generated_root,
        model_id="ddcolor",
        run_id="small",
        generated_images=[(sample, np.ones((8, 10, 3), dtype=np.uint8) * 64)],
        max_saved_images=10,
    )
    _save_generated_images(
        output_dir=generated_root,
        model_id="ddcolor",
        run_id="large",
        generated_images=[(sample, np.ones((8, 10, 3), dtype=np.uint8) * 255)],
        max_saved_images=10,
    )

    result = aggregate_generated_panels(
        project_root=tmp_path,
        models=["ddcolor:small", "ddcolor:large"],
        benchmark_output_dir="outputs/benchmark",
        max_images=10,
        samples=[sample],
    )

    assert result["sample_count"] == 1
    output_path = benchmark_root / "comparisons" / "sample_0.png"
    panel = np.asarray(Image.open(output_path).convert("RGB"))
    assert panel.shape == (8, 40, 3)
    assert [model["label"] for model in result["models"]] == [
        "ddcolor_small",
        "ddcolor_large",
    ]
