from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from illustration_colorizer.benchmark.aggregate import (
    _CAPTION_HEIGHT,
    aggregate_generated_panels,
)
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
    assert panel.shape == (8 + _CAPTION_HEIGHT, 40, 3)
    assert not np.array_equal(
        panel[8:, :10],
        np.zeros((_CAPTION_HEIGHT, 10, 3), dtype=np.uint8),
    )


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
    assert panel.shape == (8 + _CAPTION_HEIGHT, 40, 3)
    assert [model["label"] for model in result["models"]] == [
        "ddcolor_small",
        "ddcolor_large",
    ]


def test_aggregate_generated_panels_uses_model_intersection_in_dataset_order(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "outputs" / "benchmark"
    generated_root = benchmark_root / "generated"
    samples = [
        BenchmarkSample(
            sample_id=f"sample_{index}",
            input_image=np.zeros((8, 10, 3), dtype=np.uint8),
            target_image=np.ones((8, 10, 3), dtype=np.uint8) * 127,
        )
        for index in range(4)
    ]

    _save_generated_images(
        output_dir=generated_root,
        model_id="ddcolor",
        run_id="run_a",
        generated_images=[
            (samples[0], np.ones((8, 10, 3), dtype=np.uint8) * 32),
            (samples[2], np.ones((8, 10, 3), dtype=np.uint8) * 64),
            (samples[3], np.ones((8, 10, 3), dtype=np.uint8) * 96),
        ],
        max_saved_images=10,
    )
    _save_generated_images(
        output_dir=generated_root,
        model_id="deoldify",
        run_id="run_b",
        generated_images=[
            (samples[1], np.ones((8, 10, 3), dtype=np.uint8) * 128),
            (samples[2], np.ones((8, 10, 3), dtype=np.uint8) * 160),
            (samples[3], np.ones((8, 10, 3), dtype=np.uint8) * 192),
        ],
        max_saved_images=10,
    )

    result = aggregate_generated_panels(
        project_root=tmp_path,
        models=["ddcolor", "deoldify"],
        benchmark_output_dir="outputs/benchmark",
        max_images=1,
        random_seed=0,
        samples=samples,
    )

    assert result["sample_count"] == 1
    assert result["sample_ids"] == ["sample_2"]
    assert (benchmark_root / "comparisons" / "sample_2.png").exists()
    assert not (benchmark_root / "comparisons" / "sample_3.png").exists()


def test_aggregate_generated_panels_expands_reference_modes(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "outputs" / "benchmark"
    generated_root = benchmark_root / "generated"
    aggregate_sample = BenchmarkSample(
        sample_id="sample_0",
        input_image=np.zeros((8, 10, 3), dtype=np.uint8),
        target_image=np.ones((8, 10, 3), dtype=np.uint8) * 127,
        metadata={"title": "A"},
    )
    fixed_sample = BenchmarkSample(
        sample_id="sample_0",
        input_image=aggregate_sample.input_image,
        target_image=aggregate_sample.target_image,
        metadata={"title": "A", "reference_mode": "fixed_by_title"},
    )
    previous_sample = BenchmarkSample(
        sample_id="sample_0",
        input_image=aggregate_sample.input_image,
        target_image=aggregate_sample.target_image,
        metadata={"title": "A", "reference_mode": "previous_output_by_title"},
    )

    _save_generated_images(
        output_dir=generated_root,
        model_id="cgan_reference",
        run_id="cgan_reference_fixed_by_title_cuda_images_all",
        generated_images=[(fixed_sample, np.ones((8, 10, 3), dtype=np.uint8) * 64)],
        max_saved_images=10,
    )
    _save_generated_images(
        output_dir=generated_root,
        model_id="cgan_reference",
        run_id="cgan_reference_previous_output_by_title_cuda_images16",
        generated_images=[
            (previous_sample, np.ones((8, 10, 3), dtype=np.uint8) * 96)
        ],
        max_saved_images=10,
    )
    _save_generated_images(
        output_dir=generated_root,
        model_id="ddcolor",
        run_id="ddcolor_cuda_images_all",
        generated_images=[
            (aggregate_sample, np.ones((8, 10, 3), dtype=np.uint8) * 128)
        ],
        max_saved_images=10,
    )

    result = aggregate_generated_panels(
        project_root=tmp_path,
        models=["cgan_reference", "ddcolor"],
        benchmark_output_dir="outputs/benchmark",
        max_images=10,
        samples=[aggregate_sample],
    )

    assert result["sample_count"] == 1
    assert [model["label"] for model in result["models"]] == [
        "cgan_reference_fixed_by_title",
        "cgan_reference_previous_output_by_title",
        "ddcolor",
    ]
    panel = np.asarray(
        Image.open(benchmark_root / "comparisons" / "sample_0.png").convert("RGB")
    )
    assert panel.shape == (8 + _CAPTION_HEIGHT, 50, 3)


def test_aggregate_generated_panels_balances_samples_by_title(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "outputs" / "benchmark"
    generated_root = benchmark_root / "generated"
    samples = [
        BenchmarkSample(
            sample_id=f"sample_{index}",
            input_image=np.zeros((8, 10, 3), dtype=np.uint8),
            target_image=np.ones((8, 10, 3), dtype=np.uint8) * 127,
            metadata={"title": title},
        )
        for index, title in enumerate(["A", "A", "B", "B"])
    ]

    for model_id, value in [("ddcolor", 64), ("deoldify", 128)]:
        _save_generated_images(
            output_dir=generated_root,
            model_id=model_id,
            run_id=f"{model_id}_all",
            generated_images=[
                (sample, np.ones((8, 10, 3), dtype=np.uint8) * value)
                for sample in samples
            ],
            max_saved_images=10,
        )

    result = aggregate_generated_panels(
        project_root=tmp_path,
        models=["ddcolor", "deoldify"],
        benchmark_output_dir="outputs/benchmark",
        max_images=3,
        random_seed=1,
        samples=samples,
    )

    assert result["sample_ids"] == ["sample_1", "sample_3", "sample_0"]
    assert (benchmark_root / "comparisons" / "sample_1.png").exists()
    assert (benchmark_root / "comparisons" / "sample_3.png").exists()
    assert (benchmark_root / "comparisons" / "sample_0.png").exists()
    assert not (benchmark_root / "comparisons" / "sample_2.png").exists()
