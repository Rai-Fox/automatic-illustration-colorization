from __future__ import annotations

import numpy as np
from PIL import Image

from illustration_colorizer.benchmark.datasets import BenchmarkSample
from illustration_colorizer.benchmark.runner import (
    MetricReport,
    ModelReport,
    _build_result_panel,
    _save_generated_images,
    _write_per_model_reports,
    _write_per_model_run_reports,
)


def test_build_result_panel_uses_input_target_and_output() -> None:
    sample = BenchmarkSample(
        sample_id="sample",
        input_image=np.full((8, 10, 3), 127, dtype=np.uint8),
        target_image=np.zeros((6, 7, 3), dtype=np.uint8),
    )
    output = np.full((4, 5, 3), 255, dtype=np.uint8)

    panel = _build_result_panel(sample, output)

    assert panel.shape == (8, 30, 3)


def test_build_result_panel_without_ground_truth_uses_input_and_output() -> None:
    sample = BenchmarkSample(
        sample_id="sample",
        input_image=np.full((8, 10, 3), 127, dtype=np.uint8),
    )
    output = np.full((6, 7, 3), 255, dtype=np.uint8)

    panel = _build_result_panel(sample, output)

    assert panel.shape == (8, 20, 3)


def test_save_generated_images_uses_model_subdirectory(tmp_path) -> None:
    _save_generated_images(
        output_dir=tmp_path,
        model_id="ddcolor",
        generated_images=[
            (
                BenchmarkSample(
                    sample_id="sample_0",
                    input_image=np.zeros((8, 10, 3), dtype=np.uint8),
                    target_image=np.ones((8, 10, 3), dtype=np.uint8) * 127,
                ),
                np.ones((8, 10, 3), dtype=np.uint8) * 255,
            ),
            (
                BenchmarkSample(
                    sample_id="sample_1",
                    input_image=np.zeros((8, 10, 3), dtype=np.uint8),
                    target_image=np.ones((8, 10, 3), dtype=np.uint8) * 127,
                ),
                np.ones((8, 10, 3), dtype=np.uint8) * 255,
            ),
        ],
        max_saved_images=1,
    )

    saved = sorted(path.name for path in (tmp_path / "ddcolor").glob("*.png"))
    assert saved == ["sample_0.png"]

    first_panel = np.asarray(
        Image.open(tmp_path / "ddcolor" / "sample_0.png").convert("RGB")
    )
    assert first_panel.shape == (8, 30, 3)
    assert (tmp_path / "ddcolor" / "manifest.json").exists()


def test_write_per_model_reports_uses_model_subdirectories(tmp_path) -> None:
    run_id = "fixed_by_title_001"
    written = _write_per_model_reports(
        output_dir=tmp_path,
        benchmark_config={
            "report": {
                "per_model_dir_name": "reports",
                "json_name": "report.json",
                "csv_name": "summary.csv",
            }
        },
        dataset={"source": "unit", "sample_count": 1},
        run_id=run_id,
        model_reports=[
            ModelReport(
                model_id="ddcolor",
                metrics={
                    "colorfulness": MetricReport(
                        status="computed",
                        sample_count=1,
                        value=12.0,
                    )
                },
                performance={},
                resources={},
                counts={"successful_samples": 1},
            ),
            ModelReport(
                model_id="cgan/reference",
                metrics={},
                performance={},
                resources={},
                counts={"successful_samples": 0},
            ),
        ],
    )

    assert (tmp_path / "reports" / "ddcolor" / run_id / "report.json").exists()
    assert (tmp_path / "reports" / "ddcolor" / run_id / "summary.csv").exists()
    assert (
        tmp_path / "reports" / "cgan_reference" / run_id / "report.json"
    ).exists()
    assert written["ddcolor"]["json_report"].endswith(
        f"reports\\ddcolor\\{run_id}\\report.json"
    )


def test_write_per_model_run_reports_uses_run_subdirectories(tmp_path) -> None:
    written = _write_per_model_run_reports(
        output_dir=tmp_path,
        benchmark_config={
            "report": {
                "per_run_dir_name": "runs",
                "json_name": "report.json",
                "csv_name": "summary.csv",
            }
        },
        dataset={"source": "unit", "sample_count": 1},
        model_reports=[
            ModelReport(
                model_id="ddcolor",
                metrics={},
                performance={},
                resources={},
                counts={"successful_samples": 1},
            )
        ],
        run_id="run_001",
    )

    assert (tmp_path / "runs" / "run_001" / "ddcolor" / "report.json").exists()
    assert (tmp_path / "runs" / "run_001" / "ddcolor" / "summary.csv").exists()
    assert written["ddcolor"]["json_report"].endswith(
        "runs\\run_001\\ddcolor\\report.json"
    )
