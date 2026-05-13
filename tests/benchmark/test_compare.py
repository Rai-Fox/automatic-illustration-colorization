from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cli as cli_module
from illustration_colorizer.benchmark.compare import compare_benchmark_reports


def _write_report(
    root: Path,
    *,
    model_id: str,
    run_id: str,
    report_name: str = "report_all.json",
    reference_mode: str = "none",
    metrics: dict[str, Any],
    successful_samples: int = 10,
) -> Path:
    report_dir = root / "outputs" / "benchmark" / "reports" / model_id / run_id
    report_dir.mkdir(parents=True)
    report_path = report_dir / report_name
    report_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "source": "unit",
                    "sample_count": successful_samples,
                    "samples_with_ground_truth": successful_samples,
                    "benchmark_mode": "metrics_only",
                    "run_id": run_id,
                    "reference_mode": reference_mode,
                },
                "model_reports": [
                    {
                        "model_id": model_id,
                        "metrics": metrics,
                        "counts": {
                            "total_samples": successful_samples,
                            "successful_samples": successful_samples,
                            "failed_samples": 0,
                            "samples_with_ground_truth": successful_samples,
                        },
                        "performance": {},
                        "resources": {},
                        "failures": [],
                        "warnings": [],
                    }
                ],
                "output_dir": str(root / "outputs" / "benchmark"),
            }
        ),
        encoding="utf-8",
    )
    return report_path


def _metric(value: Any, status: str = "computed") -> dict[str, Any]:
    return {
        "status": status,
        "sample_count": 10 if status == "computed" else 0,
        "value": value,
        "reason": None if status == "computed" else "skipped",
    }


def test_compare_reports_flattens_nested_kid_and_writes_outputs(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        model_id="ddcolor",
        run_id="ddcolor_cuda_images_all",
        metrics={
            "colorfulness": _metric(2.0),
            "lpips": _metric(0.2),
            "kid": _metric({"kid_mean": 0.02, "kid_std": 0.001}),
        },
    )
    _write_report(
        tmp_path,
        model_id="deoldify",
        run_id="deoldify_cuda_images_all",
        metrics={
            "colorfulness": _metric(3.0),
            "lpips": _metric(0.1),
            "kid": _metric({"kid_mean": 0.03, "kid_std": 0.002}),
        },
    )

    result = compare_benchmark_reports(
        project_root=tmp_path,
        metrics="colorfulness,lpips,kid.kid_mean",
        make_plots=False,
    )

    csv_path = Path(str(result["csv"]))
    tex_path = Path(str(result["latex"]))
    json_path = Path(str(result["json"]))
    assert csv_path.exists()
    assert tex_path.exists()
    assert json_path.exists()
    assert "kid.kid_mean" in csv_path.read_text(encoding="utf-8")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["best_by_metric"]["colorfulness"] == ["deoldify"]
    assert summary["best_by_metric"]["lpips"] == ["deoldify"]
    assert summary["best_by_metric"]["kid.kid_mean"] == ["ddcolor"]


def test_compare_reports_latex_highlights_best_and_skips_missing(
    tmp_path: Path,
) -> None:
    _write_report(
        tmp_path,
        model_id="ddcolor",
        run_id="ddcolor_cuda_images_all",
        metrics={
            "colorfulness": _metric(1.23444),
            "lpips": _metric(0.1),
        },
    )
    _write_report(
        tmp_path,
        model_id="deoldify",
        run_id="deoldify_cuda_images_all",
        metrics={
            "colorfulness": _metric(1.23446),
            "lpips": _metric(None, status="skipped"),
        },
    )

    result = compare_benchmark_reports(
        project_root=tmp_path,
        metrics="colorfulness,lpips",
        precision=4,
        make_plots=False,
    )

    latex = Path(str(result["latex"])).read_text(encoding="utf-8")
    assert r"\begin{tabular}" in latex
    assert r"\textbf{1.234}" in latex
    assert r"\textbf{0.1}" in latex
    assert "--" in latex


def test_compare_reports_labels_reference_runs(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        model_id="cgan_reference",
        run_id="cgan_reference_fixed_by_title_cuda_images_all",
        reference_mode="fixed_by_title",
        metrics={"lpips": _metric(0.1)},
    )
    _write_report(
        tmp_path,
        model_id="cgan_reference",
        run_id="cgan_reference_previous_output_by_title_cuda_images_all",
        reference_mode="previous_output_by_title",
        metrics={"lpips": _metric(0.2)},
    )

    result = compare_benchmark_reports(
        project_root=tmp_path,
        metrics="lpips",
        make_plots=False,
    )

    latex = Path(str(result["latex"])).read_text(encoding="utf-8")
    assert r"cgan\_reference / fixed\_by\_title" in latex
    assert r"cgan\_reference / previous\_output\_by\_title" in latex


def test_compare_reports_cli_smoke(tmp_path: Path, monkeypatch: Any) -> None:
    _write_report(
        tmp_path,
        model_id="ddcolor",
        run_id="ddcolor_cuda_images_all",
        metrics={"lpips": _metric(0.2)},
    )
    monkeypatch.setattr(
        cli_module,
        "get_project_root",
        lambda path, levels_up=0: tmp_path,
    )

    result = cli_module.ColorizerCLI().compare_reports(
        metrics="lpips",
        make_plots=False,
    )

    assert Path(str(result["csv"])).exists()
    assert Path(str(result["latex"])).exists()
    assert Path(str(result["json"])).exists()
