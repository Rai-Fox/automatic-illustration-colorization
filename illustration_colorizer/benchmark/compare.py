from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from shared.paths import resolve_from_root

DEFAULT_METRICS = [
    "colorfulness",
    "line_preservation_score",
    "ink_preservation_score",
    "lpips",
    "kid.kid_mean",
    "kid.kid_std",
]

METRIC_DIRECTIONS = {
    "colorfulness": "max",
    "line_preservation_score": "max",
    "ink_preservation_score": "max",
    "lpips": "min",
    "kid.kid_mean": "min",
    "kid.kid_std": "min",
}

REFERENCE_MODE_ORDER = {
    "none": 0,
    "fixed_by_title": 1,
    "previous_output_by_title": 2,
}


@dataclass(frozen=True)
class ComparisonEntry:
    label: str
    model_id: str
    run_id: str
    report_path: Path
    dataset: dict[str, Any]
    counts: dict[str, Any]
    metrics: dict[str, Any]


def _split_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = list(value)
    return [str(item).strip() for item in values if str(item).strip()]


def _metric_value(metric_report: dict[str, Any] | None, metric_name: str) -> Any:
    if not metric_report or metric_report.get("status") != "computed":
        return None

    value = metric_report.get("value")
    nested_keys = metric_name.split(".")[1:]
    for key in nested_keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _short_label(model_id: str, run_id: str, dataset: dict[str, Any]) -> str:
    reference_mode = str(dataset.get("reference_mode") or "none")
    if reference_mode != "none":
        return f"{model_id} / {reference_mode}"
    return model_id


def _read_report_entries(
    report_path: Path,
    metrics: list[str],
) -> list[ComparisonEntry]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = dict(report.get("dataset") or {})
    run_id = str(dataset.get("run_id") or report_path.parent.name)
    entries: list[ComparisonEntry] = []

    for model_report in report.get("model_reports") or []:
        model_id = str(model_report.get("model_id") or report_path.parent.parent.name)
        metric_reports = dict(model_report.get("metrics") or {})
        flattened_metrics = {
            metric_name: _metric_value(
                metric_reports.get(metric_name.split(".", 1)[0]),
                metric_name,
            )
            for metric_name in metrics
        }
        entries.append(
            ComparisonEntry(
                label=_short_label(model_id, run_id, dataset),
                model_id=model_id,
                run_id=run_id,
                report_path=report_path,
                dataset=dataset,
                counts=dict(model_report.get("counts") or {}),
                metrics=flattened_metrics,
            )
        )

    return entries


def _entry_sort_key(entry: ComparisonEntry) -> tuple[str, int, str, str]:
    reference_mode = str(entry.dataset.get("reference_mode") or "none")
    return (
        entry.model_id,
        REFERENCE_MODE_ORDER.get(reference_mode, 100),
        reference_mode,
        entry.run_id,
    )


def _dedupe_labels(entries: list[ComparisonEntry]) -> list[ComparisonEntry]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.label] = counts.get(entry.label, 0) + 1

    deduped = []
    for entry in entries:
        if counts[entry.label] <= 1:
            deduped.append(entry)
            continue
        deduped.append(
            ComparisonEntry(
                label=f"{entry.label} ({entry.run_id})",
                model_id=entry.model_id,
                run_id=entry.run_id,
                report_path=entry.report_path,
                dataset=entry.dataset,
                counts=entry.counts,
                metrics=entry.metrics,
            )
        )
    return deduped


def _rounded_for_best(value: Any, precision: int) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return float(f"{numeric:.{precision}g}")


def _best_labels(
    entries: list[ComparisonEntry],
    *,
    metric_name: str,
    precision: int,
) -> set[str]:
    direction = METRIC_DIRECTIONS.get(metric_name, "max")
    rounded_values = {
        entry.label: _rounded_for_best(entry.metrics.get(metric_name), precision)
        for entry in entries
    }
    valid_values = [
        value for value in rounded_values.values() if value is not None
    ]
    if not valid_values:
        return set()

    best_value = min(valid_values) if direction == "min" else max(valid_values)
    return {
        label
        for label, value in rounded_values.items()
        if value is not None and value == best_value
    }


def _best_by_metric(
    entries: list[ComparisonEntry],
    *,
    metrics: list[str],
    precision: int,
) -> dict[str, set[str]]:
    return {
        metric_name: _best_labels(
            entries,
            metric_name=metric_name,
            precision=precision,
        )
        for metric_name in metrics
    }


def _comparison_rows(
    entries: list[ComparisonEntry],
    *,
    metrics: list[str],
    best_by_metric: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        row: dict[str, Any] = {
            "label": entry.label,
            "model_id": entry.model_id,
            "run_id": entry.run_id,
            "sample_count": entry.dataset.get("sample_count"),
            "successful_samples": entry.counts.get("successful_samples"),
            "report_path": str(entry.report_path),
        }
        for metric_name in metrics:
            row[metric_name] = entry.metrics.get(metric_name)
            row[f"{metric_name}.is_best"] = entry.label in best_by_metric[metric_name]
        rows.append(row)
    return rows


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _format_metric(value: Any, precision: int) -> str:
    if value is None:
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _latex_escape(value)
    if math.isnan(numeric):
        return "--"
    return f"{numeric:.{precision}g}"


def _metric_heading(metric_name: str) -> str:
    return {
        "kid.kid_mean": "kid mean",
        "kid.kid_std": "kid std",
    }.get(metric_name, metric_name.replace("_", " "))


def render_latex_table(
    entries: list[ComparisonEntry],
    *,
    metrics: list[str],
    best_by_metric: dict[str, set[str]],
    precision: int = 4,
    include_runs: bool = True,
) -> str:
    headers = ["Model"]
    if include_runs:
        headers.append("Run")
    headers.extend(["Samples", *[_metric_heading(metric) for metric in metrics]])

    column_spec = "l" + ("l" if include_runs else "") + "r" + ("r" * len(metrics))
    lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        "\\hline",
    ]

    for entry in entries:
        cells = [_latex_escape(entry.label)]
        if include_runs:
            cells.append(_latex_escape(entry.run_id))
        cells.append(_latex_escape(entry.counts.get("successful_samples", "--")))
        for metric_name in metrics:
            rendered_value = _format_metric(entry.metrics.get(metric_name), precision)
            if entry.label in best_by_metric[metric_name] and rendered_value != "--":
                rendered_value = f"\\textbf{{{rendered_value}}}"
            cells.append(rendered_value)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _write_barplots(
    entries: list[ComparisonEntry],
    *,
    metrics: list[str],
    best_by_metric: dict[str, set[str]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not entries or not metrics:
        return

    column_count = min(3, len(metrics))
    row_count = math.ceil(len(metrics) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 3.8 * row_count),
        squeeze=False,
    )
    labels = [entry.label for entry in entries]

    for metric_index, metric_name in enumerate(metrics):
        axis = axes[metric_index // column_count][metric_index % column_count]
        values = [
            float(entry.metrics[metric_name])
            if entry.metrics.get(metric_name) is not None
            else math.nan
            for entry in entries
        ]
        colors = [
            "#2f80ed" if label in best_by_metric[metric_name] else "#8a96a3"
            for label in labels
        ]
        axis.bar(labels, values, color=colors)
        axis.set_title(_metric_heading(metric_name))
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)

    for index in range(len(metrics), row_count * column_count):
        axes[index // column_count][index % column_count].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _resolve_output_dir(benchmark_root: Path, output_dir: str) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return benchmark_root / output_path


def compare_benchmark_reports(
    *,
    project_root: Path,
    benchmark_output_dir: str = "outputs/benchmark",
    report_name: str = "report_all.json",
    models: str | list[str] | None = None,
    output_dir: str = "comparison_reports",
    metrics: str | list[str] | None = None,
    precision: int = 4,
    include_runs: bool = True,
    make_plots: bool = True,
) -> dict[str, object]:
    benchmark_root = resolve_from_root(project_root, benchmark_output_dir)
    if benchmark_root is None or not benchmark_root.exists():
        raise FileNotFoundError(
            f"Benchmark output directory not found: {benchmark_output_dir}"
        )

    reports_root = benchmark_root / "reports"
    if not reports_root.exists():
        raise FileNotFoundError(
            f"Benchmark reports directory not found: {reports_root}"
        )

    selected_models = set(_split_csv(models))
    selected_metrics = _split_csv(metrics) or DEFAULT_METRICS
    output_root = _resolve_output_dir(benchmark_root, output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    entries: list[ComparisonEntry] = []
    for report_path in sorted(reports_root.rglob(report_name)):
        for entry in _read_report_entries(report_path, selected_metrics):
            if selected_models and entry.model_id not in selected_models:
                continue
            entries.append(entry)

    entries = _dedupe_labels(sorted(entries, key=_entry_sort_key))
    if not entries:
        raise FileNotFoundError(
            f"No benchmark reports named {report_name!r} found for selected models."
        )

    best_by_metric = _best_by_metric(
        entries,
        metrics=selected_metrics,
        precision=precision,
    )
    rows = _comparison_rows(
        entries,
        metrics=selected_metrics,
        best_by_metric=best_by_metric,
    )
    frame = pd.DataFrame(rows)

    csv_path = output_root / "comparison.csv"
    tex_path = output_root / "comparison.tex"
    json_path = output_root / "comparison.json"
    plot_path = output_root / "metrics_barplots.png"

    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        render_latex_table(
            entries,
            metrics=selected_metrics,
            best_by_metric=best_by_metric,
            precision=precision,
            include_runs=include_runs,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "report_name": report_name,
                "metrics": selected_metrics,
                "best_by_metric": {
                    metric_name: sorted(labels)
                    for metric_name, labels in best_by_metric.items()
                },
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if make_plots:
        _write_barplots(
            entries,
            metrics=selected_metrics,
            best_by_metric=best_by_metric,
            output_path=plot_path,
        )

    return {
        "csv": str(csv_path),
        "latex": str(tex_path),
        "json": str(json_path),
        "plot": str(plot_path) if make_plots else None,
        "row_count": len(rows),
    }
