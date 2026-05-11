import csv
import json
import logging
import statistics
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from illustration_colorizer.benchmark.datasets import (
    BenchmarkSample,
    load_benchmark_dataset_with_metadata,
)
from illustration_colorizer.benchmark.metrics import (
    BenchmarkMetric,
    ColorfulnessMetric,
    InkPreservationMetric,
    KidMetric,
    LinePreservationMetric,
    LpipsMetric,
)
from illustration_colorizer.benchmark.resources import ResourceMonitor
from illustration_colorizer.models import (
    ColorizationRequest,
    create_model_from_config,
    resolve_model_configs,
)
from shared.images import to_rgb_uint8
from shared.omegaconf import to_plain_mapping
from shared.paths import resolve_from_root

MetricFactory = Callable[[dict[str, Any]], BenchmarkMetric]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    factory: MetricFactory


@dataclass
class MetricReport:
    status: str
    sample_count: int
    value: Any = None
    reason: str | None = None


@dataclass
class SampleFailure:
    sample_id: str
    error_type: str
    message: str


@dataclass
class ModelReport:
    model_id: str
    metrics: dict[str, MetricReport]
    performance: dict[str, float | int | None]
    resources: dict[str, int | None]
    counts: dict[str, int]
    failures: list[SampleFailure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkRunReport:
    dataset: dict[str, Any]
    model_reports: list[ModelReport]
    output_dir: str


METRIC_SPECS: dict[str, MetricSpec] = {
    "colorfulness": MetricSpec(
        name="colorfulness",
        factory=lambda _: ColorfulnessMetric(),
    ),
    "ink_preservation_score": MetricSpec(
        name="ink_preservation_score",
        factory=lambda _: InkPreservationMetric(),
    ),
    "kid": MetricSpec(
        name="kid",
        factory=lambda cfg: KidMetric(
            device=str(cfg["runtime"]["device"]),
            kid_subset_size=int(cfg["metrics"]["kid_subset_size"]),
        ),
    ),
    "line_preservation_score": MetricSpec(
        name="line_preservation_score",
        factory=lambda _: LinePreservationMetric(),
    ),
    "lpips": MetricSpec(
        name="lpips",
        factory=lambda cfg: LpipsMetric(
            device=str(cfg["runtime"]["device"]),
            lpips_net=str(cfg["metrics"]["lpips_net"]),
            batch_size=int(cfg["metrics"]["lpips_batch_size"]),
        ),
    ),
}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _safe_float_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(next_prefix, nested, output)
        return

    output[prefix] = value


def _write_json_report(output_path: Path, report: BenchmarkRunReport) -> None:
    output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def _write_csv_report(output_path: Path, report: BenchmarkRunReport) -> None:
    rows: list[dict[str, Any]] = []
    field_names: set[str] = set()

    for model_report in report.model_reports:
        row: dict[str, Any] = {"model_id": model_report.model_id}
        _flatten("performance", model_report.performance, row)
        _flatten("resources", model_report.resources, row)
        _flatten("counts", model_report.counts, row)

        for metric_name, metric_report in model_report.metrics.items():
            _flatten(f"metrics.{metric_name}", asdict(metric_report), row)

        field_names.update(row.keys())
        rows.append(row)

    ordered_fields = sorted(field_names)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_path_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value
    ).strip("._") or "model"


def _write_per_model_reports(
    *,
    output_dir: Path,
    benchmark_config: dict[str, Any],
    dataset: dict[str, Any],
    model_reports: list[ModelReport],
    run_id: str,
) -> dict[str, dict[str, str]]:
    reports_root = output_dir / str(
        benchmark_config["report"].get("per_model_dir_name", "reports")
    )
    reports_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, str]] = {}
    for model_report in model_reports:
        model_dir = (
            reports_root
            / _safe_path_name(model_report.model_id)
            / _safe_path_name(run_id)
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        report = BenchmarkRunReport(
            dataset={**dataset, "run_id": run_id},
            model_reports=[model_report],
            output_dir=str(output_dir),
        )
        json_path = model_dir / str(benchmark_config["report"]["json_name"])
        csv_path = model_dir / str(benchmark_config["report"]["csv_name"])
        _write_json_report(json_path, report)
        _write_csv_report(csv_path, report)
        written[model_report.model_id] = {
            "json_report": str(json_path),
            "csv_report": str(csv_path),
        }
        LOGGER.info(
            "Saved model %s run %s benchmark JSON report to %s",
            model_report.model_id,
            run_id,
            json_path,
        )
        LOGGER.info(
            "Saved model %s run %s benchmark CSV report to %s",
            model_report.model_id,
            run_id,
            csv_path,
        )
    return written


def _benchmark_run_id(benchmark_config: dict[str, Any]) -> str:
    configured_run_id = benchmark_config["report"].get("run_id")
    if configured_run_id:
        return _safe_path_name(str(configured_run_id))
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_per_model_run_reports(
    *,
    output_dir: Path,
    benchmark_config: dict[str, Any],
    dataset: dict[str, Any],
    model_reports: list[ModelReport],
    run_id: str,
) -> dict[str, dict[str, str]]:
    runs_root = output_dir / str(
        benchmark_config["report"].get("per_run_dir_name", "runs")
    )
    written: dict[str, dict[str, str]] = {}
    for model_report in model_reports:
        model_dir = runs_root / run_id / _safe_path_name(model_report.model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        report = BenchmarkRunReport(
            dataset={**dataset, "run_id": run_id},
            model_reports=[model_report],
            output_dir=str(output_dir),
        )
        json_path = model_dir / str(benchmark_config["report"]["json_name"])
        csv_path = model_dir / str(benchmark_config["report"]["csv_name"])
        _write_json_report(json_path, report)
        _write_csv_report(csv_path, report)
        written[model_report.model_id] = {
            "json_report": str(json_path),
            "csv_report": str(csv_path),
        }
        LOGGER.info(
            "Saved model %s run %s benchmark JSON report to %s",
            model_report.model_id,
            run_id,
            json_path,
        )
        LOGGER.info(
            "Saved model %s run %s benchmark CSV report to %s",
            model_report.model_id,
            run_id,
            csv_path,
        )
    return written


def _chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [
        items[index : index + chunk_size]
        for index in range(0, len(items), chunk_size)
    ]


def _save_generated_images(
    *,
    output_dir: Path,
    model_id: str,
    generated_images: list[tuple[BenchmarkSample, np.ndarray]],
    max_saved_images: int,
) -> None:
    model_dir = output_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    for existing_path in model_dir.glob("*.png"):
        existing_path.unlink()
    manifest: dict[str, dict[str, int | bool]] = {}
    for sample, image in generated_images[:max_saved_images]:
        panel = _build_result_panel(sample, image)
        Image.fromarray(panel).save(model_dir / f"{sample.sample_id}.png")
        manifest[str(sample.sample_id)] = {
            "columns": 3 if sample.target_image is not None else 2,
            "has_ground_truth": sample.target_image is not None,
            "title": sample.metadata.get("title"),
            "reference_group_value": sample.metadata.get("reference_group_value"),
            "reference_mode": sample.metadata.get("reference_mode", "none"),
            "reference_sample_id": sample.metadata.get("reference_sample_id"),
            "reference_source": sample.metadata.get("reference_source", "none"),
        }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _resize_for_panel(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    return np.asarray(
        Image.fromarray(to_rgb_uint8(image)).resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )
    )


def _build_result_panel(
    sample: BenchmarkSample, output_image: np.ndarray
) -> np.ndarray:
    input_image = to_rgb_uint8(sample.input_image)
    height, width = input_image.shape[:2]

    panels = [input_image]

    if sample.target_image is not None:
        panels.append(
            _resize_for_panel(sample.target_image, width=width, height=height)
        )

    panels.append(_resize_for_panel(output_image, width=width, height=height))

    return np.concatenate(panels, axis=1)


def _record_success(
    *,
    sample: BenchmarkSample,
    result: Any,
    latency: float,
    latencies: list[float],
    warnings: list[str],
    records: list[dict[str, Any]],
    generated_images: list[tuple[BenchmarkSample, np.ndarray]],
) -> None:
    latencies.append(latency)
    warnings.extend(result.warnings)
    records.append(
        {
            "sample_id": sample.sample_id,
            "input_image": sample.input_image,
            "output_image": result.image,
            "target_image": sample.target_image,
        }
    )
    generated_images.append((sample, result.image))


def _sample_with_reference_metadata(
    sample: BenchmarkSample,
    *,
    reference_sample_id: str | None,
    reference_source: str,
) -> BenchmarkSample:
    metadata = dict(sample.metadata)
    metadata["reference_sample_id"] = reference_sample_id
    metadata["reference_source"] = reference_source
    return BenchmarkSample(
        sample_id=sample.sample_id,
        input_image=sample.input_image,
        target_image=sample.target_image,
        reference_image=sample.reference_image,
        metadata=metadata,
    )


def _make_request(
    sample: BenchmarkSample,
    *,
    reference_image: np.ndarray | None = None,
) -> ColorizationRequest:
    return ColorizationRequest(
        input_image=sample.input_image,
        reference_image=(
            reference_image if reference_image is not None else sample.reference_image
        ),
        sample_id=sample.sample_id,
    )


def _record_failure(
    *,
    sample: BenchmarkSample,
    model_id: str,
    exc: Exception,
    failures: list[SampleFailure],
    fail_fast: bool,
) -> None:
    LOGGER.warning(
        "Sample %s failed on model %s: %s: %s",
        sample.sample_id,
        model_id,
        type(exc).__name__,
        exc,
    )
    failures.append(
        SampleFailure(
            sample_id=sample.sample_id,
            error_type=type(exc).__name__,
            message=str(exc),
        )
    )
    if fail_fast:
        raise exc


def _serialize_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_metric_value(nested) for key, nested in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _log_metric_summary(model_id: str, metric_reports: dict[str, MetricReport]) -> None:
    if not metric_reports:
        LOGGER.info("Metrics for %s: none configured", model_id)
        return

    parts: list[str] = []
    for metric_name, metric_report in metric_reports.items():
        if metric_report.status == "computed":
            parts.append(
                f"{metric_name}={_format_metric_value(metric_report.value)} "
                f"(n={metric_report.sample_count})"
            )
        else:
            reason = metric_report.reason or metric_report.status
            parts.append(f"{metric_name}=skipped ({reason})")
    LOGGER.info("Metrics for %s: %s", model_id, "; ".join(parts))


def _build_metric_report(
    *,
    metric_spec: MetricSpec,
    benchmark_config: dict[str, Any],
    records: list[dict[str, Any]],
) -> MetricReport:
    metric = metric_spec.factory(benchmark_config)
    if not isinstance(metric, BenchmarkMetric):
        raise TypeError(
            f"Metric factory for {metric_spec.name} must return BenchmarkMetric."
        )

    if metric.requires_ground_truth:
        metric_records = [
            record for record in records if record["target_image"] is not None
        ]
        if not metric_records:
            return MetricReport(
                status="skipped",
                sample_count=0,
                reason="Ground truth is not available for this metric.",
            )
    else:
        metric_records = records
        if not metric_records:
            return MetricReport(
                status="skipped",
                sample_count=0,
                reason="No successful samples available for this metric.",
            )

    x_images = [record["input_image"] for record in metric_records]
    y_images = [record["output_image"] for record in metric_records]
    g_images = [record["target_image"] for record in metric_records]

    try:
        if metric.requires_ground_truth:
            value = metric.compute(
                x_images=x_images,
                y_images=y_images,
                g_images=g_images,
            )
        else:
            value = metric.compute(
                x_images=x_images,
                y_images=y_images,
                g_images=None,
            )
    except ImportError as exc:
        return MetricReport(
            status="skipped",
            sample_count=len(metric_records),
            reason=str(exc),
        )
    except ValueError as exc:
        return MetricReport(
            status="skipped",
            sample_count=len(metric_records),
            reason=str(exc),
        )

    return MetricReport(
        status="computed",
        sample_count=len(metric_records),
        value=_serialize_metric_value(value),
    )


def _run_single_model(
    *,
    project_root: Path,
    benchmark_config: dict[str, Any],
    model_config: dict[str, Any],
    samples: list[BenchmarkSample],
    report_dir: Path,
) -> ModelReport:
    model = create_model_from_config(model_config, project_root=project_root)
    per_sample_logging = bool(benchmark_config["logging"]["per_sample"])
    batch_size = int(benchmark_config["runtime"].get("batch_size", 1))
    reference_config = dict(benchmark_config.get("reference") or {})
    reference_mode = str(reference_config.get("mode", "none"))
    reference_group_key = str(reference_config.get("group_key", "title"))
    latencies: list[float] = []
    failures: list[SampleFailure] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    generated_images: list[tuple[BenchmarkSample, np.ndarray]] = []
    collect_resources = bool(benchmark_config["runtime"]["collect_resources"])
    poll_interval_seconds = float(
        benchmark_config["runtime"]["resource_poll_interval_seconds"]
    )

    model_start = time.perf_counter()
    monitor = ResourceMonitor(poll_interval_seconds) if collect_resources else None
    context = monitor if monitor is not None else _NullContext()

    try:
        with context:
            LOGGER.info("Loading model %s", model.model_id)
            model.load()
            LOGGER.info("Loaded model %s", model.model_id)
            batch_samples_source = samples
            if reference_mode == "previous_output_by_title":
                reference_state: dict[
                    str, tuple[np.ndarray | None, str | None, str]
                ] = {}
                for sample_index, sample in enumerate(samples, start=1):
                    title = str(sample.metadata.get(reference_group_key))
                    if title not in reference_state:
                        reference_state[title] = (
                            sample.reference_image,
                            sample.metadata.get("reference_sample_id"),
                            str(sample.metadata.get("reference_source", "gt_seed")),
                        )
                    reference_image, reference_sample_id, reference_source = (
                        reference_state[title]
                    )
                    sample_for_record = _sample_with_reference_metadata(
                        sample,
                        reference_sample_id=reference_sample_id,
                        reference_source=reference_source,
                    )
                    if per_sample_logging:
                        LOGGER.info(
                            "Running model %s on sample %s (%d/%d) with %s reference",
                            model.model_id,
                            sample.sample_id,
                            sample_index,
                            len(samples),
                            reference_source,
                        )
                    sample_started = time.perf_counter()
                    try:
                        result = model.colorize(
                            _make_request(sample, reference_image=reference_image)
                        )
                    except Exception as sample_exc:
                        _record_failure(
                            sample=sample_for_record,
                            model_id=model.model_id,
                            exc=sample_exc,
                            failures=failures,
                            fail_fast=bool(benchmark_config["runtime"]["fail_fast"]),
                        )
                        continue

                    latency = (
                        float(result.latency_seconds)
                        if result.latency_seconds is not None
                        else time.perf_counter() - sample_started
                    )
                    if per_sample_logging:
                        LOGGER.info(
                            "Completed model %s on sample %s in %.3fs",
                            model.model_id,
                            sample.sample_id,
                            latency,
                        )
                    _record_success(
                        sample=sample_for_record,
                        result=result,
                        latency=latency,
                        latencies=latencies,
                        warnings=warnings,
                        records=records,
                        generated_images=generated_images,
                    )
                    reference_state[title] = (
                        result.image,
                        sample.sample_id,
                        "previous_output",
                    )
                batch_samples_source = []

            requests_with_samples = [
                (sample, _make_request(sample)) for sample in batch_samples_source
            ]
            for batch_start, batch_items in enumerate(
                _chunked(requests_with_samples, batch_size),
                start=0,
            ):
                batch_samples = [sample for sample, _ in batch_items]
                batch_requests = [request for _, request in batch_items]
                batch_offset = batch_start * batch_size
                if per_sample_logging:
                    sample_range = ", ".join(
                        sample.sample_id for sample in batch_samples
                    )
                    LOGGER.info(
                        "Running model %s on batch %d (%d-%d/%d): %s",
                        model.model_id,
                        batch_start + 1,
                        batch_offset + 1,
                        batch_offset + len(batch_items),
                        len(samples),
                        sample_range,
                    )
                batch_started = time.perf_counter()
                try:
                    batch_results = model.colorize_batch(batch_requests)
                except Exception as exc:
                    if len(batch_items) == 1:
                        sample = batch_items[0][0]
                        _record_failure(
                            sample=sample,
                            model_id=model.model_id,
                            exc=exc,
                            failures=failures,
                            fail_fast=bool(benchmark_config["runtime"]["fail_fast"]),
                        )
                        continue

                    LOGGER.warning(
                        "Batch %d failed on model %s: %s: %s. "
                        "Falling back to per-sample execution.",
                        batch_start + 1,
                        model.model_id,
                        type(exc).__name__,
                        exc,
                    )
                    for inner_offset, (sample, request) in enumerate(
                        batch_items,
                        start=1,
                    ):
                        if per_sample_logging:
                            LOGGER.info(
                                "Retrying model %s on sample %s (%d/%d)",
                                model.model_id,
                                sample.sample_id,
                                batch_offset + inner_offset,
                                len(samples),
                            )
                        sample_started = time.perf_counter()
                        try:
                            result = model.colorize(request)
                        except Exception as sample_exc:
                            _record_failure(
                                sample=sample,
                                model_id=model.model_id,
                                exc=sample_exc,
                                failures=failures,
                                fail_fast=bool(
                                    benchmark_config["runtime"]["fail_fast"]
                                ),
                            )
                            continue

                        latency = (
                            float(result.latency_seconds)
                            if result.latency_seconds is not None
                            else time.perf_counter() - sample_started
                        )
                        if per_sample_logging:
                            LOGGER.info(
                                "Completed model %s on sample %s in %.3fs",
                                model.model_id,
                                sample.sample_id,
                                latency,
                            )
                        _record_success(
                            sample=sample,
                            result=result,
                            latency=latency,
                            latencies=latencies,
                            warnings=warnings,
                            records=records,
                            generated_images=generated_images,
                        )
                    continue

                if len(batch_results) != len(batch_items):
                    raise ValueError(
                        f"Model {model.model_id} returned "
                        f"{len(batch_results)} results for "
                        f"{len(batch_items)} requests."
                    )

                batch_elapsed = time.perf_counter() - batch_started
                fallback_latency = batch_elapsed / len(batch_results)
                for (sample, _), result in zip(
                    batch_items,
                    batch_results,
                    strict=True,
                ):
                    latency = (
                        float(result.latency_seconds)
                        if result.latency_seconds is not None
                        else fallback_latency
                    )
                    if per_sample_logging:
                        LOGGER.info(
                            "Completed model %s on sample %s in %.3fs",
                            model.model_id,
                            sample.sample_id,
                            latency,
                        )
                    _record_success(
                        sample=sample,
                        result=result,
                        latency=latency,
                        latencies=latencies,
                        warnings=warnings,
                        records=records,
                        generated_images=generated_images,
                    )
    except Exception as exc:
        LOGGER.error(
            "Model %s failed before metrics computation: %s: %s",
            model.model_id,
            type(exc).__name__,
            exc,
        )
        failure = SampleFailure(
            sample_id="__model__",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return ModelReport(
            model_id=model.model_id,
            metrics={
                str(metric_name): MetricReport(
                    status="skipped",
                    sample_count=0,
                    reason="Model failed before metrics computation.",
                )
                for metric_name in benchmark_config["metrics"]["enabled"]
            },
            performance={
                "total_runtime_seconds": None,
                "mean_latency_seconds": None,
                "median_latency_seconds": None,
                "p95_latency_seconds": None,
                "throughput_images_per_second": None,
            },
            resources={
                "peak_cpu_rss_bytes": None,
                "peak_gpu_memory_bytes": None,
            },
            counts={
                "total_samples": len(samples),
                "successful_samples": 0,
                "failed_samples": len(samples),
                "samples_with_ground_truth": sum(
                    1 for sample in samples if sample.target_image is not None
                ),
            },
            failures=[failure],
            warnings=[],
        )
    finally:
        try:
            LOGGER.info("Unloading model %s", model.model_id)
            model.unload()
        except Exception:
            pass

    runtime_seconds = time.perf_counter() - model_start
    usage = monitor.snapshot() if monitor is not None else None

    if bool(benchmark_config["report"]["save_images"]):
        generated_root = report_dir / str(
            benchmark_config["report"]["generated_dir_name"]
        )
        _save_generated_images(
            output_dir=generated_root,
            model_id=model.model_id,
            generated_images=generated_images,
            max_saved_images=int(benchmark_config["report"]["max_saved_images"]),
        )

    metric_reports: dict[str, MetricReport] = {}
    for metric_name in benchmark_config["metrics"]["enabled"]:
        metric_spec = METRIC_SPECS.get(str(metric_name))
        if metric_spec is None:
            metric_reports[str(metric_name)] = MetricReport(
                status="skipped",
                sample_count=0,
                reason="Metric is not registered.",
            )
            continue
        metric_reports[str(metric_name)] = _build_metric_report(
            metric_spec=metric_spec,
            benchmark_config=benchmark_config,
            records=records,
        )
    _log_metric_summary(model.model_id, metric_reports)

    success_count = len(records)
    total_samples = len(samples)
    performance = {
        "total_runtime_seconds": runtime_seconds,
        "mean_latency_seconds": _safe_float_mean(latencies),
        "median_latency_seconds": (
            float(statistics.median(latencies)) if latencies else None
        ),
        "p95_latency_seconds": _percentile(latencies, 95.0),
        "throughput_images_per_second": (
            float(success_count / runtime_seconds) if runtime_seconds > 0.0 else None
        ),
    }
    resources = {
        "peak_cpu_rss_bytes": usage.peak_cpu_rss_bytes if usage is not None else None,
        "peak_gpu_memory_bytes": (
            usage.peak_gpu_memory_bytes if usage is not None else None
        ),
    }
    counts = {
        "total_samples": total_samples,
        "successful_samples": success_count,
        "failed_samples": len(failures),
        "samples_with_ground_truth": sum(
            1 for sample in samples if sample.target_image is not None
        ),
    }

    deduped_warnings = sorted(set(warnings))
    return ModelReport(
        model_id=model.model_id,
        metrics=metric_reports,
        performance=performance,
        resources=resources,
        counts=counts,
        failures=failures,
        warnings=deduped_warnings,
    )


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def _configure_logging(benchmark_config: dict[str, Any]) -> None:
    level_name = str(benchmark_config["logging"]["level"]).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _configure_warning_filters(benchmark_config: dict[str, Any]) -> None:
    if not bool(benchmark_config["warnings"]["suppress_known"]):
        return

    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Your training set is empty\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Your validation set is empty\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"Arguments other than a weight enum or `None` for 'weights' "
            r"are deprecated.*"
        ),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"Metric `Kernel Inception Distance` will save all extracted "
            r"features in buffer\..*"
        ),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"The torch\.cuda\.\*DtypeTensor constructors are no longer "
            r"recommended\..*"
        ),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Conversion from CIE-LAB, via XYZ to sRGB color space resulted in .*",
        category=UserWarning,
    )


def run_benchmark(
    *,
    project_root: Path,
    config: DictConfig,
) -> dict[str, object]:
    config_dict = to_plain_mapping(config)
    benchmark_config = to_plain_mapping(config_dict["benchmark"])
    _configure_logging(benchmark_config)
    _configure_warning_filters(benchmark_config)
    model_configs = resolve_model_configs(
        config_dict["models"],
        selected_models=list(benchmark_config.get("selected_models") or []),
    )
    LOGGER.info(
        "Starting benchmark for models: %s",
        ", ".join(str(model_cfg["model_id"]) for model_cfg in model_configs),
    )
    benchmark_dataset = load_benchmark_dataset_with_metadata(
        project_root=project_root,
        dataset_config=benchmark_config["dataset"],
        reference_config=benchmark_config.get("reference"),
    )
    samples = benchmark_dataset.samples
    LOGGER.info(
        "Loaded %d benchmark samples from source=%s",
        len(samples),
        benchmark_config["dataset"]["source"],
    )

    output_dir = resolve_from_root(
        project_root,
        str(benchmark_config["report"]["output_dir"]),
    )
    if output_dir is None:
        raise ValueError("Benchmark output directory must not be null.")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_reports = [
        _run_single_model(
            project_root=project_root,
            benchmark_config=benchmark_config,
            model_config=model_config,
            samples=samples,
            report_dir=output_dir,
        )
        for model_config in model_configs
    ]

    run_id = _benchmark_run_id(benchmark_config)
    dataset_report = {
        "source": benchmark_config["dataset"]["source"],
        "sample_count": len(samples),
        "samples_with_ground_truth": sum(
            1 for sample in samples if sample.target_image is not None
        ),
        "run_id": run_id,
        **benchmark_dataset.metadata,
    }
    report = BenchmarkRunReport(
        dataset=dataset_report,
        model_reports=model_reports,
        output_dir=str(output_dir),
    )

    json_path = output_dir / str(benchmark_config["report"]["json_name"])
    csv_path = output_dir / str(benchmark_config["report"]["csv_name"])
    _write_json_report(json_path, report)
    _write_csv_report(csv_path, report)
    per_model_reports = _write_per_model_reports(
        output_dir=output_dir,
        benchmark_config=benchmark_config,
        dataset=dataset_report,
        model_reports=model_reports,
        run_id=run_id,
    )
    model_run_reports = _write_per_model_run_reports(
        output_dir=output_dir,
        benchmark_config=benchmark_config,
        dataset=dataset_report,
        model_reports=model_reports,
        run_id=run_id,
    )
    LOGGER.info("Saved benchmark JSON report to %s", json_path)
    LOGGER.info("Saved benchmark CSV report to %s", csv_path)

    return {
        "json_report": str(json_path),
        "csv_report": str(csv_path),
        "run_id": run_id,
        "model_reports": per_model_reports,
        "model_run_reports": model_run_reports,
        "sample_count": len(samples),
        "models": [report.model_id for report in model_reports],
    }
