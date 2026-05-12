from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from illustration_colorizer.benchmark.datasets import (
    BenchmarkSample,
    load_hf_arrow_benchmark_dataset,
)
from illustration_colorizer.benchmark.runner import _run_single_model
from illustration_colorizer.models.base import (
    ColorizationModel,
    ColorizationRequest,
    ColorizationResult,
)


class _FakeImage:
    def __init__(self, value: int) -> None:
        self.value = value

    def convert(self, mode: str) -> np.ndarray:
        assert mode == "RGB"
        return np.full((2, 2, 3), self.value, dtype=np.uint8)


class _FakeDataset:
    column_names = ["bw_image", "color_image", "title", "tags"]

    def __init__(self) -> None:
        titles = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        self.rows = [
            {
                "bw_image": _FakeImage(index),
                "color_image": _FakeImage(100 + index),
                "title": title,
                "tags": "",
            }
            for index, title in enumerate(titles)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


def _patch_fake_arrow_dataset(monkeypatch, tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "arrow"
    dataset_dir.mkdir()
    (dataset_dir / "dummy.arrow").write_bytes(b"placeholder")

    def fake_load_dataset(*args: Any, **kwargs: Any) -> dict[str, _FakeDataset]:
        return {"train": _FakeDataset()}

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return dataset_dir


def test_none_reference_mode_balances_titles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_dir = _patch_fake_arrow_dataset(monkeypatch, tmp_path)

    dataset = load_hf_arrow_benchmark_dataset(
        project_root=tmp_path,
        dataset_dir=str(dataset_dir),
        limit=5,
        reference_config={
            "mode": "none",
            "group_key": "title",
            "sampling": "balanced_titles",
        },
    )

    assert [sample.sample_id for sample in dataset.samples] == [
        "0",
        "3",
        "6",
        "1",
        "4",
    ]
    assert [sample.metadata["title"] for sample in dataset.samples] == [
        "A",
        "B",
        "C",
        "A",
        "B",
    ]
    assert all(sample.reference_image is None for sample in dataset.samples)
    assert dataset.metadata["title_count"] == 3
    assert dataset.metadata["excluded_seed_samples"] == 0


def test_fixed_reference_mode_uses_first_color_image_per_title(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_dir = _patch_fake_arrow_dataset(monkeypatch, tmp_path)

    dataset = load_hf_arrow_benchmark_dataset(
        project_root=tmp_path,
        dataset_dir=str(dataset_dir),
        limit=3,
        reference_config={
            "mode": "fixed_by_title",
            "group_key": "title",
            "sampling": "balanced_titles",
        },
    )

    assert [sample.sample_id for sample in dataset.samples] == ["1", "4", "7"]
    assert [sample.metadata["title"] for sample in dataset.samples] == ["A", "B", "C"]
    assert [sample.metadata["reference_sample_id"] for sample in dataset.samples] == [
        "0",
        "3",
        "6",
    ]
    assert [int(sample.reference_image[0, 0, 0]) for sample in dataset.samples] == [
        100,
        103,
        106,
    ]
    assert dataset.metadata["title_count"] == 3
    assert dataset.metadata["excluded_seed_samples"] == 3


def test_previous_output_mode_marks_seed_reference_and_balances_titles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_dir = _patch_fake_arrow_dataset(monkeypatch, tmp_path)

    dataset = load_hf_arrow_benchmark_dataset(
        project_root=tmp_path,
        dataset_dir=str(dataset_dir),
        limit=5,
        reference_config={
            "mode": "previous_output_by_title",
            "group_key": "title",
            "sampling": "balanced_titles",
        },
    )

    assert [sample.sample_id for sample in dataset.samples] == [
        "1",
        "4",
        "7",
        "2",
        "5",
    ]
    assert all(
        sample.metadata["reference_source"] == "gt_seed"
        for sample in dataset.samples
    )
    assert dataset.metadata["title_count"] == 3


class _PreviousReferenceRecorder(ColorizationModel):
    seen_references: list[int] = []

    def load(self) -> None:
        self.seen_references = []

    def unload(self) -> None:
        return None

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        assert request.reference_image is not None
        reference_value = int(request.reference_image[0, 0, 0])
        self.seen_references.append(reference_value)
        output = np.full((2, 2, 3), 200 + len(self.seen_references), dtype=np.uint8)
        return ColorizationResult(image=output, model_id=self.model_id)

    def colorize_batch(
        self, requests: list[ColorizationRequest]
    ) -> list[ColorizationResult]:
        raise AssertionError("previous_output_by_title must not use batch execution")


def test_previous_output_runner_uses_model_output_as_next_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = _PreviousReferenceRecorder({"model_id": "mock_reference"})

    def fake_create_model_from_config(
        model_config: dict[str, Any],
        *,
        project_root: Path,
    ) -> ColorizationModel:
        return model

    monkeypatch.setattr(
        "illustration_colorizer.benchmark.runner.create_model_from_config",
        fake_create_model_from_config,
    )
    samples = [
        BenchmarkSample(
            sample_id="1",
            input_image=np.zeros((2, 2, 3), dtype=np.uint8),
            target_image=np.zeros((2, 2, 3), dtype=np.uint8),
            reference_image=np.full((2, 2, 3), 100, dtype=np.uint8),
            metadata={
                "title": "A",
                "reference_mode": "previous_output_by_title",
                "reference_sample_id": "0",
                "reference_source": "gt_seed",
            },
        ),
        BenchmarkSample(
            sample_id="2",
            input_image=np.zeros((2, 2, 3), dtype=np.uint8),
            target_image=np.zeros((2, 2, 3), dtype=np.uint8),
            reference_image=np.full((2, 2, 3), 100, dtype=np.uint8),
            metadata={
                "title": "A",
                "reference_mode": "previous_output_by_title",
                "reference_sample_id": "0",
                "reference_source": "gt_seed",
            },
        ),
    ]

    report = _run_single_model(
        project_root=tmp_path,
        benchmark_config={
            "reference": {
                "mode": "previous_output_by_title",
                "group_key": "title",
            },
            "logging": {"per_sample": False},
            "runtime": {
                "batch_size": 8,
                "collect_resources": False,
                "resource_poll_interval_seconds": 0.05,
                "fail_fast": False,
                "device": "cpu",
            },
            "report": {
                "save_images": True,
                "generated_dir_name": "generated",
                "max_saved_images": 10,
            },
            "metrics": {"enabled": []},
        },
        model_config={"model_id": "mock_reference"},
        samples=samples,
        report_dir=tmp_path,
    )

    assert report.counts["successful_samples"] == 2
    assert model.seen_references == [100, 201]


class _FailingPreviousReferenceRecorder(_PreviousReferenceRecorder):
    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        if request.sample_id == "2":
            raise RuntimeError("synthetic failure")
        return super().colorize(request)


def test_previous_output_runner_keeps_last_successful_reference_after_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = _FailingPreviousReferenceRecorder({"model_id": "mock_reference"})

    def fake_create_model_from_config(
        model_config: dict[str, Any],
        *,
        project_root: Path,
    ) -> ColorizationModel:
        return model

    monkeypatch.setattr(
        "illustration_colorizer.benchmark.runner.create_model_from_config",
        fake_create_model_from_config,
    )
    samples = [
        BenchmarkSample(
            sample_id=sample_id,
            input_image=np.zeros((2, 2, 3), dtype=np.uint8),
            target_image=np.zeros((2, 2, 3), dtype=np.uint8),
            reference_image=np.full((2, 2, 3), 100, dtype=np.uint8),
            metadata={
                "title": "A",
                "reference_mode": "previous_output_by_title",
                "reference_sample_id": "0",
                "reference_source": "gt_seed",
            },
        )
        for sample_id in ["1", "2", "3"]
    ]

    report = _run_single_model(
        project_root=tmp_path,
        benchmark_config={
            "reference": {
                "mode": "previous_output_by_title",
                "group_key": "title",
            },
            "logging": {"per_sample": False},
            "runtime": {
                "batch_size": 8,
                "collect_resources": False,
                "resource_poll_interval_seconds": 0.05,
                "fail_fast": False,
                "device": "cpu",
            },
            "report": {
                "save_images": False,
                "generated_dir_name": "generated",
                "max_saved_images": 10,
            },
            "metrics": {"enabled": []},
        },
        model_config={"model_id": "mock_reference"},
        samples=samples,
        report_dir=tmp_path,
    )

    assert report.counts["successful_samples"] == 2
    assert report.counts["failed_samples"] == 1
    assert model.seen_references == [100, 201]
