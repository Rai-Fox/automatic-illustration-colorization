from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobFiles:
    input_path: Path
    reference_paths: list[Path]
    result_path: Path


class JobFileStorage:
    def __init__(self, storage_root: str | Path) -> None:
        self.storage_root = Path(storage_root)

    def save_job_inputs(
        self,
        *,
        job_id: str,
        image_bytes: bytes,
        reference_image_bytes: bytes | None = None,
        reference_images_bytes: list[bytes] | None = None,
    ) -> JobFiles:
        job_dir = self.storage_root / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        LOGGER.info("created job storage directory job_id=%s path=%s", job_id, job_dir)

        input_path = job_dir / "input.png"
        input_path.write_bytes(image_bytes)
        LOGGER.info(
            "saved job input image job_id=%s path=%s size_bytes=%s",
            job_id,
            input_path,
            len(image_bytes),
        )

        reference_paths: list[Path] = []
        if reference_image_bytes is not None:
            reference_path = job_dir / "reference_0.png"
            reference_path.write_bytes(reference_image_bytes)
            reference_paths.append(reference_path)
            LOGGER.info(
                "saved job reference image job_id=%s path=%s size_bytes=%s",
                job_id,
                reference_path,
                len(reference_image_bytes),
            )

        start_index = len(reference_paths)
        for index, reference_bytes in enumerate(
            reference_images_bytes or [],
            start=start_index,
        ):
            reference_path = job_dir / f"reference_{index}.png"
            reference_path.write_bytes(reference_bytes)
            reference_paths.append(reference_path)
            LOGGER.info(
                "saved job reference image job_id=%s path=%s size_bytes=%s",
                job_id,
                reference_path,
                len(reference_bytes),
            )

        return JobFiles(
            input_path=input_path,
            reference_paths=reference_paths,
            result_path=job_dir / "result.png",
        )

    def read_input(self, record: dict[str, object]) -> bytes:
        input_path = Path(str(record["input_path"]))
        LOGGER.info(
            "reading job input image job_id=%s path=%s",
            record.get("job_id"),
            input_path,
        )
        return input_path.read_bytes()

    def read_references(self, record: dict[str, object]) -> list[bytes]:
        LOGGER.info(
            "reading job reference images job_id=%s references_count=%s",
            record.get("job_id"),
            len(record.get("reference_paths", [])),  # type: ignore[arg-type]
        )
        return [
            Path(str(path)).read_bytes()
            for path in record.get("reference_paths", [])  # type: ignore[arg-type]
        ]

    def write_result(self, record: dict[str, object], image_bytes: bytes) -> Path:
        input_path = Path(str(record["input_path"]))
        result_path = input_path.parent / "result.png"
        result_path.write_bytes(image_bytes)
        LOGGER.info(
            "saved job result image job_id=%s path=%s size_bytes=%s",
            record.get("job_id"),
            result_path,
            len(image_bytes),
        )
        return result_path

    def get_result_path(self, record: dict[str, object]) -> Path | None:
        result_path = record.get("result_path")
        if result_path is None:
            return None
        return Path(str(result_path))
