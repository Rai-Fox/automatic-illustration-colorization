from __future__ import annotations

from pathlib import Path

from shared.paths import resolve_from_root


def _has_materialized_contents(path: Path) -> bool:
    return any(child.name != ".gitkeep" for child in path.iterdir())


def resolve_existing_dir(
    project_root: Path, raw_path: str | Path | None
) -> Path | None:
    path = resolve_from_root(project_root, raw_path)
    if path is None:
        return None
    if path.exists() and path.is_dir():
        return path
    return None


def ensure_hf_snapshot_dir(
    *,
    project_root: Path,
    raw_path: str | Path,
    repo_id: str,
    allow_download: bool,
) -> Path:
    target_dir = resolve_from_root(project_root, raw_path)
    if target_dir is None:
        raise ValueError("Target directory path must not be null.")

    if target_dir.exists() and _has_materialized_contents(target_dir):
        return target_dir

    if not allow_download:
        raise FileNotFoundError(
            f"Required local snapshot is missing: {target_dir}. "
            f"Enable allow_download or prepare artifacts manually for {repo_id}."
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
    )
    return target_dir
