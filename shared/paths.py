from __future__ import annotations

import sys
from collections.abc import Collection
from pathlib import Path


def get_project_root(start: Path, levels_up: int) -> Path:
    path = start.resolve()
    if path.is_file():
        path = path.parent

    for _ in range(levels_up):
        path = path.parent

    return path


def resolve_from_root(project_root: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None

    path = Path(raw_path)
    if path.is_absolute():
        return path

    return (project_root / path).resolve()


def ensure_on_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def find_file_by_stem(
    directory: Path,
    stem: str,
    extensions: Collection[str],
) -> Path | None:
    exact = directory / f"{stem}.png"
    if exact.exists():
        return exact

    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
        and path.stem == stem
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple files matched stem={stem} in {directory}.")
    return matches[0]
