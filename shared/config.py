from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path


def resolve_project_paths(project_root: Path) -> ProjectPaths:
    return ProjectPaths(root=project_root, data_dir=project_root / "data")
