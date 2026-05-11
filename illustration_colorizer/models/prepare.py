from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from illustration_colorizer.models.local_assets import ensure_hf_snapshot_dir
from illustration_colorizer.models.registry import resolve_model_configs
from shared.omegaconf import to_plain_mapping
from shared.paths import resolve_from_root


@dataclass(frozen=True)
class PreparedModelReport:
    model_id: str
    status: str
    repo_path: str | None = None
    artifacts_path: str | None = None
    message: str | None = None


def _prepare_ddcolor(
    project_root: Path, model_config: dict[str, Any]
) -> PreparedModelReport:
    repo_path = resolve_from_root(project_root, model_config.get("repo_path"))
    if repo_path is None or not repo_path.exists():
        raise FileNotFoundError(f"DDColor repository not found: {repo_path}")

    checkpoint_path = resolve_from_root(
        project_root, model_config.get("checkpoint_path")
    )
    if checkpoint_path is not None and checkpoint_path.exists():
        return PreparedModelReport(
            model_id="ddcolor",
            status="ready",
            repo_path=str(repo_path),
            artifacts_path=str(checkpoint_path),
        )

    pretrained_local_dir = model_config.get("pretrained_local_dir")
    if not pretrained_local_dir:
        raise ValueError(
            "DDColor requires pretrained_local_dir for local artifact preparation."
        )

    snapshot_dir = ensure_hf_snapshot_dir(
        project_root=project_root,
        raw_path=str(pretrained_local_dir),
        repo_id=str(model_config["pretrained_id"]),
        allow_download=bool(model_config.get("allow_download", True)),
    )
    return PreparedModelReport(
        model_id="ddcolor",
        status="ready",
        repo_path=str(repo_path),
        artifacts_path=str(snapshot_dir),
    )


def _prepare_deoldify(
    project_root: Path, model_config: dict[str, Any]
) -> PreparedModelReport:
    repo_path = resolve_from_root(project_root, model_config.get("repo_path"))
    if repo_path is None or not repo_path.exists():
        raise FileNotFoundError(f"DeOldify repository not found: {repo_path}")

    root_folder = ensure_hf_snapshot_dir(
        project_root=project_root,
        raw_path=str(model_config["root_folder"]),
        repo_id=str(model_config.get("hf_repo_id", "leonelhs/deoldify")),
        allow_download=bool(model_config.get("allow_download", True)),
    )
    return PreparedModelReport(
        model_id="deoldify",
        status="ready",
        repo_path=str(repo_path),
        artifacts_path=str(root_folder),
    )


def _prepare_ctrlcolor(
    project_root: Path, model_config: dict[str, Any]
) -> PreparedModelReport:
    repo_path = resolve_from_root(project_root, model_config.get("repo_path"))
    checkpoint_path = resolve_from_root(
        project_root, model_config.get("checkpoint_path")
    )
    config_path = resolve_from_root(project_root, model_config.get("config_path"))

    if repo_path is None or not repo_path.exists():
        raise FileNotFoundError(f"CtrlColor repository not found: {repo_path}")
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"CtrlColor config not found: {config_path}")
    if checkpoint_path is None or not checkpoint_path.exists():
        return PreparedModelReport(
            model_id="ctrlcolor",
            status="missing_checkpoint",
            repo_path=str(repo_path),
            artifacts_path=(
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
            message="Checkpoint is not present in data/models/ctrlcolor/checkpoints.",
        )

    return PreparedModelReport(
        model_id="ctrlcolor",
        status="ready",
        repo_path=str(repo_path),
        artifacts_path=str(checkpoint_path),
    )


def _prepare_existing_repo(
    project_root: Path,
    model_config: dict[str, Any],
    *,
    artifact_key: str | None = None,
    artifact_relative_to_repo: bool = False,
) -> PreparedModelReport:
    model_id = str(model_config["model_id"])
    repo_path = resolve_from_root(project_root, model_config.get("repo_path"))
    if repo_path is None or not repo_path.exists():
        raise FileNotFoundError(f"{model_id} repository not found: {repo_path}")

    artifact_path = None
    if artifact_key is not None:
        raw_artifact_path = model_config.get(artifact_key)
        if artifact_relative_to_repo and raw_artifact_path is not None:
            artifact_path = (repo_path / str(raw_artifact_path)).resolve()
        else:
            artifact_path = resolve_from_root(project_root, raw_artifact_path)
        if artifact_path is None or not artifact_path.exists():
            return PreparedModelReport(
                model_id=model_id,
                status="missing_artifact",
                repo_path=str(repo_path),
                artifacts_path=(
                    str(artifact_path) if artifact_path is not None else None
                ),
                message=f"Configured artifact is missing: {artifact_key}.",
            )

    return PreparedModelReport(
        model_id=model_id,
        status="ready",
        repo_path=str(repo_path),
        artifacts_path=str(artifact_path) if artifact_path is not None else None,
    )


def _prepare_colorcomic_reference(
    project_root: Path,
    model_config: dict[str, Any],
) -> PreparedModelReport:
    repo_path = resolve_from_root(project_root, model_config.get("repo_path"))
    if repo_path is None or not repo_path.exists():
        raise FileNotFoundError(
            f"colorcomic_reference repository not found: {repo_path}"
        )

    weights_dir = repo_path / str(model_config.get("weights_dir", "models/weights"))
    manganinja_dir = weights_dir / "manganinja"
    required = [
        manganinja_dir / "denoising_unet.pth",
        manganinja_dir / "reference_unet.pth",
        manganinja_dir / "point_net.pth",
        manganinja_dir / "controlnet.pth",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        return PreparedModelReport(
            model_id="colorcomic_reference",
            status="missing_artifact",
            repo_path=str(repo_path),
            artifacts_path=str(manganinja_dir),
            message="Missing MangaNinja weights: "
            + ", ".join(path.name for path in missing),
        )

    return PreparedModelReport(
        model_id="colorcomic_reference",
        status="ready",
        repo_path=str(repo_path),
        artifacts_path=str(manganinja_dir),
    )


def _prepare_cobra(
    project_root: Path,
    model_config: dict[str, Any],
) -> PreparedModelReport:
    repo_report = _prepare_existing_repo(
        project_root,
        model_config,
        artifact_key="snapshot_path",
    )
    if repo_report.status != "ready":
        return repo_report

    pixart_local_dir = model_config.get("pixart_local_dir")
    if not pixart_local_dir:
        raise ValueError("cobra requires pixart_local_dir in config.")

    pixart_dir = ensure_hf_snapshot_dir(
        project_root=project_root,
        raw_path=str(pixart_local_dir),
        repo_id=str(model_config["pixart_model"]),
        allow_download=bool(model_config.get("allow_download", False)),
    )
    return PreparedModelReport(
        model_id="cobra",
        status="ready",
        repo_path=repo_report.repo_path,
        artifacts_path=str(pixart_dir),
        message=f"Cobra snapshot ready; PixArt snapshot ready at {pixart_dir}",
    )


def prepare_model_artifacts(
    *,
    project_root: Path,
    config: DictConfig,
) -> dict[str, object]:
    config_dict = to_plain_mapping(config)
    benchmark_config = to_plain_mapping(config_dict["benchmark"])
    model_configs = resolve_model_configs(
        config_dict["models"],
        selected_models=list(benchmark_config.get("selected_models") or []),
    )

    reports: list[PreparedModelReport] = []
    for model_config in model_configs:
        model_id = str(model_config["model_id"])
        if model_id == "ddcolor":
            reports.append(_prepare_ddcolor(project_root, model_config))
            continue
        if model_id == "deoldify":
            reports.append(_prepare_deoldify(project_root, model_config))
            continue
        if model_id == "ctrlcolor":
            reports.append(_prepare_ctrlcolor(project_root, model_config))
            continue
        if model_id == "colorcomic_auto":
            reports.append(
                _prepare_existing_repo(
                    project_root,
                    model_config,
                    artifact_key="generator_path",
                    artifact_relative_to_repo=True,
                )
            )
            continue
        if model_id == "colorcomic_reference":
            reports.append(_prepare_colorcomic_reference(project_root, model_config))
            continue
        if model_id == "cgan_reference":
            reports.append(
                _prepare_existing_repo(
                    project_root, model_config, artifact_key="checkpoint_path"
                )
            )
            continue
        if model_id == "cobra":
            reports.append(_prepare_cobra(project_root, model_config))
            continue
        reports.append(
            PreparedModelReport(
                model_id=model_id,
                status="skipped",
                message="No artifact preparation is required for this model.",
            )
        )

    return {
        "models": [
            {
                "model_id": report.model_id,
                "status": report.status,
                "repo_path": report.repo_path,
                "artifacts_path": report.artifacts_path,
                "message": report.message,
            }
            for report in reports
        ]
    }
