from pathlib import Path
from typing import Any

from shared.hydra import append_override, extend_overrides, load_config
from shared.paths import get_project_root

_AGGREGATE_PANEL_SUBCOMMANDS = {"aggregate_panel", "aggregate_panels"}


def _render_models_override(models: Any) -> str:
    if isinstance(models, str):
        return models
    if isinstance(models, (list, tuple)):
        return ",".join(str(model) for model in models)
    return str(models)


def _split_model_ids(models: Any) -> list[str]:
    if models is None:
        return []
    if isinstance(models, str):
        values = models.split(",")
    else:
        try:
            values = list(models)
        except TypeError:
            values = [models]
    return [str(model_id).strip() for model_id in values if str(model_id).strip()]


class ColorizerCLI:
    """CLI entrypoint for benchmarking and image colorization."""

    def benchmark(
        self,
        *config_overrides: str,
        models: str | None = None,
        dataset_source: str | None = None,
        hf_dataset_dir: str | None = None,
        input_dir: str | None = None,
        target_dir: str | None = None,
        reference_dir: str | None = None,
        sample_limit: int | None = None,
        output_dir: str | None = None,
        json_name: str | None = None,
        csv_name: str | None = None,
        run_id: str | None = None,
        generated_dir_name: str | None = None,
        output_dir_name: str | None = None,
        save_images: bool | None = None,
        max_saved_images: int | None = None,
        max_images: int | None = None,
        random_seed: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        metrics: str | None = None,
        mode: str | None = None,
        reference_mode: str | None = None,
        reference_group_key: str | None = None,
    ) -> dict[str, object]:
        from illustration_colorizer.benchmark.runner import run_benchmark

        if (
            config_overrides
            and config_overrides[0] in _AGGREGATE_PANEL_SUBCOMMANDS
        ):
            return self.aggregate_panels(
                *config_overrides[1:],
                models=models,
                benchmark_output_dir=output_dir,
                generated_dir_name=generated_dir_name,
                output_dir_name=output_dir_name,
                max_images=max_images if max_images is not None else max_saved_images,
                random_seed=random_seed,
                sample_limit=sample_limit,
                reference_mode=reference_mode,
                reference_group_key=reference_group_key,
            )

        project_root = get_project_root(Path(__file__), levels_up=0)
        overrides: list[str] = []

        if models:
            overrides.append(
                f"benchmark.selected_models=[{_render_models_override(models)}]"
            )

        if metrics:
            overrides.append(
                f"benchmark.metrics.enabled=[{_render_models_override(metrics)}]"
            )

        append_override(overrides, "benchmark.dataset.source", dataset_source)
        append_override(overrides, "benchmark.dataset.hf_dataset_dir", hf_dataset_dir)
        append_override(overrides, "benchmark.dataset.input_dir", input_dir)
        append_override(overrides, "benchmark.dataset.target_dir", target_dir)
        append_override(overrides, "benchmark.dataset.reference_dir", reference_dir)
        append_override(overrides, "benchmark.dataset.limit", sample_limit)
        append_override(overrides, "benchmark.report.output_dir", output_dir)
        append_override(overrides, "benchmark.report.json_name", json_name)
        append_override(overrides, "benchmark.report.csv_name", csv_name)
        append_override(overrides, "benchmark.report.run_id", run_id)
        append_override(overrides, "benchmark.report.save_images", save_images)
        append_override(
            overrides, "benchmark.report.generated_dir_name", generated_dir_name
        )
        append_override(
            overrides, "benchmark.report.max_saved_images", max_saved_images
        )
        append_override(overrides, "benchmark.runtime.batch_size", batch_size)
        append_override(overrides, "benchmark.runtime.device", device)
        append_override(overrides, "benchmark.mode", mode)
        append_override(overrides, "benchmark.reference.mode", reference_mode)
        append_override(
            overrides, "benchmark.reference.group_key", reference_group_key
        )
        extend_overrides(overrides, config_overrides)

        config = load_config(
            project_root / "illustration_colorizer" / "conf", overrides
        )
        return run_benchmark(project_root=project_root, config=config)

    def aggregate_panels(
        self,
        *config_overrides: str,
        models: str | None = None,
        benchmark_output_dir: str | None = None,
        generated_dir_name: str | None = None,
        output_dir_name: str | None = None,
        max_images: int | None = None,
        random_seed: int | None = None,
        sample_limit: int | None = None,
        reference_mode: str | None = None,
        reference_group_key: str | None = None,
    ) -> dict[str, object]:
        from illustration_colorizer.benchmark.aggregate import (
            aggregate_generated_panels,
        )

        project_root = get_project_root(Path(__file__), levels_up=0)
        overrides: list[str] = []
        append_override(overrides, "benchmark.dataset.limit", sample_limit)
        append_override(overrides, "benchmark.reference.mode", reference_mode)
        append_override(
            overrides, "benchmark.reference.group_key", reference_group_key
        )
        extend_overrides(overrides, config_overrides)
        config = load_config(
            project_root / "illustration_colorizer" / "conf", overrides
        )

        benchmark_cfg = config.benchmark.report
        selected_models = _split_model_ids(models)
        if not selected_models:
            selected_models = _split_model_ids(config.benchmark.selected_models)
        return aggregate_generated_panels(
            project_root=project_root,
            models=selected_models,
            benchmark_output_dir=(
                benchmark_output_dir or str(benchmark_cfg.output_dir)
            ),
            generated_dir_name=(
                generated_dir_name or str(benchmark_cfg.generated_dir_name)
            ),
            output_dir_name=(output_dir_name or "comparisons"),
            max_images=max_images,
            random_seed=random_seed,
            dataset_config=config.benchmark.dataset,
            reference_config=config.benchmark.reference,
        )

    def prepare_models(
        self,
        *config_overrides: str,
        models: str | None = None,
        allow_download: bool | None = None,
    ) -> dict[str, object]:
        from illustration_colorizer.models import prepare_model_artifacts

        project_root = get_project_root(Path(__file__), levels_up=0)
        overrides: list[str] = []

        if models:
            overrides.append(
                f"benchmark.selected_models=[{_render_models_override(models)}]"
            )

        if allow_download is not None:
            rendered = "true" if allow_download else "false"
            overrides.append(f"models.ddcolor.allow_download={rendered}")
            overrides.append(f"models.deoldify.allow_download={rendered}")
            overrides.append(f"models.colorcomic_auto.allow_download={rendered}")
            overrides.append(f"models.colorcomic_reference.allow_download={rendered}")
            overrides.append(f"models.cobra.allow_download={rendered}")

        extend_overrides(overrides, config_overrides)
        config = load_config(
            project_root / "illustration_colorizer" / "conf", overrides
        )
        return prepare_model_artifacts(project_root=project_root, config=config)


def main() -> None:
    import fire

    fire.Fire(ColorizerCLI)


if __name__ == "__main__":
    main()
