from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cli as cli_module
import illustration_colorizer.benchmark.aggregate as aggregate_module


def _benchmark_config() -> SimpleNamespace:
    return SimpleNamespace(
        benchmark=SimpleNamespace(
            report=SimpleNamespace(
                output_dir="outputs/benchmark",
                generated_dir_name="generated",
            ),
            dataset={"limit": 8},
            reference={"mode": "none"},
            selected_models=["ddcolor", "deoldify"],
        )
    )


def test_benchmark_aggregate_panel_subcommand_uses_config_models(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_aggregate_generated_panels(**kwargs: Any) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        cli_module,
        "get_project_root",
        lambda path, levels_up=0: tmp_path,
    )
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda config_dir, overrides: _benchmark_config(),
    )
    monkeypatch.setattr(
        aggregate_module,
        "aggregate_generated_panels",
        fake_aggregate_generated_panels,
    )

    result = cli_module.ColorizerCLI().benchmark("aggregate_panel", random_seed=123)

    assert result == {"ok": True}
    assert captured["project_root"] == tmp_path
    assert captured["models"] == ["ddcolor", "deoldify"]
    assert captured["benchmark_output_dir"] == "outputs/benchmark"
    assert captured["random_seed"] == 123
