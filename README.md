# Automatic Illustration Colorization

Benchmarking and inference wrappers for automatic and reference-based
illustration colorization models. Local third-party model repositories live in
`data/models`, while project wrappers live in `illustration_colorizer/models`
and expose a common `ColorizationModel` interface.

## Supported Models

- `ddcolor` - automatic colorization.
- `deoldify` - automatic colorization.
- `colorcomic_auto` - automatic ColorComic backend.
- `colorcomic_reference` - reference-based ColorComic/MangaNinja backend.
- `cgan_reference` - example-based manga cGAN backend.
- `cobra` - experimental CUDA-only reference model.

Reference-only models require a reference image. In benchmark runs, references
can be produced from comic titles using the modes described below.

## Project Structure

- `illustration_colorizer/models/` - model wrappers and artifact preparation.
- `illustration_colorizer/benchmark/` - dataset loading, metrics, runner, panel aggregation.
- `illustration_colorizer/conf/` - Hydra configuration.
- `scripts/` - benchmark launch scripts.
- `data/models/` - local vendored model repositories and artifacts.
- `outputs/benchmark/` - reports, generated panels, and aggregate outputs.
- `services/api/` - FastAPI inference service.
- `services/bot/` - Telegram bot integration.

## Installation

Install the base benchmark dependencies:

```bash
uv sync --group benchmark
```

Install dependencies for a specific model:

```bash
uv sync --group benchmark --group model-ddcolor
uv sync --group benchmark --group model-deoldify
uv sync --group benchmark --group model-colorcomic
uv sync --group benchmark --group model-cgan
uv sync --group benchmark --group model-cobra
```

For development:

```bash
uv sync --group benchmark --group dev
```

The benchmark scripts call `uv run --group benchmark --group model-...`
automatically. If Git Bash cannot find `uv` on Windows, pass it explicitly:

```bash
UV_BIN="$HOME/.local/bin/uv.exe" bash scripts/run_ddcolor_benchmark.sh
```

## Prepare Model Artifacts

Prepare artifacts before benchmarking a model:

```bash
uv run --group benchmark --group model-ddcolor python cli.py prepare_models --models ddcolor --allow_download=false
uv run --group benchmark --group model-deoldify python cli.py prepare_models --models deoldify --allow_download=false
uv run --group benchmark --group model-colorcomic python cli.py prepare_models --models colorcomic_auto,colorcomic_reference --allow_download=false
uv run --group benchmark --group model-cgan python cli.py prepare_models --models cgan_reference --allow_download=false
uv run --group benchmark --group model-cobra python cli.py prepare_models --models cobra --allow_download=false
```

Set `--allow_download=true` only when missing artifacts should be downloaded.

## Benchmark Scripts

Run all configured models one by one:

```bash
bash scripts/run_all_models_benchmark.sh
```

Run one model:

```bash
bash scripts/run_ddcolor_benchmark.sh
bash scripts/run_deoldify_benchmark.sh
bash scripts/run_colorcomic_auto_benchmark.sh
bash scripts/run_cgan_reference_benchmark.sh
bash scripts/run_colorcomic_reference_benchmark.sh
bash scripts/run_cobra_benchmark.sh
```

Common environment overrides:

```bash
SAMPLE_LIMIT=16 DEVICE=cuda bash scripts/run_ddcolor_benchmark.sh
MAX_SAVED_IMAGES=100 bash scripts/run_all_models_benchmark.sh
METRICS=colorfulness,line_preservation_score bash scripts/run_deoldify_benchmark.sh
```

Reference modes:

```bash
REFERENCE_MODE=fixed_by_title bash scripts/run_cgan_reference_benchmark.sh
REFERENCE_MODE=previous_output_by_title bash scripts/run_cgan_reference_benchmark.sh
REFERENCE_MODE=fixed_by_title bash scripts/run_colorcomic_reference_benchmark.sh
REFERENCE_MODE=previous_output_by_title bash scripts/run_colorcomic_reference_benchmark.sh
```

Cobra low-memory run:

```bash
COBRA_SAMPLE_LIMIT=1 COBRA_MAX_SIDE=384 COBRA_STEPS=2 COBRA_TOP_K=2 bash scripts/run_cobra_benchmark.sh
```

## Direct CLI Usage

Automatic models:

```bash
uv run --group benchmark --group model-ddcolor python cli.py benchmark --models ddcolor --sample_limit=8 --device cuda
uv run --group benchmark --group model-deoldify python cli.py benchmark --models deoldify --sample_limit=8 --device cuda
```

Reference models:

```bash
uv run --group benchmark --group model-cgan python cli.py benchmark --models cgan_reference --reference_mode fixed_by_title --sample_limit=8 --device cuda
uv run --group benchmark --group model-cgan python cli.py benchmark --models cgan_reference --reference_mode previous_output_by_title --sample_limit=8 --device cuda
```

Cobra:

```bash
uv run --group benchmark --group model-cobra python cli.py benchmark --models cobra --reference_mode fixed_by_title --sample_limit=1 --device cuda --batch_size 1 models.cobra.max_side=512 models.cobra.num_inference_steps=4 models.cobra.top_k=8
```

## Reference Benchmark Modes

- `none` - no reference image is attached.
- `fixed_by_title` - for every comic title, the first `color_image` is used as a fixed reference and excluded from evaluation.
- `previous_output_by_title` - for every title, the first `color_image` seeds the sequence; each next sample uses the previous successful model output as reference.

For title-aware modes, HF Arrow samples are balanced by `title` when
`benchmark.dataset.limit` is small.

## Reports and Generated Images

Benchmark outputs are written to `outputs/benchmark`.

Per-model reports are run-scoped, so different parameters do not overwrite each
other:

```text
outputs/benchmark/reports/<model>/<run_id>/report.json
outputs/benchmark/reports/<model>/<run_id>/summary.csv
```

Run-indexed copies are also written:

```text
outputs/benchmark/runs/<run_id>/<model>/report.json
outputs/benchmark/runs/<run_id>/<model>/summary.csv
```

Generated comparison panels are saved as:

```text
outputs/benchmark/generated/<model>/<sample_id>.png
outputs/benchmark/generated/<model>/manifest.json
```

Each generated panel contains:

```text
bw | ground truth | model result
```

The top-level `outputs/benchmark/report.json` and `summary.csv` represent the
latest aggregate run and may be overwritten.

## Aggregate Panels

After running several models, build shared comparison panels:

```bash
uv run --group benchmark python cli.py aggregate_panels --models ddcolor,deoldify,cgan_reference --max_images=8
```

## Configuration

Default benchmark config:

```text
illustration_colorizer/conf/benchmark/default.yaml
```

Model configs:

```text
illustration_colorizer/conf/model/*.yaml
```

Hydra overrides can be passed after CLI arguments:

```bash
uv run --group benchmark --group model-ddcolor python cli.py benchmark --models ddcolor --sample_limit=16 models.ddcolor.input_size=256
```

## API and Bot

Start the API:

```bash
docker compose up --build api redis
```

Start the Telegram bot:

```bash
copy .env.example .env
docker compose up --build bot
```
