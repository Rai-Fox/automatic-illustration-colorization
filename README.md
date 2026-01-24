# Automatic Illustration Colorization

Skeleton project for an illustration colorization workflow, including data
preparation, training runs, FastAPI service, and a Telegram bot.

## Features
- Data preparation and model training entry points
- FastAPI service for inference
- Telegram bot integration
- Hydra-based configuration
- DVC pipeline with Google Drive remote
- MLflow tracking hooks

## Structure
- illustration_colorizer/ - data prep + training package
- services/api/ - FastAPI app for colorization requests
- services/bot/ - Telegram bot that calls the API
- data/ - raw/processed data and model artifacts
- illustration_colorizer/conf - Hydra configs for training data and model
- services/api/conf - Hydra configs for the API
- services/bot/conf - Hydra configs for the bot

## Installation (uv)
1) Create a virtual environment and install dependencies for training

```bash
uv venv
uv pip install -e . --group train --group dev
```

## Quick Start
1) Run the data pipeline

```bash
uv run --group train python cli.py data
```

2) Run a training run

```bash
uv run --group train python cli.py train
```

3) Start the API (Docker)

```bash
docker compose up --build api redis
```

4) Start the Telegram bot (Docker, requires env vars)

```bash
copy .env.example .env
docker compose up --build bot
```

## Configuration (Hydra)
Default configs live in component folders. Override values on the command line:

```bash
uv run --group train python cli.py train --epochs=20 --learning_rate=0.0003
uv run --group train python cli.py data --raw_dir=data/new_raw
```

## Workflow
- CLI: `uv run --group train python cli.py data|train`
- DVC: `dvc repro` runs pipeline + train stages
- MLflow: logs metrics and artifacts to `mlruns/`

## Docker
1) Copy env file and set values

```bash
copy .env.example .env
```

2) Build and start services (API, bot, Redis)

```bash
docker compose up --build
```
