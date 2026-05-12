# Automatic Illustration Colorization

Сервис и набор бенчмарков для автоматической и reference-based колоризации
иллюстраций. Модели обернуты единым интерфейсом в
`illustration_colorizer/models`, сервисный слой состоит из FastAPI API,
Redis-очереди, worker-процесса и Telegram-бота.

Поддерживаемые модели:

- `ddcolor` - автоматическая колоризация.
- `deoldify` - автоматическая колоризация.
- `colorcomic_auto` - автоматическая колоризация (показывает наилучшие резульаты).
- `cgan_reference` - reference-based manga cGAN.
- `colorcomic_reference` - reference-based MangaNinja.
- `cobra` - экспериментальная CUDA-only reference-модель.
- `passthrough` - тестовая модель, возвращает входное изображение.

Тяжелые артефакты моделей лежат вне Docker image в `data/models`, результаты
сервисной обработки - в `outputs/service`, результаты бенчмарков - в
`outputs/benchmark`.

## 0. Настройка Окружения

Требования:

- Python `3.12`.
- `uv`.
- Docker Desktop для запуска API, worker, Redis, PostgreSQL и Telegram-бота.
- CUDA/GPU опционально; CPU-запуск подходит не для всех моделей и медленнее.

Базовая установка зависимостей для локальной разработки и тестов:

```bash
uv sync --group dev
```

Установка зависимостей для бенчмарков:

```bash
uv sync --group benchmark
```

Установка зависимостей конкретных моделей:

```bash
uv sync --group benchmark --group model-ddcolor
uv sync --group benchmark --group model-deoldify
uv sync --group benchmark --group model-colorcomic
uv sync --group benchmark --group model-cgan
uv sync --group benchmark --group model-cobra
```

Подготовка `.env` для Docker/бота:

```bash
cp .env.example .env
```

В `.env` задайте минимум:

```dotenv
TELEGRAM_BOT_TOKEN=your-telegram-token
COLORIZATION_MODEL_ID=ddcolor
COLORIZATION_DEVICE=cpu
ENABLED_MODELS=cgan_reference,colorcomic_auto,ddcolor,deoldify
```

Для прямого `docker compose` тяжелые зависимости моделей управляются отдельно:

```dotenv
API_EXTRA_UV_GROUPS=
WORKER_EXTRA_UV_GROUPS=--group model-cgan --group model-colorcomic --group model-ddcolor --group model-deoldify
```

По умолчанию API image остается легким. Тяжелые PyTorch/Diffusers/FastAI
зависимости ставятся в worker image, потому что основной runtime выполняет
инференс через очередь задач.

Подготовка артефактов моделей перед запуском тяжелых моделей:

```bash
uv run --group benchmark --group model-ddcolor python cli.py prepare_models --models ddcolor --allow_download=false
uv run --group benchmark --group model-deoldify python cli.py prepare_models --models deoldify --allow_download=false
uv run --group benchmark --group model-colorcomic python cli.py prepare_models --models colorcomic_auto,colorcomic_reference --allow_download=false
uv run --group benchmark --group model-cgan python cli.py prepare_models --models cgan_reference --allow_download=false
```

Ставьте `--allow_download=true` только если недостающие веса можно скачать
автоматически.

## 1. Запуск Сервиса

Рекомендуемый запуск через bash-скрипт:

```bash
./run_docker.sh
```

Запуск без Telegram-бота:

```bash
./run_docker.sh --no-bot
```

Запуск с CUDA:

```bash
./run_docker.sh --device cuda
```

Выбор набора моделей:

```bash
./run_docker.sh --enabled-models "ddcolor,deoldify,colorcomic_auto,cgan_reference"
```

Скрипты:

- проверяют доступность Docker;
- подбирают `WORKER_EXTRA_UV_GROUPS` под выбранные модели;
- собирают и запускают `redis`, `api`, `worker`, `postgres` и `bot`.

Прямой запуск через compose:

```bash
docker compose build api worker bot
docker compose up -d redis postgres api worker bot
```

Если бот не нужен:

```bash
docker compose build api worker
docker compose up -d redis api worker
```

Проверка состояния:

```bash
docker compose ps
docker compose logs -f api worker bot
```

API после запуска доступен на:

```text
http://localhost:8000
```

Проверка health endpoint:

```bash
curl http://localhost:8000/health
```

Остановка контейнеров:

```bash
docker compose down
```

## 2. Описание Интерфейса Бота

Бот работает поверх async-архитектуры:

1. пользователь отправляет команду и изображение;
2. бот создает job через FastAPI;
3. API сохраняет входные файлы в `outputs/service/jobs/<job_id>/`;
4. worker забирает задачу из Redis, запускает модель и сохраняет `result.png`;
5. API публикует событие о завершении;
6. бот получает событие и автоматически отправляет результат пользователю.

Команды Telegram-бота:

- `/start` - описание бота и базового сценария работы.
- `/help` - список команд и примеры настроек.
- `/settings` - показать текущие настройки пользователя:
  `model_id`, `seed`, `options`, статус reference image.
- `/models` - показать доступные модели и inline-кнопки выбора.
- `/model` - alias для `/models`.
- `/set_settings <param> <value>` - изменить настройку вручную.
- `/set_reference + изображение` - сохранить reference image в PostgreSQL.
- `/colorize + изображение` - поставить изображение в очередь колоризации.

Примеры:

```text
/models
/set_settings seed 1
/set_settings size 576
/set_settings seed clear
```

Для reference-моделей сначала задайте reference image:

```text
/set_reference + изображение
/models -> выбрать cgan_reference
/colorize + изображение
```

Reference image хранится в пользовательских настройках в PostgreSQL и не
сбрасывается при смене модели. Если выбранная модель требует reference, бот
проверит, что reference image уже задан.

## 3. Описание API

Базовый URL при локальном запуске:

```text
http://localhost:8000
```

### `GET /health`

Проверка доступности API.

Ответ:

```json
{"status": "ok"}
```

### `GET /models`

Возвращает список моделей и их возможности.

Поля ответа:

- `model_id`
- `enabled`
- `requires_reference`
- `supports_multiple_references`
- `supports_cpu`

Пример:

```bash
curl http://localhost:8000/models
```

### `POST /colorize`

Синхронный MVP endpoint. Принимает multipart form-data и сразу запускает
колоризацию в процессе API.

Поля:

- `file` - обязательное изображение.
- `model_id` - опционально, по умолчанию `COLORIZATION_MODEL_ID`.
- `reference` - опциональное reference image.
- `references` - опциональный список reference images.
- `seed` - опциональное целое число.
- `options` - опциональный JSON object строкой.

Ответ содержит PNG в base64:

```json
{
  "image_base64": "...",
  "model": "ddcolor",
  "warnings": [],
  "metadata": {}
}
```

Важно: в текущей Docker-архитектуре API image по умолчанию легкий и не содержит
тяжелые model groups. Основной production-flow - `POST /jobs` и worker.
`/colorize` удобен для MVP/debug или для API image, собранного с нужными
`API_EXTRA_UV_GROUPS`.

Пример:

```bash
curl -X POST http://localhost:8000/colorize \
  -F "model_id=passthrough" \
  -F "file=@input.png"
```

### `POST /jobs`

Основной endpoint для асинхронной обработки. API валидирует входные данные,
сохраняет файлы и ставит задачу в Redis-очередь.

Поля такие же, как у `/colorize`, дополнительно:

- `chat_id` - опционально, нужен боту для автоматической доставки результата.

Пример:

```bash
curl -X POST http://localhost:8000/jobs \
  -F "model_id=ddcolor" \
  -F "seed=1" \
  -F 'options={"size":576}' \
  -F "file=@input.png"
```

Ответ:

```json
{
  "job_id": "0f72a772087f4e0f92bb3836f4a2303a",
  "status": "queued"
}
```

### `GET /jobs/{job_id}`

Возвращает состояние задачи:

- `queued`
- `running`
- `succeeded`
- `failed`

Пример:

```bash
JOB_ID=0f72a772087f4e0f92bb3836f4a2303a
curl "http://localhost:8000/jobs/$JOB_ID"
```

### `GET /jobs/{job_id}/result`

Возвращает PNG, если задача завершилась успешно. Если задача еще не завершена,
API вернет `409`.

Пример:

```bash
JOB_ID=0f72a772087f4e0f92bb3836f4a2303a
curl -o result.png "http://localhost:8000/jobs/$JOB_ID/result"
```

## 4. Описание И Команды Для Запуска Бенчмарка

Бенчмарк находится в `illustration_colorizer/benchmark` и запускается через
`cli.py` или готовые shell-скрипты из `scripts/`.

Основной конфиг:

```text
illustration_colorizer/conf/benchmark/default.yaml
```

Важные параметры:

- `benchmark.dataset.source` - источник данных: `hf_arrow` или локальные папки.
- `benchmark.dataset.limit` - лимит сэмплов.
- `benchmark.mode` - `full`, `images_only`, `metrics_only`.
- `benchmark.reference.mode` - `none`, `fixed_by_title`,
  `previous_output_by_title`.
- `benchmark.metrics.enabled` - список метрик.
- `benchmark.runtime.device` - `cuda` или `cpu`.
- `benchmark.runtime.batch_size` - batch size.
- `benchmark.report.output_dir` - директория отчетов.

Установка benchmark-зависимостей:

```bash
uv sync --group benchmark
```

Запуск всех основных моделей:

```bash
bash scripts/run_all_models_benchmark.sh
```

По умолчанию all-model script пропускает `cobra`. Для включения:

```bash
RUN_COBRA=true bash scripts/run_all_models_benchmark.sh
```

Запуск одной модели:

```bash
bash scripts/run_ddcolor_benchmark.sh
bash scripts/run_deoldify_benchmark.sh
bash scripts/run_colorcomic_auto_benchmark.sh
bash scripts/run_cgan_reference_benchmark.sh
bash scripts/run_colorcomic_reference_benchmark.sh
bash scripts/run_cobra_benchmark.sh
```

Типичные overrides:

```bash
SAMPLE_LIMIT=16 DEVICE=cuda bash scripts/run_ddcolor_benchmark.sh
SAMPLE_LIMIT=all bash scripts/run_ddcolor_benchmark.sh
RUN_NAME=experiment_01 bash scripts/run_all_models_benchmark.sh
MAX_SAVED_IMAGES=1000000 bash scripts/run_all_models_benchmark.sh
METRICS=colorfulness,line_preservation_score bash scripts/run_deoldify_benchmark.sh
```

Режимы бенчмарка:

```bash
BENCHMARK_MODE=full bash scripts/run_ddcolor_benchmark.sh
BENCHMARK_MODE=images_only bash scripts/run_ddcolor_benchmark.sh
BENCHMARK_MODE=metrics_only bash scripts/run_ddcolor_benchmark.sh
```

- `full` - инференс, сохранение изображений и расчет метрик.
- `images_only` - только инференс и сохранение generated images.
- `metrics_only` - расчет метрик по уже сохраненным изображениям.

Reference modes:

```bash
REFERENCE_MODE=fixed_by_title bash scripts/run_cgan_reference_benchmark.sh
REFERENCE_MODE=previous_output_by_title bash scripts/run_cgan_reference_benchmark.sh
REFERENCE_MODE=fixed_by_title bash scripts/run_colorcomic_reference_benchmark.sh
REFERENCE_MODE=previous_output_by_title bash scripts/run_colorcomic_reference_benchmark.sh
```

Прямой CLI-запуск:

```bash
uv run --group benchmark --group model-ddcolor python cli.py benchmark --models ddcolor --sample_limit=8 --device cuda
uv run --group benchmark --group model-deoldify python cli.py benchmark --models deoldify --sample_limit=8 --device cuda
uv run --group benchmark --group model-cgan python cli.py benchmark --models cgan_reference --reference_mode fixed_by_title --sample_limit=8 --device cuda
```

Precompute images, затем отдельно метрики:

```bash
uv run --group benchmark --group model-ddcolor python cli.py benchmark --models ddcolor --mode images_only --sample_limit=32 --device cuda --run_id ddcolor_precomputed
uv run --group benchmark --group model-ddcolor python cli.py benchmark --models ddcolor --mode metrics_only --sample_limit=32 --device cuda --run_id ddcolor_precomputed
```

Низкопамятный запуск Cobra:

```bash
COBRA_SAMPLE_LIMIT=1 COBRA_MAX_SIDE=384 COBRA_STEPS=2 COBRA_TOP_K=2 bash scripts/run_cobra_benchmark.sh
```

Сбор aggregate comparison panels после нескольких запусков:

```bash
uv run --group benchmark python cli.py aggregate_panels --models ddcolor,deoldify,cgan_reference --max_images=8
```

Сравнение разных run_id одной модели:

```bash
uv run --group benchmark python cli.py aggregate_panels --models ddcolor:ddcolor_small,ddcolor:ddcolor_large --max_images=8
```

Основные выходные файлы:

```text
outputs/benchmark/reports/<model>/<run_id>/report.json
outputs/benchmark/reports/<model>/<run_id>/summary.csv
outputs/benchmark/runs/<run_id>/<model>/report.json
outputs/benchmark/generated/<model>/<run_id>/<sample_id>.png
outputs/benchmark/generated/<model>/<run_id>/manifest.json
outputs/benchmark/comparisons/
```

Hydra overrides можно передавать после CLI-аргументов:

```bash
uv run --group benchmark --group model-ddcolor python cli.py benchmark --models ddcolor --sample_limit=16 models.ddcolor.input_size=256
```
