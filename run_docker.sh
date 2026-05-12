#!/usr/bin/env bash
set -euo pipefail

no_bot=0
no_build=0
logs=0
model_id="ddcolor"
device="cpu"
enabled_models="cgan_reference,colorcomic_auto,ddcolor,deoldify"
extra_uv_groups=""

usage() {
  cat <<'USAGE'
Usage: ./run_docker.sh [options]

Options:
  --no-bot                    Start only redis, api, and worker.
  --no-build                  Do not rebuild images before starting.
  --logs                      Follow logs after services start.
  --model-id VALUE            Model to run. Default: ddcolor. Supports passthrough, cgan,
                              cgan_reference, colorcomic_auto, ddcolor,
                              deoldify, colorcomic_reference, cobra.
  --device VALUE              cpu, cuda, or auto. Default: cpu.
  --enabled-models VALUE      Comma-separated enabled models.
                              Default: cgan_reference,colorcomic_auto,ddcolor,deoldify.
  --extra-uv-groups VALUE     Additional uv groups for Docker build.
  -h, --help                  Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bot)
      no_bot=1
      shift
      ;;
    --no-build)
      no_build=1
      shift
      ;;
    --logs)
      logs=1
      shift
      ;;
    --model-id)
      model_id="${2:?--model-id requires a value}"
      shift 2
      ;;
    --device)
      device="${2:?--device requires a value}"
      shift 2
      ;;
    --enabled-models)
      enabled_models="${2:?--enabled-models requires a value}"
      shift 2
      ;;
    --extra-uv-groups)
      extra_uv_groups="${2:?--extra-uv-groups requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

normalize_model_id() {
  local value
  value="$(echo "$1" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$value" in
    cgan) echo "cgan_reference" ;;
    *) echo "$value" ;;
  esac
}

split_models() {
  local raw="$1"
  local item
  local models=()
  IFS=',' read -ra parts <<< "$raw"
  for item in "${parts[@]}"; do
    item="$(echo "$item" | xargs)"
    if [[ -n "$item" ]]; then
      models+=("$(normalize_model_id "$item")")
    fi
  done
  printf '%s\n' "${models[@]}"
}

model_group() {
  case "$1" in
    passthrough) ;;
    cgan_reference) echo "--group model-cgan" ;;
    colorcomic_auto|colorcomic_reference) echo "--group model-colorcomic" ;;
    ddcolor) echo "--group model-ddcolor" ;;
    deoldify) echo "--group model-deoldify" ;;
    cobra) echo "--group model-cobra" ;;
    *)
      echo "Unsupported model '$1'. Supported: passthrough, cgan, cgan_reference, colorcomic_auto, colorcomic_reference, ddcolor, deoldify, cobra." >&2
      exit 2
      ;;
  esac
}

dotenv_value() {
  local name="$1"
  local line key value
  if [[ ! -f .env ]]; then
    return 0
  fi
  while IFS= read -r line; do
    line="$(echo "$line" | xargs)"
    if [[ -z "$line" || "$line" == \#* || "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="$(echo "$key" | xargs)"
    if [[ "$key" == "$name" ]]; then
      value="$(echo "$value" | xargs)"
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      echo "$value"
      return 0
    fi
  done < .env
  return 0
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI was not found. Install Docker and try again." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available. Start Docker and try again." >&2
  exit 1
fi

model_id="$(normalize_model_id "$model_id")"
if [[ -z "$enabled_models" ]]; then
  enabled_models="$model_id"
fi

mapfile -t enabled_model_list < <(split_models "$enabled_models")
enabled_models="$(IFS=,; echo "${enabled_model_list[*]}")"
models_for_groups=("$model_id" "${enabled_model_list[@]}")

declare -A seen_groups=()
resolved_groups=()
for model in "${models_for_groups[@]}"; do
  if ! group="$(model_group "$model")"; then
    exit 2
  fi
  if [[ -n "${group:-}" && -z "${seen_groups[$group]:-}" ]]; then
    seen_groups["$group"]=1
    resolved_groups+=("$group")
  fi
done
if [[ -n "$extra_uv_groups" ]]; then
  resolved_groups+=("$extra_uv_groups")
fi

export COLORIZATION_MODEL_ID="$model_id"
export COLORIZATION_DEVICE="$device"
export ENABLED_MODELS="$enabled_models"
export EXTRA_UV_GROUPS="${resolved_groups[*]}"

mkdir -p data outputs/service

services=(redis api worker)
build_services=(api worker)
if [[ "$no_bot" -eq 0 ]]; then
  token="${TELEGRAM_BOT_TOKEN:-}"
  if [[ -z "$token" ]]; then
    token="$(dotenv_value TELEGRAM_BOT_TOKEN || true)"
  fi
  if [[ -z "$token" || "$token" == "your-telegram-token" ]]; then
    echo "Set TELEGRAM_BOT_TOKEN in .env or run ./run_docker.sh --no-bot." >&2
    exit 1
  fi
  services=(redis postgres api worker bot)
  build_services=(api worker bot)
fi

docker compose config >/dev/null

echo "Starting services: ${services[*]}"
echo "Model: $model_id; device: $device; enabled models: $enabled_models"
if [[ -n "$EXTRA_UV_GROUPS" ]]; then
  echo "Extra uv groups: $EXTRA_UV_GROUPS"
fi

if [[ "$no_build" -eq 0 ]]; then
  echo
  echo "Building images with plain Docker progress..."
  echo "This can take a long time for heavy model groups because torch/CUDA wheels are large."
  docker compose --progress plain build "${build_services[@]}"
fi

echo
echo "Starting containers..."
up_args=(compose up -d)
if [[ "$no_build" -eq 0 ]]; then
  up_args+=(--no-build)
fi
up_args+=("${services[@]}")

docker "${up_args[@]}"

echo
echo "API: http://localhost:8000"
echo "Health: http://localhost:8000/health"
echo "Status: docker compose ps"
echo "Stop: docker compose down"

if [[ "$logs" -eq 1 ]]; then
  docker compose logs -f "${services[@]}"
fi
