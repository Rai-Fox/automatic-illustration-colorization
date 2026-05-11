#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
MAX_SAVED_IMAGES="${MAX_SAVED_IMAGES:-100}"

FULL_METRICS="${FULL_METRICS:-colorfulness,line_preservation_score,ink_preservation_score,lpips,kid}"
LIGHT_METRICS="${LIGHT_METRICS:-colorfulness,line_preservation_score,ink_preservation_score}"
EXTRA_UV_GROUPS="${EXTRA_UV_GROUPS:-}"

resolve_uv_command() {
  if [[ -n "${UV_BIN:-}" ]]; then
    UV_CMD=("${UV_BIN}")
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    UV_CMD=("uv")
    return
  fi

  if command -v uv.exe >/dev/null 2>&1; then
    UV_CMD=("uv.exe")
    return
  fi

  if command -v powershell.exe >/dev/null 2>&1; then
    local uv_path
    uv_path="$(
      powershell.exe -NoProfile -Command \
        "(Get-Command uv -ErrorAction SilentlyContinue).Source" 2>/dev/null \
        | tr -d '\r' \
        | head -n 1
    )"
    if [[ -n "${uv_path}" ]]; then
      if command -v cygpath >/dev/null 2>&1; then
        uv_path="$(cygpath -u "${uv_path}")"
      fi
      UV_CMD=("${uv_path}")
      return
    fi
  fi

  echo "uv command not found. Set UV_BIN=/path/to/uv or add uv to PATH." >&2
  return 127
}

run_single_model_benchmark() {
  local model="$1"
  local run_id="$2"
  local sample_limit="$3"
  local batch_size="$4"
  local metrics="$5"
  local reference_mode="$6"
  shift 6
  local effective_uv_groups="${UV_GROUPS:-benchmark}"

  echo "Benchmark run stamp: ${STAMP}"
  echo "Model: ${model}"
  echo "Run id: ${run_id}"
  echo "Sample limit: ${sample_limit}"
  echo "Device: ${DEVICE}"
  echo "Reference mode: ${reference_mode}"
  echo "UV groups: ${effective_uv_groups}${EXTRA_UV_GROUPS:+ ${EXTRA_UV_GROUPS}}"
  echo

  resolve_uv_command

  local uv_group_args=()
  local group
  for group in ${effective_uv_groups} ${EXTRA_UV_GROUPS}; do
    if [[ -n "${group}" ]]; then
      uv_group_args+=(--group "${group}")
    fi
  done

  "${UV_CMD[@]}" run "${uv_group_args[@]}" python cli.py benchmark \
    --models "${model}" \
    --reference_mode "${reference_mode}" \
    --sample_limit "${sample_limit}" \
    --device "${DEVICE}" \
    --batch_size "${batch_size}" \
    --metrics "${metrics}" \
    --max_saved_images "${MAX_SAVED_IMAGES}" \
    --run_id "${run_id}" \
    "$@"

  echo
  echo "Reports:"
  echo "  Model run: outputs/benchmark/reports/${model}/${run_id}/report.json"
  echo "  Run:    outputs/benchmark/runs/${run_id}/${model}/report.json"
  echo "  Images: outputs/benchmark/generated/${model}/"
}
