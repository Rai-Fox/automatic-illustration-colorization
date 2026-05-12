#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_NAME="${RUN_NAME:-}"
STAMP="${RUN_STAMP:-${RUN_NAME:-parameterized}}"
DEVICE="${DEVICE:-cuda}"
MAX_SAVED_IMAGES="${MAX_SAVED_IMAGES:-1000000}"
BENCHMARK_MODE="${BENCHMARK_MODE:-full}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-32}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-1000}"

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

  local sample_limit_label="${sample_limit:-all}"

  echo "Benchmark run name: ${RUN_NAME:-${run_id}}"
  echo "Model: ${model}"
  echo "Run id: ${run_id}"
  echo "Sample limit: ${sample_limit_label}"
  echo "Device: ${DEVICE}"
  echo "Benchmark mode: ${BENCHMARK_MODE}"
  echo "Batch size: ${batch_size}"
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

  local benchmark_args=(
    --models "${model}"
    --reference_mode "${reference_mode}"
    --device "${DEVICE}"
    --mode "${BENCHMARK_MODE}"
    --batch_size "${batch_size}"
    --metrics "${metrics}"
    --max_saved_images "${MAX_SAVED_IMAGES}"
    --run_id "${run_id}"
    "benchmark.metrics.lpips_batch_size=${LPIPS_BATCH_SIZE}"
    "benchmark.metrics.kid_subset_size=${KID_SUBSET_SIZE}"
  )

  if [[ -n "${sample_limit}" && "${sample_limit}" != "all" ]]; then
    benchmark_args+=(--sample_limit "${sample_limit}")
  else
    benchmark_args+=("benchmark.dataset.limit=null")
  fi

  "${UV_CMD[@]}" run "${uv_group_args[@]}" python cli.py benchmark \
    "${benchmark_args[@]}" \
    "$@"

  echo
  echo "Reports:"
  echo "  Model run: outputs/benchmark/reports/${model}/${run_id}/report.json"
  echo "  Run:    outputs/benchmark/runs/${run_id}/${model}/report.json"
  echo "  Images: outputs/benchmark/generated/${model}/${run_id}/"
}
