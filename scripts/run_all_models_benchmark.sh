#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
MAX_SAVED_IMAGES="${MAX_SAVED_IMAGES:-1000000}"
DEVICE="${DEVICE:-cuda}"
BENCHMARK_MODE="${BENCHMARK_MODE:-full}"
STAMP="${RUN_STAMP:-${RUN_NAME:-parameterized}}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-32}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-1000}"
METRICS_BATCH_SIZE="${METRICS_BATCH_SIZE:-32}"
KID_BATCH_SIZE="${KID_BATCH_SIZE:-${METRICS_BATCH_SIZE}}"

FULL_METRICS="${FULL_METRICS:-colorfulness,line_preservation_score,ink_preservation_score,lpips,kid}"
LIGHT_METRICS="${LIGHT_METRICS:-colorfulness,line_preservation_score,ink_preservation_score}"

run_model_script() {
  local script="$1"
  shift

  echo
  echo "==> ${script}"
  RUN_NAME="${RUN_NAME}" \
  RUN_STAMP="${STAMP}" \
  SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
  MAX_SAVED_IMAGES="${MAX_SAVED_IMAGES}" \
  DEVICE="${DEVICE}" \
  BENCHMARK_MODE="${BENCHMARK_MODE}" \
  LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE}" \
  KID_SUBSET_SIZE="${KID_SUBSET_SIZE}" \
  METRICS_BATCH_SIZE="${METRICS_BATCH_SIZE}" \
  KID_BATCH_SIZE="${KID_BATCH_SIZE}" \
  FULL_METRICS="${FULL_METRICS}" \
  LIGHT_METRICS="${LIGHT_METRICS}" \
  REFERENCE_MODE="${REFERENCE_MODE:-}" \
  COBRA_SAMPLE_LIMIT="${COBRA_SAMPLE_LIMIT:-}" \
  COBRA_STEPS="${COBRA_STEPS:-}" \
  COBRA_TOP_K="${COBRA_TOP_K:-}" \
  COBRA_MAX_SIDE="${COBRA_MAX_SIDE:-}" \
  COBRA_HIGH_RES_SCALE="${COBRA_HIGH_RES_SCALE:-}" \
  bash "scripts/${script}" "$@"
}

run_reference_script() {
  local script="$1"
  local mode="$2"
  shift 2

  REFERENCE_MODE="${mode}" run_model_script "${script}" "$@"
}

run_cobra_script() {
  local mode="$1"
  shift

  REFERENCE_MODE="${mode}" \
  COBRA_SAMPLE_LIMIT="${COBRA_SAMPLE_LIMIT:-${SAMPLE_LIMIT}}" \
  COBRA_STEPS="${COBRA_STEPS:-4}" \
  COBRA_TOP_K="${COBRA_TOP_K:-8}" \
  COBRA_MAX_SIDE="${COBRA_MAX_SIDE:-512}" \
  COBRA_HIGH_RES_SCALE="${COBRA_HIGH_RES_SCALE:-1.0}" \
  COBRA_BATCH_SIZE="${COBRA_BATCH_SIZE:-1}" \
  run_model_script "run_cobra_benchmark.sh" "$@"
}

echo "Benchmark run name: ${RUN_NAME:-model-parameter defaults}"
echo "Sample limit: ${SAMPLE_LIMIT:-all}"
echo "Device: ${DEVICE}"
echo "Benchmark mode: ${BENCHMARK_MODE}"
echo "LPIPS batch size: ${LPIPS_BATCH_SIZE}"
echo "KID subset size: ${KID_SUBSET_SIZE}"
echo "Metrics batch size: ${METRICS_BATCH_SIZE}"
echo "KID batch size: ${KID_BATCH_SIZE}"

# Automatic colorization models. Full metrics are reasonable here.
run_model_script "run_ddcolor_benchmark.sh"
run_model_script "run_deoldify_benchmark.sh"
run_model_script "run_colorcomic_auto_benchmark.sh"

# Reference models. Use title-aware fixed references from HF Arrow color_image.
# Heavy perceptual metrics are skipped by default to avoid masking model failures
# with metric-side GPU memory pressure.
run_reference_script "run_cgan_reference_benchmark.sh" "fixed_by_title"
run_reference_script "run_colorcomic_reference_benchmark.sh" "fixed_by_title"

# Sequential reference propagation benchmark. Batch execution is intentionally
# bypassed by the runner for this mode because every sample depends on the
# previous successful output of the same title.
run_reference_script "run_cgan_reference_benchmark.sh" "previous_output_by_title"
run_reference_script "run_colorcomic_reference_benchmark.sh" "previous_output_by_title"

# Cobra is CUDA-only and memory-sensitive. It is excluded from the default
# all-model run; launch scripts/run_cobra_benchmark.sh manually when needed.
if [[ "${RUN_COBRA:-false}" == "true" ]]; then
  run_cobra_script "fixed_by_title"
  run_cobra_script "previous_output_by_title"
fi

echo
echo "Reports:"
echo "  Per-model runs:   outputs/benchmark/reports/<model>/<run_id>/report.json"
echo "  Run snapshots:    outputs/benchmark/runs/<run_id>/<model>/report.json"
echo "  Images:           outputs/benchmark/generated/<model>/<run_id>/"
