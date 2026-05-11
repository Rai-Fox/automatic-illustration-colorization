#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/benchmark_common.sh"

MODEL="cgan_reference"
RUN_ID="${RUN_ID:-cgan_reference_${REFERENCE_MODE:-fixed_by_title}_${STAMP}}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
METRICS="${METRICS:-${LIGHT_METRICS}}"
REFERENCE_MODE="${REFERENCE_MODE:-fixed_by_title}"
UV_GROUPS="${UV_GROUPS:-benchmark model-cgan}"

run_single_model_benchmark \
  "${MODEL}" \
  "${RUN_ID}" \
  "${SAMPLE_LIMIT}" \
  "${BATCH_SIZE}" \
  "${METRICS}" \
  "${REFERENCE_MODE}" \
  "$@"
