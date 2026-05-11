#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/benchmark_common.sh"

MODEL="ddcolor"
RUN_ID="${RUN_ID:-ddcolor_${STAMP}}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
METRICS="${METRICS:-${FULL_METRICS}}"
REFERENCE_MODE="${REFERENCE_MODE:-none}"
UV_GROUPS="${UV_GROUPS:-benchmark model-ddcolor}"

run_single_model_benchmark \
  "${MODEL}" \
  "${RUN_ID}" \
  "${SAMPLE_LIMIT}" \
  "${BATCH_SIZE}" \
  "${METRICS}" \
  "${REFERENCE_MODE}" \
  "$@"
