#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/benchmark_common.sh"

MODEL="ddcolor"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
METRICS="${METRICS:-${FULL_METRICS}}"
REFERENCE_MODE="${REFERENCE_MODE:-none}"
UV_GROUPS="${UV_GROUPS:-benchmark model-ddcolor}"
RUN_SUFFIX="${DEVICE}${RUN_NAME:+_${RUN_NAME}}"
RUN_ID="${RUN_ID:-${MODEL}_${RUN_SUFFIX}}"

run_single_model_benchmark \
  "${MODEL}" \
  "${RUN_ID}" \
  "${SAMPLE_LIMIT}" \
  "${BATCH_SIZE}" \
  "${METRICS}" \
  "${REFERENCE_MODE}" \
  "$@"
