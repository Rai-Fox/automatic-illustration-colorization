#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/benchmark_common.sh"

MODEL="cobra"
REFERENCE_MODE="${REFERENCE_MODE:-fixed_by_title}"
SAMPLE_LIMIT="${COBRA_SAMPLE_LIMIT:-${SAMPLE_LIMIT:-}}"
BATCH_SIZE="${COBRA_BATCH_SIZE:-${BATCH_SIZE:-1}}"
METRICS="${METRICS:-${LIGHT_METRICS}}"
COBRA_STEPS="${COBRA_STEPS:-4}"
COBRA_TOP_K="${COBRA_TOP_K:-8}"
COBRA_MAX_SIDE="${COBRA_MAX_SIDE:-512}"
COBRA_HIGH_RES_SCALE="${COBRA_HIGH_RES_SCALE:-1.0}"
UV_GROUPS="${UV_GROUPS:-benchmark model-cobra}"
RUN_SUFFIX="${REFERENCE_MODE}_${DEVICE}${RUN_NAME:+_${RUN_NAME}}"
RUN_ID="${RUN_ID:-${MODEL}_${RUN_SUFFIX}}"

run_single_model_benchmark \
  "${MODEL}" \
  "${RUN_ID}" \
  "${SAMPLE_LIMIT}" \
  "${BATCH_SIZE}" \
  "${METRICS}" \
  "${REFERENCE_MODE}" \
  "models.cobra.num_inference_steps=${COBRA_STEPS}" \
  "models.cobra.top_k=${COBRA_TOP_K}" \
  "models.cobra.max_side=${COBRA_MAX_SIDE}" \
  "models.cobra.high_res_scale=${COBRA_HIGH_RES_SCALE}" \
  "models.cobra.vae_slicing=true" \
  "models.cobra.vae_tiling=true" \
  "$@"
