#!/usr/bin/env bash
set -euo pipefail

# run_fillrandom_gpu.sh — GPU compaction benchmark sweeping value sizes,
# mirroring benchmarks/cpu/scripts/run_fillrandom.sh.
#
# Runs 4 combinations matching the CPU fillrandom sweep:
#   value sizes: 32 64 128 256 bytes × 1 GPU config = 4 runs
#
# The GPU is always used at full capacity (single kernel launch, all CUDA
# cores available). This is the GPU equivalent of the CPU's subcomp_1 baseline.
#
# Usage (from repository root):
#   bash benchmarks/gpu/scripts/run_fillrandom_gpu.sh
#
# Inline overrides:
#   NUM_REPS=10 VALUE_SIZES="128 256" bash benchmarks/gpu/scripts/run_fillrandom_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

load_bench_env \
  NUM_REPS \
  ALIGNMENT \
  VALUE_SIZES

# ── Knobs ────────────────────────────────────────────────────────────────────
NUM_REPS="${NUM_REPS:-5}"
ALIGNMENT="${ALIGNMENT:-4096}"

# Mirror CPU fillrandom: 32 64 128 256 byte values
VALUE_SIZES="${VALUE_SIZES:-32 64 128 256}"

RUN_ID="${RUN_ID:-$(date +%m%d_%H%M)}"

# ── Run ──────────────────────────────────────────────────────────────────────
for value_size in ${VALUE_SIZES}; do
  # Scale INPUT_SST_MB with value size to keep a sensible SST per hook call.
  if   (( value_size <= 64  )); then INPUT_SST_MB=8
  elif (( value_size <= 128 )); then INPUT_SST_MB=16
  else                               INPUT_SST_MB=32
  fi

  RUN_DIR="${OUTPUT_DIR}/fillrandom_gpu/value_${value_size}/${RUN_ID}"

  echo "========================================================"
  echo "  GPU fillrandom — value_size=${value_size}B  input_sst=${INPUT_SST_MB}MB  reps=${NUM_REPS}"
  echo "  RUN_DIR: ${RUN_DIR}"
  echo "========================================================"
  echo ""

  NUM_REPS="${NUM_REPS}" \
  ALIGNMENT="${ALIGNMENT}" \
  INPUT_SST_MB="${INPUT_SST_MB}" \
  RUN_ID="${RUN_ID}" \
  OUTPUT_DIR="${OUTPUT_DIR}/fillrandom_gpu/value_${value_size}" \
  bash "${SCRIPT_DIR}/run_rocksdb_gpu_hook_bench.sh"

  echo ""
done

echo "=== GPU fillrandom sweep complete ==="
echo "Results under: ${OUTPUT_DIR}/fillrandom_gpu/"
echo ""
echo "Analyse with:"
echo "  python3 benchmarks/gpu/python/plot_gpu_compaction_hook_bench.py <RUN_DIR> --plot"

