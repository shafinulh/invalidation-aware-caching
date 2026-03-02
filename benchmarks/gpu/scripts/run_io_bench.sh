#!/usr/bin/env bash
set -euo pipefail

# run_io_bench.sh — Run GPU vs CPU IO compaction simulation.
#
# Iterates over L0_SIZES and runs gpu_io_bench for each, collecting CSV
# results under OUTPUT_DIR.
#
# Usage (from repo root):
#   ./benchmarks/gpu/scripts/run_io_bench.sh
#
# Override any config via env:
#   L0_SIZES="8388608" NUM_REPS=5 ./benchmarks/gpu/scripts/run_io_bench.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=benchmark_common.sh
source "${SCRIPT_DIR}/benchmark_common.sh"

SCRIPT_NAME="$(basename "$0")"

# Direct IO flag for the binary (1 or 0).
DIRECT_IO_FLAG=1
if [[ "${DIRECT_IO}" != "true" ]]; then
  DIRECT_IO_FLAG=0
fi

# Run directory for this invocation.
RUN_DIR="${OUTPUT_DIR}/io_bench/${RUN_ID}"
mkdir -p "${RUN_DIR}"

write_run_config "${RUN_DIR}" "${SCRIPT_NAME}"

echo "========================================"
echo " GPU IO Bench — ${RUN_ID}"
echo "========================================"
echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "  RUN_DIR     : ${RUN_DIR}"
echo "  DATA_DIR    : ${DATA_DIR}"
echo "  L0_SIZES    : ${L0_SIZES}"
echo "  NUM_L0_READ : ${NUM_L0_READ}"
echo "  NUM_L1_WRITE: ${NUM_L1_WRITE}"
echo "  NUM_REPS    : ${NUM_REPS}"
echo "  ALIGNMENT   : ${ALIGNMENT}"
echo "  DIRECT_IO   : ${DIRECT_IO}"
echo "  GPU_DEVICE  : ${GPU_DEVICE}"
echo "========================================"
echo ""

for l0_size in ${L0_SIZES}; do
  l0_mb=$(echo "scale=0; ${l0_size} / 1048576" | bc)
  csv_file="${RUN_DIR}/io_bench_${l0_mb}mb.csv"
  log_file="${RUN_DIR}/io_bench_${l0_mb}mb.log"

  echo "── L0 size: ${l0_size} bytes (${l0_mb} MB) ──"

  "${GPU_IO_BENCH}" \
    --data_dir "${DATA_DIR}" \
    --l0_size "${l0_size}" \
    --num_l0_read "${NUM_L0_READ}" \
    --num_l1_write "${NUM_L1_WRITE}" \
    --alignment "${ALIGNMENT}" \
    --direct_io "${DIRECT_IO_FLAG}" \
    --gpu_device "${GPU_DEVICE}" \
    --reps "${NUM_REPS}" \
    --csv "${csv_file}" \
    --drop_caches 1 \
    2>&1 | tee "${log_file}"

  echo ""
  echo "Results: ${csv_file}"
  echo ""
done

echo "All results in: ${RUN_DIR}"
echo "Done."
