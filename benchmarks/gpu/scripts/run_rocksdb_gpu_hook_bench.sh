#!/usr/bin/env bash
set -euo pipefail

# run_rocksdb_gpu_hook_bench.sh - Run the RocksDB dummy GPU compaction hook
# benchmark and collect per-repetition CSV output.
#
# Usage (from repo root):
#   ./benchmarks/gpu/scripts/run_rocksdb_gpu_hook_bench.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT_SCRIPT_DIR="$(cd "${BENCH_ROOT_SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT_DIR="$(cd "${REPO_ROOT_SCRIPT_DIR}/.." && pwd)"

# Locate local GDS library path (for libcufile.so) and default worker binary.
# Check repo-local path first, then fall back to workspace-root cuda_test.
if [[ -d "${REPO_ROOT_SCRIPT_DIR}/cuda_test/gds/local/lib" ]]; then
  GDS_LOCAL_LIB="${GDS_LOCAL_LIB:-${REPO_ROOT_SCRIPT_DIR}/cuda_test/gds/local/lib}"
else
  GDS_LOCAL_LIB="${GDS_LOCAL_LIB:-${WORKSPACE_ROOT_DIR}/cuda_test/gds/local/lib}"
fi
export LD_LIBRARY_PATH="${GDS_LOCAL_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

load_bench_env \
  ALIGNMENT \
  NUM_REPS \
  INPUT_SST_MB

ALIGNMENT="${ALIGNMENT:-4096}"
NUM_REPS="${NUM_REPS:-1}"
INPUT_SST_MB="${INPUT_SST_MB:-8}"

SCRIPT_NAME="$(basename "$0")"
REPO_ROOT_DIR="$(cd "${BENCH_ROOT_DIR}/../.." && pwd)"
ROCKSDB_ROOT_DIR="${REPO_ROOT_DIR}/rocksdb-gpu"
ROCKSDB_HOOK_BENCH="${ROCKSDB_ROOT_DIR}/gpu_compaction_hook_benchmark_test"
GPU_FILE_REPLAY_BENCH="${BENCH_ROOT_DIR}/gpu_file_replay_bench"
GPU_COMPACTION_WORKER="${BENCH_ROOT_DIR}/gpu_compaction_worker"

echo "Ensuring ${GPU_FILE_REPLAY_BENCH} is up to date ..."
make -C "${BENCH_ROOT_DIR}" gpu_file_replay_bench

echo "Ensuring ${GPU_COMPACTION_WORKER} is up to date ..."
make -C "${BENCH_ROOT_DIR}" gpu_compaction_worker

echo "Ensuring ${ROCKSDB_HOOK_BENCH} is up to date ..."
make -C "${ROCKSDB_ROOT_DIR}" \
  LIB_MODE="${LIB_MODE:-static}" \
  USE_RTTI="${USE_RTTI:-1}" \
  DISABLE_WARNING_AS_ERROR=1 \
  gpu_compaction_hook_benchmark_test

RUN_DIR="${OUTPUT_DIR}/rocksdb_hook/${RUN_ID}"
CSV_FILE="${RUN_DIR}/gpu_compaction_hook.csv"
LOG_FILE="${RUN_DIR}/gpu_compaction_hook.log"

mkdir -p "${RUN_DIR}"
rm -f "${CSV_FILE}" "${LOG_FILE}"
write_run_config "${RUN_DIR}" "${SCRIPT_NAME}"

echo "========================================"
echo " RocksDB GPU Hook Bench - ${RUN_ID}"
echo "========================================"
echo "  OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "  RUN_DIR         : ${RUN_DIR}"
echo "  GPU_DEVICE      : ${GPU_DEVICE}"
echo "  ALIGNMENT       : ${ALIGNMENT}"
echo "  INPUT_SST_MB    : ${INPUT_SST_MB}"
echo "  NUM_REPS        : ${NUM_REPS}"
echo "  REPLAY_HELPER   : ${GPU_FILE_REPLAY_BENCH}"
echo "  GPU_WORKER      : ${GPU_COMPACTION_WORKER}"
echo "  ROCKSDB_BENCH   : ${ROCKSDB_HOOK_BENCH}"
echo "========================================"
echo ""

GPU_COMPACTION_HOOK_CSV="${CSV_FILE}" \
GPU_FILE_REPLAY_BENCH="${GPU_FILE_REPLAY_BENCH}" \
GPU_COMPACTION_WORKER="${GPU_COMPACTION_WORKER}" \
GPU_DEVICE="${GPU_DEVICE}" \
ALIGNMENT="${ALIGNMENT}" \
INPUT_SST_MB="${INPUT_SST_MB}" \
NUM_REPS="${NUM_REPS}" \
"${ROCKSDB_HOOK_BENCH}" \
  --gtest_filter=GpuCompactionHookBenchmarkTest.ManualCompactionReplaysGroundTruthSsts \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "Results: ${CSV_FILE}"
echo "Log    : ${LOG_FILE}"
echo ""
echo "Plot with:"
echo "  python3 benchmarks/gpu/python/plot_gpu_compaction_hook_bench.py ${RUN_DIR} --plot"
echo ""
echo "Done."
