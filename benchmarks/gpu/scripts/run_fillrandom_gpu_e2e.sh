#!/usr/bin/env bash
set -euo pipefail

# run_fillrandom_gpu_e2e.sh -- end-to-end fillrandom benchmark using the
# integrated RocksDB GPU compaction path.
#
# Mirrors benchmarks/cpu/scripts/run_fillrandom.sh, but routes compactions
# through a RocksDB options file with enable_gpu_compaction=true and keeps
# data/results under benchmarks/gpu/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_BENCH_ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT_DIR="$(cd "${GPU_BENCH_ROOT_DIR}/../.." && pwd)"
WORKSPACE_ROOT_DIR="$(cd "${REPO_ROOT_DIR}/.." && pwd)"

GFLAGS_LOCAL_LIB="${GFLAGS_LOCAL_LIB:-}"
if [[ -z "${GFLAGS_LOCAL_LIB}" ]]; then
  if [[ -d "${WORKSPACE_ROOT_DIR}/local/gflags/usr/lib/x86_64-linux-gnu" ]]; then
    GFLAGS_LOCAL_LIB="${WORKSPACE_ROOT_DIR}/local/gflags/usr/lib/x86_64-linux-gnu"
  elif [[ -d "${REPO_ROOT_DIR}/../local/gflags/usr/lib/x86_64-linux-gnu" ]]; then
    GFLAGS_LOCAL_LIB="${REPO_ROOT_DIR}/../local/gflags/usr/lib/x86_64-linux-gnu"
  fi
fi
if [[ -n "${GFLAGS_LOCAL_LIB}" && -d "${GFLAGS_LOCAL_LIB}" ]]; then
  export LD_LIBRARY_PATH="${GFLAGS_LOCAL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Override the CPU benchmark defaults before sourcing benchmark_common.sh so we
# can reuse the exact fillrandom workload/config plumbing.
export DB_BENCH="${DB_BENCH:-${REPO_ROOT_DIR}/rocksdb-gpu/db_bench}"
export DB_BASE_DIR="${DB_BASE_DIR:-${REPO_ROOT_DIR}/benchmarks/gpu/data/fillrandom_e2e_db}"
export WAL_BASE_DIR="${WAL_BASE_DIR:-${REPO_ROOT_DIR}/benchmarks/gpu/data/fillrandom_e2e_wal}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT_DIR}/benchmarks/gpu/results}"

# shellcheck source=/nfs/ug/groups/ece1755_w26_group1/rocksdb/benchmarks/cpu/scripts/benchmark_common.sh
source "${REPO_ROOT_DIR}/benchmarks/cpu/scripts/benchmark_common.sh"

NUM_KEYS="${NUM_KEYS:-20000000}"
WRITES="${WRITES:--1}"
USE_GPU_COMPACTION="${USE_GPU_COMPACTION:-true}"
GPU_SUBCOMP_THREADS="${GPU_SUBCOMP_THREADS:-1}"
GPU_BG_COMP_THREADS="${GPU_BG_COMP_THREADS:-1}"
GPU_OPTIONS_TEMPLATE="${GPU_OPTIONS_TEMPLATE:-${REPO_ROOT_DIR}/benchmarks/cpu/results/fillrandom/value_32/subcomp_1/0309_1637/metadata/rocksdb_options_after_fillrandom.ini}"

# Persist the actual GPU benchmark settings into metadata via run_config.env.
export SUBCOMP_THREADS_LIST="${GPU_SUBCOMP_THREADS}"
export BG_COMP_THREADS_LIST="${GPU_BG_COMP_THREADS}"

for value_size in ${VALUE_SIZES}; do
  RUN_DIR="${OUTPUT_DIR}/fillrandom_gpu_e2e/value_${value_size}/${RUN_ID}"
  DB_DIR="${DB_BASE_DIR}/value_${value_size}"
  WAL_DIR="${WAL_BASE_DIR}/value_${value_size}"
  OPTIONS_FILE_ARGS=()

  mkdir -p "${RUN_DIR}" "${DB_DIR}" "${WAL_DIR}"
  write_run_config "${RUN_DIR}" "run_fillrandom_gpu_e2e.sh"

  if [[ "${USE_GPU_COMPACTION}" == "true" ]]; then
    if [[ ! -f "${GPU_OPTIONS_TEMPLATE}" ]]; then
      echo "error: GPU_OPTIONS_TEMPLATE not found: ${GPU_OPTIONS_TEMPLATE}" >&2
      exit 1
    fi

    GPU_OPTIONS_FILE="${RUN_DIR}/metadata/gpu_compaction_options.ini"
    awk -v wal_dir="${WAL_DIR}" '
      BEGIN { inserted = 0 }
      /^\[DBOptions\]/ { in_db = 1; print; next }
      /^\[/ {
        if (in_db && !inserted) {
          print "  enable_gpu_compaction=true"
          inserted = 1
        }
        in_db = 0
      }
      {
        if (in_db && $0 ~ /^[[:space:]]*wal_dir=/) {
          print "  wal_dir=" wal_dir
          next
        }
        if (in_db && $0 ~ /^[[:space:]]*enable_gpu_compaction=/) {
          print "  enable_gpu_compaction=true"
          inserted = 1
          next
        }
        print
      }
      END {
        if (in_db && !inserted) {
          print "  enable_gpu_compaction=true"
        }
      }
    ' "${GPU_OPTIONS_TEMPLATE}" > "${GPU_OPTIONS_FILE}"
    OPTIONS_FILE_ARGS+=(--options_file="${GPU_OPTIONS_FILE}")
  fi

  run_db_bench "${RUN_DIR}/db_bench.log" \
    --benchmarks=fillrandom \
    --num="${NUM_KEYS}" \
    --writes="${WRITES}" \
    --value_size="${value_size}" \
    --db="${DB_DIR}" \
    --wal_dir="${WAL_DIR}" \
    --report_file="${RUN_DIR}/report.csv" \
    --metrics_file="${RUN_DIR}/metrics.csv" \
    --use_existing_db=0 \
    --subcompactions="${GPU_SUBCOMP_THREADS}" \
    --max_background_compactions="${GPU_BG_COMP_THREADS}" \
    "${OPTIONS_FILE_ARGS[@]}" \
    "${COMMON_FLAGS[@]}"

  copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_fillrandom_gpu_e2e"
  copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_fillrandom_gpu_e2e"
  cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
done