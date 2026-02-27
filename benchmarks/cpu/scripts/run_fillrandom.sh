#!/usr/bin/env bash
set -euo pipefail

# FillRandom workload.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/benchmark_common.sh"

# FillRandom-specific knobs.
NUM_KEYS="${NUM_KEYS:-20000000}"
WRITES="${WRITES:--1}"
bg_comp_threads_list="${BG_COMP_THREADS_LIST:-1}"

for value_size in ${VALUE_SIZES}; do
  for bg_comp_threads in ${bg_comp_threads_list}; do
    for subcomp_threads in ${SUBCOMP_THREADS_LIST}; do
      RUN_BASE_DIR="${OUTPUT_DIR}/fillrandom/value_${value_size}/subcomp_${subcomp_threads}"
      if [[ "${bg_comp_threads}" =~ ^[0-9]+$ ]] && (( bg_comp_threads > 1 )); then
        RUN_BASE_DIR="${RUN_BASE_DIR}/bgcomp_${bg_comp_threads}"
      fi
      RUN_DIR="${RUN_BASE_DIR}/${RUN_ID}"
      DB_DIR="${DB_BASE_DIR}/fillrandom/value_${value_size}"
      WAL_DIR="${WAL_BASE_DIR}/fillrandom/value_${value_size}"

      mkdir -p "${RUN_DIR}" "${DB_DIR}" "${WAL_DIR}"
      write_run_config "${RUN_DIR}" "run_fillrandom.sh"

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
        --subcompactions="${subcomp_threads}" \
        --max_background_compactions="${bg_comp_threads}" \
        "${COMMON_FLAGS[@]}"

      copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_fillrandom"
      copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_fillrandom"
      cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
    done
  done
done
