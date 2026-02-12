#!/usr/bin/env bash
set -euo pipefail

# FillRandom workload (200M keys; key size 16B; value sizes 32..256).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREADS=1
source "${SCRIPT_DIR}/bench_env.sh"

# VALUE_SIZES="${VALUE_SIZES:-"32 64 128 256"}"
VALUE_SIZES="${VALUE_SIZES:-"32"}"
NUM_KEYS="${NUM_KEYS:-20000000}"
COMP_THREADS_LIST="${COMP_THREADS_LIST:-"1 2 4 8"}"

for value_size in ${VALUE_SIZES}; do
  for comp_threads in ${COMP_THREADS_LIST}; do
    RUN_DIR="${OUTPUT_DIR}/fillrandom/value_${value_size}/comp_${comp_threads}/${RUN_ID}"
    DB_DIR="${DB_BASE_DIR}/fillrandom/value_${value_size}"
    WAL_DIR="${WAL_BASE_DIR}/fillrandom/value_${value_size}"

    mkdir -p "${RUN_DIR}" "${DB_DIR}" "${WAL_DIR}"

    run_db_bench "${RUN_DIR}/db_bench.log" \
      --benchmarks=fillrandom \
      --num="${NUM_KEYS}" \
      --value_size="${value_size}" \
      --db="${DB_DIR}" \
      --wal_dir="${WAL_DIR}" \
      --report_file="${RUN_DIR}/report.csv" \
      --use_existing_db=0 \
      --subcompactions="${comp_threads}" \
      --max_background_compactions="${comp_threads}" \
      --max_background_flushes=2 \
      --use_direct_io_for_flush_and_compaction=true \
      --use_direct_reads=true \
      "${COMMON_FLAGS[@]}"
  done

done
