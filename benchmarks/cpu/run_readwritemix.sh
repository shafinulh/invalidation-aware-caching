#!/usr/bin/env bash
set -euo pipefail

# Read/Write mix workload.
# Load 50M keys, then run 100M mixed ops.
# Write ratio is percent writes (read ratio = 100 - write ratio).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_env.sh
source "${SCRIPT_DIR}/bench_env.sh"

VALUE_SIZES="${VALUE_SIZES:-"32 64 128 256"}"
WRITE_RATIOS="${WRITE_RATIOS:-"10 20 50"}"
LOAD_NUM="${LOAD_NUM:-50000000}"
MIX_NUM="${MIX_NUM:-100000000}"
MIX_BENCH="${MIX_BENCH:-readrandomwriterandom}"

for value_size in ${VALUE_SIZES}; do
  for write_ratio in ${WRITE_RATIOS}; do
    read_ratio=$((100 - write_ratio))

    RUN_DIR="${OUTPUT_DIR}/readwritemix/value_${value_size}/write_${write_ratio}/${RUN_ID}"
    DB_DIR="${DB_BASE_DIR}/readwritemix/value_${value_size}/write_${write_ratio}"
    WAL_DIR="${WAL_BASE_DIR}/readwritemix/value_${value_size}/write_${write_ratio}"

    mkdir -p "${RUN_DIR}" "${DB_DIR}" "${WAL_DIR}"

    run_db_bench "${RUN_DIR}/load.log" \
      --benchmarks=fillrandom \
      --num="${LOAD_NUM}" \
      --value_size="${value_size}" \
      --db="${DB_DIR}" \
      --wal_dir="${WAL_DIR}" \
      --report_file="${RUN_DIR}/load_report.csv" \
      --use_existing_db=0 \
      "${COMMON_FLAGS[@]}"

    run_db_bench "${RUN_DIR}/mix.log" \
      --benchmarks="${MIX_BENCH}" \
      --num="${MIX_NUM}" \
      --value_size="${value_size}" \
      --readwritepercent="${read_ratio}" \
      --db="${DB_DIR}" \
      --wal_dir="${WAL_DIR}" \
      --report_file="${RUN_DIR}/mix_report.csv" \
      --use_existing_db=1 \
      "${COMMON_FLAGS[@]}"
  done

done
