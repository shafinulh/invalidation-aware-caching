#!/usr/bin/env bash
set -euo pipefail

# Read/Write mix workload.
# Load keys once (if missing), then run mixed operations.
# Write ratio is percent writes (read ratio = 100 - write ratio).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=benchmark_common.sh
source "${SCRIPT_DIR}/benchmark_common.sh"

# ReadWriteMix-specific knobs.
NUM_KEYS="${NUM_KEYS:-200000000}"
NUM_LOADS="${NUM_LOADS:-50000000}"
LOAD_BENCH="${LOAD_BENCH:-fillrandom}"
PRELOAD_READY_FILENAME=".preload_ready"

WRITE_RATIOS="${WRITE_RATIOS:-"25"}"
MIX_READS="${MIX_READS:-20000000}"
MIX_BENCH="${MIX_BENCH:-readrandomwriterandom}"
bg_comp_threads_list="${BG_COMP_THREADS_LIST:-1}"

load_disable_auto_compactions=0
case "${LOAD_BENCH}" in
  fillseqdeterministic|filluniquerandomdeterministic)
    load_disable_auto_compactions=1
    ;;
esac

for value_size in ${VALUE_SIZES}; do
  PRELOAD_RUN_DIR="${OUTPUT_DIR}/readwritemix_preload/value_${value_size}/${RUN_ID}"
  PRELOAD_DB_DIR="${DB_BASE_DIR}/readwritemix_preload/value_${value_size}"
  PRELOAD_WAL_DIR="${WAL_BASE_DIR}/readwritemix_preload/value_${value_size}"
  PRELOAD_READY_FILE="${PRELOAD_DB_DIR}/${PRELOAD_READY_FILENAME}"

  if [[ ! -f "${PRELOAD_READY_FILE}" ]]; then
    cleanup_db_wal_dirs "${PRELOAD_DB_DIR}" "${PRELOAD_WAL_DIR}"
    mkdir -p "${PRELOAD_RUN_DIR}" "${PRELOAD_DB_DIR}" "${PRELOAD_WAL_DIR}"
    write_run_config "${PRELOAD_RUN_DIR}" "run_readwritemix.sh"

    run_db_bench "${PRELOAD_RUN_DIR}/load.log" \
      --benchmarks="${LOAD_BENCH}" \
      --disable_auto_compactions="${load_disable_auto_compactions}" \
      --num="${NUM_KEYS}" \
      --writes="${NUM_LOADS}" \
      --value_size="${value_size}" \
      --db="${PRELOAD_DB_DIR}" \
      --wal_dir="${PRELOAD_WAL_DIR}" \
      --report_file="${PRELOAD_RUN_DIR}/load_report.csv" \
      --use_existing_db=0 \
      --subcompactions=8 \
      --max_background_compactions=1 \
      "${COMMON_FLAGS[@]}"
    copy_latest_rocksdb_options "${PRELOAD_DB_DIR}" "${PRELOAD_RUN_DIR}" "after_load"
    copy_rocksdb_log_file "${PRELOAD_DB_DIR}" "${PRELOAD_RUN_DIR}" "after_load"
    touch "${PRELOAD_READY_FILE}"
  fi

  for bg_comp_threads in ${bg_comp_threads_list}; do
    for subcomp_threads in ${SUBCOMP_THREADS_LIST}; do

      for write_ratio in ${WRITE_RATIOS}; do
        read_ratio=$((100 - write_ratio))

        RUN_PREFIX="${OUTPUT_DIR}/readwritemix/value_${value_size}/subcomp_${subcomp_threads}"
        DB_PREFIX="${DB_BASE_DIR}/readwritemix/value_${value_size}/subcomp_${subcomp_threads}"
        WAL_PREFIX="${WAL_BASE_DIR}/readwritemix/value_${value_size}/subcomp_${subcomp_threads}"
        if [[ "${bg_comp_threads}" =~ ^[0-9]+$ ]] && (( bg_comp_threads > 1 )); then
          RUN_PREFIX="${RUN_PREFIX}/bgcomp_${bg_comp_threads}"
          DB_PREFIX="${DB_PREFIX}/bgcomp_${bg_comp_threads}"
          WAL_PREFIX="${WAL_PREFIX}/bgcomp_${bg_comp_threads}"
        fi

        RUN_DIR="${RUN_PREFIX}/write_${write_ratio}/${RUN_ID}"
        DB_DIR="${DB_PREFIX}/write_${write_ratio}"
        WAL_DIR="${WAL_PREFIX}/write_${write_ratio}"

        cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
        mkdir -p "${RUN_DIR}" "${DB_DIR}" "${WAL_DIR}"
        write_run_config "${RUN_DIR}" "run_readwritemix.sh"

        cp -a "${PRELOAD_DB_DIR}/." "${DB_DIR}/"
        if [[ -d "${PRELOAD_WAL_DIR}" ]]; then
          cp -a "${PRELOAD_WAL_DIR}/." "${WAL_DIR}/"
        fi

        run_db_bench "${RUN_DIR}/mix.log" \
          --benchmarks="${MIX_BENCH}" \
          --disable_auto_compactions=0 \
          --num="${NUM_KEYS}" \
          --reads="${MIX_READS}" \
          --value_size="${value_size}" \
          --readwritepercent="${read_ratio}" \
          --db="${DB_DIR}" \
          --wal_dir="${WAL_DIR}" \
          --report_file="${RUN_DIR}/mix_report.csv" \
          --use_existing_db=1 \
          --subcompactions="${subcomp_threads}" \
          --max_background_compactions="${bg_comp_threads}" \
          "${COMMON_FLAGS[@]}"
        copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_mix"
        copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_mix"
        cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
      done
    done
  done

done
