#!/usr/bin/env bash
set -euo pipefail

# Read/Write mix workload.
# Load keys once (if missing), then run mixed operations.
# Write ratio is percent writes (read ratio = 100 - write ratio).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common/benchmark_common.sh
source "$(cd "${SCRIPT_DIR}/../.." && pwd)/common/benchmark_common.sh"

# ReadWriteMix-specific knobs.
NUM_KEYS="${NUM_KEYS:-200000000}"
NUM_LOADS="${NUM_LOADS:-50000000}"
LOAD_BENCH="${LOAD_BENCH:-fillrandom}"
PRELOAD_READY_FILENAME=".preload_ready"

WRITE_RATIOS="${WRITE_RATIOS:-"25"}"
MIX_READS="${MIX_READS:-20000000}"
MIX_BENCH="${MIX_BENCH:-readrandomwriterandom}"
bg_comp_threads_list="${BG_COMP_THREADS_LIST:-1}"
PRELOAD_DIR_NAME="${PRELOAD_DIR_NAME:-readwritemix_preload}"
COMPACT_PRELOAD_ON_CREATE="${COMPACT_PRELOAD_ON_CREATE:-0}"

case "${COMPACT_PRELOAD_ON_CREATE}" in
  0|1)
    ;;
  *)
    echo "error: COMPACT_PRELOAD_ON_CREATE must be 0 or 1 (got: ${COMPACT_PRELOAD_ON_CREATE})" >&2
    exit 1
    ;;
esac

if [[ -z "${PRELOAD_DIR_NAME}" || "${PRELOAD_DIR_NAME}" == "." || "${PRELOAD_DIR_NAME}" == ".." || "${PRELOAD_DIR_NAME}" == */* ]]; then
  echo "error: PRELOAD_DIR_NAME must be a directory name without path separators (got: ${PRELOAD_DIR_NAME})" >&2
  exit 1
fi

load_disable_auto_compactions=0
load_threads="${THREADS}"
case "${LOAD_BENCH}" in
  fillseqdeterministic|filluniquerandomdeterministic)
    load_disable_auto_compactions=1
    load_threads=1
    ;;
esac

load_common_flags=()
for flag in "${COMMON_FLAGS[@]}"; do
  case "${flag}" in
    --threads=*)
      load_common_flags+=("--threads=${load_threads}")
      ;;
    *)
      load_common_flags+=("${flag}")
      ;;
  esac
done

for value_size in ${VALUE_SIZES}; do
  preload_run_dir="${OUTPUT_DIR}/${PRELOAD_DIR_NAME}/value_${value_size}/${RUN_ID}"
  preload_db_dir="${DB_BASE_DIR}/${PRELOAD_DIR_NAME}/value_${value_size}"
  preload_wal_dir="${WAL_BASE_DIR}/${PRELOAD_DIR_NAME}/value_${value_size}"
  preload_ready_file="${preload_db_dir}/${PRELOAD_READY_FILENAME}"

  if [[ ! -f "${preload_ready_file}" ]]; then
    cleanup_db_wal_dirs "${preload_db_dir}" "${preload_wal_dir}"
    mkdir -p "${preload_run_dir}" "${preload_db_dir}" "${preload_wal_dir}"
    write_run_config "${preload_run_dir}" "run_readwritemix.sh"

    load_status=0
    set +e
    run_db_bench "${preload_run_dir}/load.log" \
      --benchmarks="${LOAD_BENCH}" \
      --disable_auto_compactions="${load_disable_auto_compactions}" \
      --num="${NUM_KEYS}" \
      --writes="${NUM_LOADS}" \
      --value_size="${value_size}" \
      --db="${preload_db_dir}" \
      --wal_dir="${preload_wal_dir}" \
      --report_file="${preload_run_dir}/load_report.csv" \
      --metrics_file="${preload_run_dir}/load_metrics.csv" \
      --use_existing_db=0 \
      --subcompactions=8 \
      --max_background_compactions=1 \
      "${load_common_flags[@]}"
    load_status=$?
    set -e

    if (( load_status != 0 )); then
      case "${LOAD_BENCH}" in
        fillseqdeterministic|filluniquerandomdeterministic)
          if compgen -G "${preload_db_dir}/*.sst" > /dev/null; then
            echo "warning: ${LOAD_BENCH} exited with status ${load_status}; continuing because preload SSTs were created" >&2
          else
            exit "${load_status}"
          fi
          ;;
        *)
          exit "${load_status}"
          ;;
      esac
    fi

    copy_latest_rocksdb_options "${preload_db_dir}" "${preload_run_dir}" "after_load"
    copy_rocksdb_log_file "${preload_db_dir}" "${preload_run_dir}" "after_load"

    if [[ "${COMPACT_PRELOAD_ON_CREATE}" == "1" ]]; then
      run_db_bench "${preload_run_dir}/compact.log" \
        --benchmarks=compactall,stats \
        --disable_auto_compactions=0 \
        --num="${NUM_KEYS}" \
        --value_size="${value_size}" \
        --db="${preload_db_dir}" \
        --wal_dir="${preload_wal_dir}" \
        --report_file="${preload_run_dir}/compact_report.csv" \
        --metrics_file="${preload_run_dir}/compact_metrics.csv" \
        --use_existing_db=1 \
        --subcompactions=1 \
        --max_background_compactions=1 \
        "${COMMON_FLAGS[@]}"
      copy_latest_rocksdb_options "${preload_db_dir}" "${preload_run_dir}" "after_preload_compact"
      copy_rocksdb_log_file "${preload_db_dir}" "${preload_run_dir}" "after_preload_compact"
    fi

    touch "${preload_ready_file}"
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

        cp -a "${preload_db_dir}/." "${DB_DIR}/"
        if [[ -d "${preload_wal_dir}" ]]; then
          cp -a "${preload_wal_dir}/." "${WAL_DIR}/"
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
          --metrics_file="${RUN_DIR}/metrics.csv" \
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
