#!/usr/bin/env bash
set -euo pipefail

# FillRandom workload.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "${SCRIPT_DIR}/../.." && pwd)/common/benchmark_common.sh"

# FillRandom-specific knobs.
NUM_KEYS="${NUM_KEYS:-20000000}"
WRITES="${WRITES:--1}"
bg_comp_threads_list="${BG_COMP_THREADS_LIST:-1}"

lookup_writes_for_value_size() {
  local value_size="$1"
  local mapping

  if [[ -z "${WRITES_BY_VALUE_SIZE:-}" ]]; then
    printf "%s\n" "${WRITES}"
    return 0
  fi

  for mapping in ${WRITES_BY_VALUE_SIZE}; do
    if [[ "${mapping}" == "${value_size}:"* ]]; then
      printf "%s\n" "${mapping#*:}"
      return 0
    fi
  done

  printf "%s\n" "${WRITES}"
}

pause_between_runs_if_needed() {
  local current_index="$1"
  local total_runs="$2"

  if (( current_index >= total_runs )); then
    return 0
  fi

  if [[ "${RUN_PAUSE_SECONDS}" == "0" || "${RUN_PAUSE_SECONDS}" == "0.0" ]]; then
    return 0
  fi

  if [[ "${RUN_QUIET}" != "1" ]]; then
    echo "Cooling down for ${RUN_PAUSE_SECONDS}s before next fillrandom run..."
  fi
  sleep "${RUN_PAUSE_SECONDS}"
}

total_runs=0
for _value_size in ${VALUE_SIZES}; do
  for _bg_comp_threads in ${bg_comp_threads_list}; do
    for _subcomp_threads in ${SUBCOMP_THREADS_LIST}; do
      total_runs=$((total_runs + 1))
    done
  done
done

run_index=0

for value_size in ${VALUE_SIZES}; do
  for bg_comp_threads in ${bg_comp_threads_list}; do
    for subcomp_threads in ${SUBCOMP_THREADS_LIST}; do
      run_index=$((run_index + 1))

      writes_for_value_size="$(lookup_writes_for_value_size "${value_size}")"
      disable_auto_compactions="${DISABLE_AUTO_COMPACTIONS}"
      effective_subcompactions="${subcomp_threads}"
      if [[ "${subcomp_threads}" == "0" ]]; then
        disable_auto_compactions=1
        effective_subcompactions=0
      fi

      RUN_DIR="${OUTPUT_DIR}/fillrandom/value_${value_size}/${RUN_ID}/subcomp_${subcomp_threads}/bgcomp_${bg_comp_threads}"
      DB_DIR="${DB_BASE_DIR}/fillrandom/value_${value_size}"
      WAL_DIR="${WAL_BASE_DIR}/fillrandom/value_${value_size}"
      metrics_file_flag=()
      if [[ "${METRICS_INTERVAL_MS}" != "0" ]]; then
        metrics_file_flag=(--metrics_file="${RUN_DIR}/metrics.csv")
      fi

      mkdir -p "${RUN_DIR}" "${DB_DIR}" "${WAL_DIR}"
      write_run_config "${RUN_DIR}" "run_fillrandom.sh"
      write_kv_metadata "${RUN_DIR}/metadata/fillrandom_effective.env" \
        VALUE_SIZE "${value_size}" \
        WRITES "${writes_for_value_size}" \
        SUBCOMP_THREADS "${subcomp_threads}" \
        EFFECTIVE_SUBCOMPACTIONS "${effective_subcompactions}" \
        BG_COMP_THREADS "${bg_comp_threads}" \
        DISABLE_AUTO_COMPACTIONS "${disable_auto_compactions}"

      run_db_bench "${RUN_DIR}/db_bench.log" \
        --benchmarks=fillrandom \
        --disable_auto_compactions="${disable_auto_compactions}" \
        --num="${NUM_KEYS}" \
        --writes="${writes_for_value_size}" \
        --value_size="${value_size}" \
        --db="${DB_DIR}" \
        --wal_dir="${WAL_DIR}" \
        --report_file="${RUN_DIR}/report.csv" \
        --use_existing_db=0 \
        --subcompactions="${effective_subcompactions}" \
        --max_background_compactions="${bg_comp_threads}" \
        "${metrics_file_flag[@]}" \
        "${COMMON_FLAGS[@]}"

      copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_fillrandom"
      copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_fillrandom"
      cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
      pause_between_runs_if_needed "${run_index}" "${total_runs}"
    done
  done
done
