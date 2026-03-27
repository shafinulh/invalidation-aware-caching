#!/usr/bin/env bash
set -euo pipefail

# Isolated CPU compaction-scaling experiment.
#
# Flow:
#   1. Build a preload DB with fillrandom and auto-compactions disabled.
#   2. Reuse that preload for every requested subcompaction setting by copying
#      it into an ephemeral run-specific DB directory.
#   3. Trigger a manual compaction benchmark (compactall by default) and collect
#      RocksDB timing plus host-side CPU / IO metrics for the compaction phase.
#
# The target input size is expressed in logical KV bytes:
#   target_bytes = input_data_mb * 1024 * 1024
#   loads        = ceil(target_bytes / (key_size + value_size))
#
# Actual compaction input/output bytes are recovered later from RocksDB event
# logs so the analysis can report both the target logical size and the observed
# bytes processed by RocksDB.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CALLER_SCRIPT_DIR="${SCRIPT_DIR}"
source "$(cd "${SCRIPT_DIR}/../.." && pwd)/common/benchmark_common.sh"

PRELOAD_READY_FILENAME=".preload_ready"

PRELOAD_DIR_NAME="${PRELOAD_DIR_NAME:-cpu_compaction_sweep_preload}"
CPU_COMPACTION_OUT_ROOT="${CPU_COMPACTION_OUT_ROOT:-${OUTPUT_DIR}/cpu_compaction_sweep}"
CPU_COMPACTION_PRELOAD_ROOT="${CPU_COMPACTION_PRELOAD_ROOT:-${CPU_COMPACTION_OUT_ROOT}/preload}"
INPUT_DATA_MB_LIST="${INPUT_DATA_MB_LIST:-}"
INPUT_SST_COUNT_LIST="${INPUT_SST_COUNT_LIST:-}"
SST_SIZE_MB_LIST="${SST_SIZE_MB_LIST:-8 16 32 64}"
COMPACTION_BENCH="${COMPACTION_BENCH:-compactall}"
PRELOAD_THREADS="${PRELOAD_THREADS:-${THREADS}}"
PRELOAD_KEYSPACE_MULTIPLIER="${PRELOAD_KEYSPACE_MULTIPLIER:-1}"
PRELOAD_MIN_NUM_KEYS="${PRELOAD_MIN_NUM_KEYS:-200000000}"
COMPACTION_BG_THREADS="${COMPACTION_BG_THREADS:-1}"
COMPACTION_RUNS="${COMPACTION_RUNS:-1}"
PRESERVE_COMPACTION_OUTPUTS="${PRESERVE_COMPACTION_OUTPUTS:-0}"
DB_BENCH_EXTRA_FLAGS="${DB_BENCH_EXTRA_FLAGS:-}"
LOAD_DB_BENCH_EXTRA_FLAGS="${LOAD_DB_BENCH_EXTRA_FLAGS:-${DB_BENCH_EXTRA_FLAGS}}"
COMPACTION_DB_BENCH_EXTRA_FLAGS="${COMPACTION_DB_BENCH_EXTRA_FLAGS:-${DB_BENCH_EXTRA_FLAGS}}"

LOAD_PERF_LEVEL="${LOAD_PERF_LEVEL:-1}"
LOAD_METRICS_INTERVAL_MS="${LOAD_METRICS_INTERVAL_MS:-0}"
LOAD_HOST_METRICS_INTERVAL_SEC="${LOAD_HOST_METRICS_INTERVAL_SEC:-0}"
LOAD_THREAD_STATUS_PER_INTERVAL="${LOAD_THREAD_STATUS_PER_INTERVAL:-0}"

COMPACTION_PERF_LEVEL="${COMPACTION_PERF_LEVEL:-5}"
COMPACTION_METRICS_INTERVAL_MS="${COMPACTION_METRICS_INTERVAL_MS:-${METRICS_INTERVAL_MS}}"
COMPACTION_HOST_METRICS_INTERVAL_SEC="${COMPACTION_HOST_METRICS_INTERVAL_SEC:-${HOST_METRICS_INTERVAL_SEC}}"
COMPACTION_THREAD_STATUS_PER_INTERVAL="${COMPACTION_THREAD_STATUS_PER_INTERVAL:-${THREAD_STATUS_PER_INTERVAL}}"

# Keep automatic L0-triggered compactions disabled for this isolated flow.
EXPERIMENT_LEVEL0_FILE_TRIGGER="${EXPERIMENT_LEVEL0_FILE_TRIGGER:-1000000}"
EXPERIMENT_LEVEL0_SLOWDOWN_TRIGGER="${EXPERIMENT_LEVEL0_SLOWDOWN_TRIGGER:-1000000}"
EXPERIMENT_LEVEL0_STOP_TRIGGER="${EXPERIMENT_LEVEL0_STOP_TRIGGER:-1000000}"

case "${COMPACTION_BENCH}" in
  compact0|compactall|compact1)
    ;;
  *)
    echo "error: COMPACTION_BENCH must be one of compact0, compact1, or compactall (got: ${COMPACTION_BENCH})" >&2
    exit 1
    ;;
esac

if [[ -z "${PRELOAD_DIR_NAME}" || "${PRELOAD_DIR_NAME}" == "." || "${PRELOAD_DIR_NAME}" == ".." || "${PRELOAD_DIR_NAME}" == */* ]]; then
  echo "error: PRELOAD_DIR_NAME must be a directory name without path separators (got: ${PRELOAD_DIR_NAME})" >&2
  exit 1
fi

if [[ -n "${INPUT_SST_COUNT_LIST}" && -n "${INPUT_DATA_MB_LIST}" ]]; then
  echo "error: set either INPUT_SST_COUNT_LIST or INPUT_DATA_MB_LIST, not both" >&2
  exit 1
fi

if [[ -z "${INPUT_SST_COUNT_LIST}" && -z "${INPUT_DATA_MB_LIST}" ]]; then
  INPUT_DATA_MB_LIST="32 64 128 256 512 1024 2048"
fi

ceil_div() {
  local numerator="$1"
  local denominator="$2"
  printf "%s\n" "$(( (numerator + denominator - 1) / denominator ))"
}

split_flag_string() {
  local flag_string="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ -n "${flag_string}" ]]; then
    # Deliberately split on shell whitespace for simple extra-flag injection.
    read -r -a out_ref <<< "${flag_string}"
  fi
}

compute_target_bytes() {
  local input_data_mb="$1"
  printf "%s\n" "$(( input_data_mb * 1024 * 1024 ))"
}

repeat_dir_name() {
  local repeat_index="$1"
  printf "repeat_%02d\n" "${repeat_index}"
}

emit_input_specs() {
  local sst_size_mb="$1"

  if [[ -n "${INPUT_SST_COUNT_LIST}" ]]; then
    local input_sst_count
    local input_data_mb
    for input_sst_count in ${INPUT_SST_COUNT_LIST}; do
      if [[ ! "${input_sst_count}" =~ ^[0-9]+$ ]] || (( input_sst_count <= 0 )); then
        echo "error: INPUT_SST_COUNT_LIST must contain positive integers (got: ${input_sst_count})" >&2
        exit 1
      fi
      input_data_mb=$(( input_sst_count * sst_size_mb ))
      printf "%s|%s|input_%ssst\n" "${input_data_mb}" "${input_sst_count}" "${input_sst_count}"
    done
    return 0
  fi

  local input_data_mb
  for input_data_mb in ${INPUT_DATA_MB_LIST}; do
    if [[ ! "${input_data_mb}" =~ ^[0-9]+$ ]] || (( input_data_mb <= 0 )); then
      echo "error: INPUT_DATA_MB_LIST must contain positive integers (got: ${input_data_mb})" >&2
      exit 1
    fi
    printf "%s|0|input_%sMB\n" "${input_data_mb}" "${input_data_mb}"
  done
}

compute_load_entries() {
  local input_data_mb="$1"
  local value_size="$2"
  local entry_bytes=$(( KEY_SIZE + value_size ))
  local target_bytes
  target_bytes="$(compute_target_bytes "${input_data_mb}")"
  ceil_div "${target_bytes}" "${entry_bytes}"
}

compute_num_keys() {
  local load_entries="$1"
  local num_keys=$(( load_entries * PRELOAD_KEYSPACE_MULTIPLIER ))
  if (( num_keys < PRELOAD_MIN_NUM_KEYS )); then
    num_keys="${PRELOAD_MIN_NUM_KEYS}"
  fi
  printf "%s\n" "${num_keys}"
}

dir_size_bytes() {
  local target_dir="$1"
  if [[ ! -d "${target_dir}" ]]; then
    printf "0\n"
    return 0
  fi
  du -sb "${target_dir}" 2>/dev/null | awk '{print $1}'
}

sst_size_bytes() {
  local db_dir="$1"
  if [[ ! -d "${db_dir}" ]]; then
    printf "0\n"
    return 0
  fi
  find "${db_dir}" -maxdepth 1 -name "*.sst" -printf "%s\n" 2>/dev/null | awk '{sum += $1} END {print sum + 0}'
}

sst_file_count() {
  local db_dir="$1"
  if [[ ! -d "${db_dir}" ]]; then
    printf "0\n"
    return 0
  fi
  find "${db_dir}" -maxdepth 1 -name "*.sst" 2>/dev/null | wc -l | tr -d ' '
}

copy_tree_contents() {
  local src_dir="$1"
  local dst_dir="$2"

  if cp --help 2>/dev/null | grep -q -- "--reflink"; then
    cp -a --reflink=auto "${src_dir}/." "${dst_dir}/"
  else
    cp -a "${src_dir}/." "${dst_dir}/"
  fi
}

build_common_flags() {
  local threads_value="$1"
  local perf_level_value="$2"
  local metrics_interval_ms_value="$3"
  local report_bg_io_stats_value="$4"
  local thread_status_value="$5"
  local write_buffer_size_bytes="$6"
  local target_file_size_bytes="$7"
  local max_bytes_for_level_base_bytes="$8"
  local -n out_ref="$9"

  local thread_status_seen=0
  out_ref=()

  for flag in "${COMMON_FLAGS[@]}"; do
    case "${flag}" in
      --threads=*)
        out_ref+=(--threads="${threads_value}")
        ;;
      --perf_level=*)
        out_ref+=(--perf_level="${perf_level_value}")
        ;;
      --metrics_interval_ms=*)
        out_ref+=(--metrics_interval_ms="${metrics_interval_ms_value}")
        ;;
      --report_bg_io_stats=*)
        out_ref+=(--report_bg_io_stats="${report_bg_io_stats_value}")
        ;;
      --thread_status_per_interval=*)
        thread_status_seen=1
        if [[ "${thread_status_value}" != "0" ]]; then
          out_ref+=(--thread_status_per_interval="${thread_status_value}")
        fi
        ;;
      --write_buffer_size=*)
        out_ref+=(--write_buffer_size="${write_buffer_size_bytes}")
        ;;
      --target_file_size_base=*)
        out_ref+=(--target_file_size_base="${target_file_size_bytes}")
        ;;
      --max_bytes_for_level_base=*)
        out_ref+=(--max_bytes_for_level_base="${max_bytes_for_level_base_bytes}")
        ;;
      --level0_file_num_compaction_trigger=*)
        out_ref+=(--level0_file_num_compaction_trigger="${EXPERIMENT_LEVEL0_FILE_TRIGGER}")
        ;;
      --level0_slowdown_writes_trigger=*)
        out_ref+=(--level0_slowdown_writes_trigger="${EXPERIMENT_LEVEL0_SLOWDOWN_TRIGGER}")
        ;;
      --level0_stop_writes_trigger=*)
        out_ref+=(--level0_stop_writes_trigger="${EXPERIMENT_LEVEL0_STOP_TRIGGER}")
        ;;
      *)
        out_ref+=("${flag}")
        ;;
    esac
  done

  if (( thread_status_seen == 0 )) && [[ "${thread_status_value}" != "0" ]]; then
    out_ref+=(--thread_status_per_interval="${thread_status_value}")
  fi
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
    echo "Cooling down for ${RUN_PAUSE_SECONDS}s before next compaction run..."
  fi
  sleep "${RUN_PAUSE_SECONDS}"
}

run_with_host_metrics_interval() {
  local host_interval_value="$1"
  shift
  local previous_host_interval="${HOST_METRICS_INTERVAL_SEC}"
  local cmd_rc=0
  HOST_METRICS_INTERVAL_SEC="${host_interval_value}"
  if "$@"; then
    cmd_rc=0
  else
    cmd_rc=$?
  fi
  HOST_METRICS_INTERVAL_SEC="${previous_host_interval}"
  return "${cmd_rc}"
}

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${CPU_COMPACTION_OUT_ROOT}/${RUN_ID}}"
mkdir -p "${EXPERIMENT_ROOT}/metadata" "${EXPERIMENT_ROOT}/per_run_logs"
load_db_bench_extra_flags=()
compaction_db_bench_extra_flags=()
split_flag_string "${LOAD_DB_BENCH_EXTRA_FLAGS}" load_db_bench_extra_flags
split_flag_string "${COMPACTION_DB_BENCH_EXTRA_FLAGS}" \
  compaction_db_bench_extra_flags
write_kv_metadata "${EXPERIMENT_ROOT}/metadata/experiment.env" \
  RUN_ID "${RUN_ID}" \
  COMPACTION_BENCH "${COMPACTION_BENCH}" \
  KEY_SIZE "${KEY_SIZE}" \
  INPUT_DATA_MB_LIST "${INPUT_DATA_MB_LIST}" \
  INPUT_SST_COUNT_LIST "${INPUT_SST_COUNT_LIST}" \
  COMPACTION_RUNS "${COMPACTION_RUNS}" \
  PRESERVE_COMPACTION_OUTPUTS "${PRESERVE_COMPACTION_OUTPUTS}" \
  PRELOAD_THREADS "${PRELOAD_THREADS}" \
  PRELOAD_KEYSPACE_MULTIPLIER "${PRELOAD_KEYSPACE_MULTIPLIER}" \
  PRELOAD_MIN_NUM_KEYS "${PRELOAD_MIN_NUM_KEYS}" \
  COMPACTION_BG_THREADS "${COMPACTION_BG_THREADS}" \
  LOAD_DB_BENCH_EXTRA_FLAGS "${LOAD_DB_BENCH_EXTRA_FLAGS}" \
  COMPACTION_DB_BENCH_EXTRA_FLAGS "${COMPACTION_DB_BENCH_EXTRA_FLAGS}" \
  COMPACTION_PERF_LEVEL "${COMPACTION_PERF_LEVEL}" \
  COMPACTION_METRICS_INTERVAL_MS "${COMPACTION_METRICS_INTERVAL_MS}" \
  COMPACTION_HOST_METRICS_INTERVAL_SEC "${COMPACTION_HOST_METRICS_INTERVAL_SEC}" \
  EXPERIMENT_LEVEL0_FILE_TRIGGER "${EXPERIMENT_LEVEL0_FILE_TRIGGER}" \
  EXPERIMENT_LEVEL0_SLOWDOWN_TRIGGER "${EXPERIMENT_LEVEL0_SLOWDOWN_TRIGGER}" \
  EXPERIMENT_LEVEL0_STOP_TRIGGER "${EXPERIMENT_LEVEL0_STOP_TRIGGER}"

total_runs=0
for _sst_size_mb in ${SST_SIZE_MB_LIST}; do
  while IFS='|' read -r _input_data_mb _requested_input_sst_count _input_path_component; do
    [[ -n "${_input_data_mb}" ]] || continue
    for _value_size in ${VALUE_SIZES}; do
      for _subcomp_threads in ${SUBCOMP_THREADS_LIST}; do
        total_runs=$((total_runs + COMPACTION_RUNS))
      done
    done
  done < <(emit_input_specs "${_sst_size_mb}")
done

run_index=0

for sst_size_mb in ${SST_SIZE_MB_LIST}; do
  write_buffer_size_bytes=$(( sst_size_mb * 1024 * 1024 ))
  target_file_size_bytes="${write_buffer_size_bytes}"
  max_bytes_for_level_base_bytes=$(( write_buffer_size_bytes * 8 ))

  load_common_flags=()
  compact_common_flags=()
  build_common_flags \
    "${PRELOAD_THREADS}" \
    "${LOAD_PERF_LEVEL}" \
    "${LOAD_METRICS_INTERVAL_MS}" \
    "false" \
    "${LOAD_THREAD_STATUS_PER_INTERVAL}" \
    "${write_buffer_size_bytes}" \
    "${target_file_size_bytes}" \
    "${max_bytes_for_level_base_bytes}" \
    load_common_flags
  build_common_flags \
    "1" \
    "${COMPACTION_PERF_LEVEL}" \
    "${COMPACTION_METRICS_INTERVAL_MS}" \
    "true" \
    "${COMPACTION_THREAD_STATUS_PER_INTERVAL}" \
    "${write_buffer_size_bytes}" \
    "${target_file_size_bytes}" \
    "${max_bytes_for_level_base_bytes}" \
    compact_common_flags

  while IFS='|' read -r input_data_mb requested_input_sst_count input_path_component; do
    [[ -n "${input_data_mb}" ]] || continue
    target_logical_bytes="$(compute_target_bytes "${input_data_mb}")"
    input_desc="${input_data_mb}MB"
    if (( requested_input_sst_count > 0 )); then
      input_desc="${requested_input_sst_count}sst (${input_data_mb}MB)"
    fi

    for value_size in ${VALUE_SIZES}; do
      load_entries="$(compute_load_entries "${input_data_mb}" "${value_size}")"
      num_keys="$(compute_num_keys "${load_entries}")"

      preload_run_dir="${CPU_COMPACTION_PRELOAD_ROOT}/${PRELOAD_DIR_NAME}/sst_${sst_size_mb}MB/${input_path_component}/value_${value_size}/${RUN_ID}"
      preload_db_dir="${DB_BASE_DIR}/${PRELOAD_DIR_NAME}/sst_${sst_size_mb}MB/${input_path_component}/value_${value_size}"
      preload_wal_dir="${WAL_BASE_DIR}/${PRELOAD_DIR_NAME}/sst_${sst_size_mb}MB/${input_path_component}/value_${value_size}"
      preload_ready_file="${preload_db_dir}/${PRELOAD_READY_FILENAME}"

      if [[ ! -f "${preload_ready_file}" ]]; then
        if [[ "${RUN_QUIET}" != "1" ]]; then
          echo "================================================================"
          echo "  Building preload: sst=${sst_size_mb}MB input=${input_desc} value=${value_size}B"
          echo "  Preload writes: ${load_entries}  num_keys: ${num_keys}"
          echo "================================================================"
        fi

        cleanup_db_wal_dirs "${preload_db_dir}" "${preload_wal_dir}"
        mkdir -p "${preload_run_dir}" "${preload_db_dir}" "${preload_wal_dir}"
        write_run_config "${preload_run_dir}" "cpu_compaction_sweep.sh"
        write_kv_metadata "${preload_run_dir}/metadata/preload.env" \
          SST_SIZE_MB "${sst_size_mb}" \
          INPUT_DATA_MB "${input_data_mb}" \
          REQUESTED_INPUT_SST_COUNT "${requested_input_sst_count}" \
          INPUT_PATH_COMPONENT "${input_path_component}" \
          TARGET_LOGICAL_INPUT_BYTES "${target_logical_bytes}" \
          KEY_SIZE "${KEY_SIZE}" \
          VALUE_SIZE "${value_size}" \
          LOAD_ENTRIES "${load_entries}" \
          NUM_KEYS "${num_keys}" \
          PRELOAD_THREADS "${PRELOAD_THREADS}" \
          WRITE_BUFFER_SIZE_BYTES "${write_buffer_size_bytes}" \
          TARGET_FILE_SIZE_BASE_BYTES "${target_file_size_bytes}" \
          MAX_BYTES_FOR_LEVEL_BASE_BYTES "${max_bytes_for_level_base_bytes}" \
          COMPACTION_BENCH "${COMPACTION_BENCH}"

        run_with_host_metrics_interval "${LOAD_HOST_METRICS_INTERVAL_SEC}" \
          run_db_bench "${preload_run_dir}/load.log" \
            --benchmarks=fillrandom \
            --disable_auto_compactions=1 \
            --num="${num_keys}" \
            --writes="${load_entries}" \
            --value_size="${value_size}" \
            --db="${preload_db_dir}" \
            --wal_dir="${preload_wal_dir}" \
            --report_file="${preload_run_dir}/load_report.csv" \
            --use_existing_db=0 \
            --subcompactions=1 \
            --max_background_compactions=1 \
            "${load_db_bench_extra_flags[@]}" \
            "${load_common_flags[@]}"

        copy_latest_rocksdb_options "${preload_db_dir}" "${preload_run_dir}" "after_load"
        copy_rocksdb_log_file "${preload_db_dir}" "${preload_run_dir}" "after_load"

        preload_sst_count="$(sst_file_count "${preload_db_dir}")"
        preload_sst_bytes="$(sst_size_bytes "${preload_db_dir}")"
        preload_db_bytes="$(dir_size_bytes "${preload_db_dir}")"
        write_kv_metadata "${preload_run_dir}/metadata/preload_summary.env" \
          PRELOAD_SST_COUNT "${preload_sst_count}" \
          PRELOAD_SST_BYTES "${preload_sst_bytes}" \
          PRELOAD_DB_BYTES "${preload_db_bytes}"

        touch "${preload_ready_file}"
      fi

      preload_sst_count="$(sst_file_count "${preload_db_dir}")"
      preload_sst_bytes="$(sst_size_bytes "${preload_db_dir}")"
      preload_db_bytes="$(dir_size_bytes "${preload_db_dir}")"

      for subcomp_threads in ${SUBCOMP_THREADS_LIST}; do
        run_dir_base="${EXPERIMENT_ROOT}/sst_${sst_size_mb}MB/${input_path_component}/value_${value_size}/subcomp_${subcomp_threads}"
        db_dir_base="${DB_BASE_DIR}/cpu_compaction_sweep/${RUN_ID}/sst_${sst_size_mb}MB/${input_path_component}/value_${value_size}/subcomp_${subcomp_threads}"
        wal_dir_base="${WAL_BASE_DIR}/cpu_compaction_sweep/${RUN_ID}/sst_${sst_size_mb}MB/${input_path_component}/value_${value_size}/subcomp_${subcomp_threads}"

        for (( repeat_index=1; repeat_index<=COMPACTION_RUNS; repeat_index++ )); do
          run_index=$((run_index + 1))

          RUN_DIR="${run_dir_base}"
          DB_DIR="${db_dir_base}"
          WAL_DIR="${wal_dir_base}"
          if (( COMPACTION_RUNS > 1 )); then
            repeat_component="$(repeat_dir_name "${repeat_index}")"
            RUN_DIR="${run_dir_base}/${repeat_component}"
            DB_DIR="${db_dir_base}/${repeat_component}"
            WAL_DIR="${wal_dir_base}/${repeat_component}"
          fi

          cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
          mkdir -p "${RUN_DIR}" "${DB_DIR}" "${WAL_DIR}"
          write_run_config "${RUN_DIR}" "cpu_compaction_sweep.sh"
          write_kv_metadata "${RUN_DIR}/metadata/cpu_compaction_sweep.env" \
            SST_SIZE_MB "${sst_size_mb}" \
            INPUT_DATA_MB "${input_data_mb}" \
            REQUESTED_INPUT_SST_COUNT "${requested_input_sst_count}" \
            INPUT_PATH_COMPONENT "${input_path_component}" \
            TARGET_LOGICAL_INPUT_BYTES "${target_logical_bytes}" \
            KEY_SIZE "${KEY_SIZE}" \
            VALUE_SIZE "${value_size}" \
            LOAD_ENTRIES "${load_entries}" \
            NUM_KEYS "${num_keys}" \
            REQUESTED_SUBCOMPACTIONS "${subcomp_threads}" \
            COMPACTION_BG_THREADS "${COMPACTION_BG_THREADS}" \
            COMPACTION_BENCH "${COMPACTION_BENCH}" \
            COMPACTION_RUNS "${COMPACTION_RUNS}" \
            COMPACTION_REPEAT_INDEX "${repeat_index}" \
            PRESERVE_COMPACTION_OUTPUTS "${PRESERVE_COMPACTION_OUTPUTS}" \
            WRITE_BUFFER_SIZE_BYTES "${write_buffer_size_bytes}" \
            TARGET_FILE_SIZE_BASE_BYTES "${target_file_size_bytes}" \
            MAX_BYTES_FOR_LEVEL_BASE_BYTES "${max_bytes_for_level_base_bytes}" \
            PRELOAD_SST_COUNT "${preload_sst_count}" \
            PRELOAD_SST_BYTES "${preload_sst_bytes}" \
            PRELOAD_DB_BYTES "${preload_db_bytes}" \
            PRELOAD_DB_DIR "${preload_db_dir}" \
            PRELOAD_RUN_DIR "${preload_run_dir}" \
            COMPACTION_PERF_LEVEL "${COMPACTION_PERF_LEVEL}" \
            COMPACTION_METRICS_INTERVAL_MS "${COMPACTION_METRICS_INTERVAL_MS}" \
            COMPACTION_HOST_METRICS_INTERVAL_SEC "${COMPACTION_HOST_METRICS_INTERVAL_SEC}"

          copy_tree_contents "${preload_db_dir}" "${DB_DIR}"
          if [[ -d "${preload_wal_dir}" ]]; then
            copy_tree_contents "${preload_wal_dir}" "${WAL_DIR}"
          fi

          copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "before_compact"

          metrics_file_flag=()
          if [[ "${COMPACTION_METRICS_INTERVAL_MS}" != "0" ]]; then
            metrics_file_flag=(--metrics_file="${RUN_DIR}/metrics.csv")
          fi

          if [[ "${RUN_QUIET}" != "1" ]]; then
            echo "================================================================"
            echo "  Manual compaction: sst=${sst_size_mb}MB input=${input_desc} value=${value_size}B subcomp=${subcomp_threads} repeat=${repeat_index}/${COMPACTION_RUNS}"
            echo "  RUN_DIR: ${RUN_DIR}"
            echo "================================================================"
          fi

          run_with_host_metrics_interval "${COMPACTION_HOST_METRICS_INTERVAL_SEC}" \
            run_db_bench "${RUN_DIR}/compact.log" \
              --benchmarks="${COMPACTION_BENCH},stats" \
              --disable_auto_compactions=0 \
              --num="${num_keys}" \
              --value_size="${value_size}" \
              --db="${DB_DIR}" \
              --wal_dir="${WAL_DIR}" \
              --report_file="${RUN_DIR}/compact_report.csv" \
              --use_existing_db=1 \
              --subcompactions="${subcomp_threads}" \
              --max_background_compactions="${COMPACTION_BG_THREADS}" \
              "${metrics_file_flag[@]}" \
              "${compaction_db_bench_extra_flags[@]}" \
              "${compact_common_flags[@]}"

          if [[ "${COMPACTION_BENCH}" == "compact0" || "${COMPACTION_BENCH}" == "compact1" ]]; then
            if grep -Eq "found no data beyond L[0-9]+|found 0 files to compact" "${RUN_DIR}/compact.log"; then
              echo "error: ${COMPACTION_BENCH} was a no-op for ${RUN_DIR}" >&2
              echo "hint: db_bench ${COMPACTION_BENCH} requires an existing lower level to compact into." >&2
              echo "hint: use COMPACTION_BENCH=compactall for the pure L0 preload flow implemented by this experiment." >&2
              exit 1
            fi
          fi

          copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_compact"
          copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_compact"

          post_compact_sst_count="$(sst_file_count "${DB_DIR}")"
          post_compact_sst_bytes="$(sst_size_bytes "${DB_DIR}")"
          post_compact_db_bytes="$(dir_size_bytes "${DB_DIR}")"
          write_kv_metadata "${RUN_DIR}/metadata/post_compact.env" \
            POST_COMPACT_SST_COUNT "${post_compact_sst_count}" \
            POST_COMPACT_SST_BYTES "${post_compact_sst_bytes}" \
            POST_COMPACT_DB_BYTES "${post_compact_db_bytes}"

          if [[ "${PRESERVE_COMPACTION_OUTPUTS}" == "1" ]]; then
            rm -rf "${RUN_DIR}/rocksdb_output"
            mkdir -p "${RUN_DIR}/rocksdb_output"
            mv "${DB_DIR}" "${RUN_DIR}/rocksdb_output/db"
            if [[ -d "${WAL_DIR}" ]]; then
              mv "${WAL_DIR}" "${RUN_DIR}/rocksdb_output/wal"
            fi
          else
            cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
          fi

          pause_between_runs_if_needed "${run_index}" "${total_runs}"
        done
      done
    done
  done < <(emit_input_specs "${sst_size_mb}")
done

echo ""
echo "CPU compaction sweep complete."
echo "Run root: ${EXPERIMENT_ROOT}"

ANALYZE="${ANALYZE:-0}"
if [[ "${ANALYZE}" == "1" ]]; then
  ANALYZER="$(cd "${_CALLER_SCRIPT_DIR}/.." && pwd)/plotting/analyze_compaction_parallelism.py"
  echo ""
  echo "--- Running analysis ---"
  python3 "${ANALYZER}" "${EXPERIMENT_ROOT}"
  echo "Analysis complete. Results in: ${EXPERIMENT_ROOT}"
else
  echo "Analyze with:"
  echo "  python3 benchmarks/cpu_compaction/plotting/analyze_compaction_parallelism.py ${EXPERIMENT_ROOT}"
fi
echo ""
