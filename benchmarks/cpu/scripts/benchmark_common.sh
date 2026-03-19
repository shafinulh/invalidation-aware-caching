#!/usr/bin/env bash
set -euo pipefail

# Shared benchmark environment setup.
# Source required machine-local config from benchmarks/cpu/config/.env.local.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_ENV_FILE="${BENCH_ROOT_DIR}/config/.env.local"

if [[ ! -f "${BENCH_ENV_FILE}" ]]; then
  echo "error: missing ${BENCH_ENV_FILE}" >&2
  echo "hint: copy ${BENCH_ROOT_DIR}/config/.env.example to ${BENCH_ENV_FILE} and edit it." >&2
  exit 1
fi

CONFIG_KEYS=(
  # Required paths
  DB_BENCH
  DB_BASE_DIR
  WAL_BASE_DIR
  OUTPUT_DIR

  # Run metadata
  RUN_ID

  # Workload knobs
  THREADS
  KEY_SIZE
  VALUE_SIZES
  SEED

  # CPU multithreading experimentation knobs
  SUBCOMP_THREADS_LIST
  BG_COMP_THREADS_LIST

  # LSM tree config knobs (additional defaults)
  COMPRESSION_TYPE
  MAX_BACKGROUND_FLUSHES
  MAX_WRITE_BUFFER_NUMBER
  DISABLE_WAL
  DISABLE_AUTO_COMPACTIONS

  # OS knobs
  DIRECT_IO
  OPEN_FILES

  # LSM tree config knobs (GP-Comp paper-specific defaults)
  WRITE_BUFFER_SIZE
  TARGET_FILE_SIZE_BASE
  MAX_BYTES_FOR_LEVEL_BASE
  BLOCK_RESTART_INTERVAL
  BLOCK_SIZE
  BLOOM_BITS
  LEVEL0_FILE_NUM_COMPACTION_TRIGGER
  LEVEL0_SLOWDOWN_WRITES_TRIGGER
  LEVEL0_STOP_WRITES_TRIGGER

  # Workload-specific knobs
  NUM_KEYS
  NUM_LOADS
  WRITES
  WRITES_BY_VALUE_SIZE
  WRITE_RATIOS
  MIX_READS
  LOAD_BENCH
  MIX_BENCH
  PRELOAD_DIR_NAME
  COMPACT_PRELOAD_ON_CREATE

  # Key distribution knobs
  KEY_DIST
  ZIPF_ALPHA

  # Block cache knobs
  CACHE_SIZE

  # Metrics collection knobs
  METRICS_INTERVAL_MS
  THREAD_STATUS_PER_INTERVAL
  REPORT_BG_IO_STATS
  PERF_LEVEL
  HOST_METRICS_INTERVAL_SEC
  HOST_METRICS_DEVICE
  RUN_PAUSE_SECONDS
  RUN_QUIET
)

# Preserve caller-provided env so inline overrides used over .env.local
for key in "${CONFIG_KEYS[@]}"; do
  cli_key="__CLI_${key}"
  if [[ -n "${!key+x}" ]]; then
    printf -v "${cli_key}" "%s" "${!key}"
  fi
done

set -a
# shellcheck disable=SC1090
source "${BENCH_ENV_FILE}"
set +a

for key in "${CONFIG_KEYS[@]}"; do
  cli_key="__CLI_${key}"
  if [[ -n "${!cli_key+x}" ]]; then
    export "${key}=${!cli_key}"
    unset "${cli_key}"
  fi
done

require_env() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "error: ${var_name} must be set in ${BENCH_ENV_FILE}" >&2
    exit 1
  fi
}

require_env DB_BENCH
require_env DB_BASE_DIR
require_env WAL_BASE_DIR
require_env OUTPUT_DIR

if [[ ! -x "${DB_BENCH}" ]]; then
  echo "error: DB_BENCH is not executable: ${DB_BENCH}" >&2
  exit 1
fi

# Run metadata
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M)}"

# LSM tree config settings (GP-Comp paper defaults)
WRITE_BUFFER_SIZE="${WRITE_BUFFER_SIZE:-8388608}"  # 8 MB
TARGET_FILE_SIZE_BASE="${TARGET_FILE_SIZE_BASE:-8388608}"  # 8 MB
MAX_BYTES_FOR_LEVEL_BASE="${MAX_BYTES_FOR_LEVEL_BASE:-67108864}"  # 64 MB
BLOCK_RESTART_INTERVAL="${BLOCK_RESTART_INTERVAL:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-32768}"  # 32 KB
BLOOM_BITS="${BLOOM_BITS:-10}"
LEVEL0_FILE_NUM_COMPACTION_TRIGGER="${LEVEL0_FILE_NUM_COMPACTION_TRIGGER:-4}"
LEVEL0_SLOWDOWN_WRITES_TRIGGER="${LEVEL0_SLOWDOWN_WRITES_TRIGGER:-8}"
LEVEL0_STOP_WRITES_TRIGGER="${LEVEL0_STOP_WRITES_TRIGGER:-12}"

# LSM tree config settings
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
MAX_BACKGROUND_FLUSHES="${MAX_BACKGROUND_FLUSHES:-2}"
MAX_WRITE_BUFFER_NUMBER="${MAX_WRITE_BUFFER_NUMBER:-2}"
DISABLE_WAL="${DISABLE_WAL:-true}"
DISABLE_AUTO_COMPACTIONS="${DISABLE_AUTO_COMPACTIONS:-0}"
case "${DISABLE_AUTO_COMPACTIONS}" in
  0|1)
    ;;
  *)
    echo "error: DISABLE_AUTO_COMPACTIONS must be 0 or 1 (got: ${DISABLE_AUTO_COMPACTIONS})" >&2
    exit 1
    ;;
esac

# Workload settings
THREADS="${THREADS:-1}"
KEY_SIZE="${KEY_SIZE:-16}"
VALUE_SIZES="${VALUE_SIZES:-32 64 128 256}"
SEED="${SEED:-23}"

# OS settings
DIRECT_IO="${DIRECT_IO:-true}"
if [[ "${DIRECT_IO}" != "true" && "${DIRECT_IO}" != "false" ]]; then
  echo "error: DIRECT_IO must be 'true' or 'false' (got: ${DIRECT_IO})" >&2
  exit 1
fi
OPEN_FILES="${OPEN_FILES:--1}"

# CPU multithreading experimentation knobs
SUBCOMP_THREADS_LIST="${SUBCOMP_THREADS_LIST:-1 2 4 8}"
BG_COMP_THREADS_LIST="${BG_COMP_THREADS_LIST:-}"

# Key distribution (uniform = default, zipfian = YCSB-like)
KEY_DIST="${KEY_DIST:-uniform}"
ZIPF_ALPHA="${ZIPF_ALPHA:-0.99}"

# Block cache size (default 128 MB; db_bench built-in default is 32 MB)
CACHE_SIZE="${CACHE_SIZE:-134217728}"

# Metrics collection (0 = disabled, value in milliseconds)
METRICS_INTERVAL_MS="${METRICS_INTERVAL_MS:-0}"
THREAD_STATUS_PER_INTERVAL="${THREAD_STATUS_PER_INTERVAL:-0}"
REPORT_BG_IO_STATS="${REPORT_BG_IO_STATS:-0}"
PERF_LEVEL="${PERF_LEVEL:-1}"
HOST_METRICS_INTERVAL_SEC="${HOST_METRICS_INTERVAL_SEC:-0}"
HOST_METRICS_DEVICE="${HOST_METRICS_DEVICE:-}"
RUN_PAUSE_SECONDS="${RUN_PAUSE_SECONDS:-0}"
RUN_QUIET="${RUN_QUIET:-0}"

LSM_TREE_ADDITIONAL_FLAGS=(
  --compression_type="${COMPRESSION_TYPE}"
  --max_background_flushes="${MAX_BACKGROUND_FLUSHES}"
  --max_write_buffer_number="${MAX_WRITE_BUFFER_NUMBER}"
  --disable_wal="${DISABLE_WAL}"
  --cache_size="${CACHE_SIZE}"
)

LSM_TREE_PAPER_FLAGS=(
  --write_buffer_size="${WRITE_BUFFER_SIZE}"
  --target_file_size_base="${TARGET_FILE_SIZE_BASE}"
  --max_bytes_for_level_base="${MAX_BYTES_FOR_LEVEL_BASE}"
  --block_restart_interval="${BLOCK_RESTART_INTERVAL}"
  --block_size="${BLOCK_SIZE}"
  --bloom_bits="${BLOOM_BITS}"
  --level0_file_num_compaction_trigger="${LEVEL0_FILE_NUM_COMPACTION_TRIGGER}"
  --level0_slowdown_writes_trigger="${LEVEL0_SLOWDOWN_WRITES_TRIGGER}"
  --level0_stop_writes_trigger="${LEVEL0_STOP_WRITES_TRIGGER}"
)

WORKLOAD_FLAGS=(
  --key_size="${KEY_SIZE}"
  --threads="${THREADS}"
  --seed="${SEED}"
  --key_dist="${KEY_DIST}"
  --zipf_alpha="${ZIPF_ALPHA}"
)

OS_FLAGS=(
  --use_direct_io_for_flush_and_compaction="${DIRECT_IO}"
  --use_direct_reads="${DIRECT_IO}"
  --open_files="${OPEN_FILES}"
)

STATISTICS_FLAGS=(
  --statistics
  --stats_interval_seconds=60
  --stats_per_interval=1
  --stats_dump_period_sec=60
  --report_interval_seconds=1
  --metrics_interval_ms="${METRICS_INTERVAL_MS}"
  --report_bg_io_stats="${REPORT_BG_IO_STATS}"
  --perf_level="${PERF_LEVEL}"
)

if [[ "${THREAD_STATUS_PER_INTERVAL}" != "0" ]]; then
  STATISTICS_FLAGS+=(--thread_status_per_interval="${THREAD_STATUS_PER_INTERVAL}")
fi

COMMON_FLAGS=(
  "${LSM_TREE_ADDITIONAL_FLAGS[@]}"
  "${LSM_TREE_PAPER_FLAGS[@]}"
  "${WORKLOAD_FLAGS[@]}"
  "${OS_FLAGS[@]}"
  "${STATISTICS_FLAGS[@]}"
)

RUN_METADATA_KEYS=(
  # Required paths
  DB_BENCH
  DB_BASE_DIR
  WAL_BASE_DIR
  OUTPUT_DIR

  # Run metadata
  RUN_ID

  # Workload settings
  THREADS
  KEY_SIZE
  VALUE_SIZES
  SEED

  # CPU multithreading experimentation
  SUBCOMP_THREADS_LIST
  BG_COMP_THREADS_LIST

  # LSM tree config settings
  COMPRESSION_TYPE
  MAX_BACKGROUND_FLUSHES
  MAX_WRITE_BUFFER_NUMBER
  DISABLE_WAL
  DISABLE_AUTO_COMPACTIONS

  # OS settings
  DIRECT_IO
  OPEN_FILES

  # LSM tree config settings (GP-Comp paper-specific defaults).
  WRITE_BUFFER_SIZE
  TARGET_FILE_SIZE_BASE
  MAX_BYTES_FOR_LEVEL_BASE
  BLOCK_RESTART_INTERVAL
  BLOCK_SIZE
  BLOOM_BITS
  LEVEL0_FILE_NUM_COMPACTION_TRIGGER
  LEVEL0_SLOWDOWN_WRITES_TRIGGER
  LEVEL0_STOP_WRITES_TRIGGER

  # Key distribution
  KEY_DIST
  ZIPF_ALPHA

  # Block cache
  CACHE_SIZE

  # Workload-specific settings
  NUM_KEYS
  NUM_LOADS
  WRITES
  WRITES_BY_VALUE_SIZE
  WRITE_RATIOS
  MIX_READS
  LOAD_BENCH
  MIX_BENCH
  PRELOAD_DIR_NAME
  COMPACT_PRELOAD_ON_CREATE

  # Metrics / profiling
  METRICS_INTERVAL_MS
  THREAD_STATUS_PER_INTERVAL
  REPORT_BG_IO_STATS
  PERF_LEVEL
  HOST_METRICS_INTERVAL_SEC
  HOST_METRICS_DEVICE
  RUN_PAUSE_SECONDS
  RUN_QUIET
)

write_run_config() {
  local run_dir="$1"
  local script_name="${2:-unknown}"
  local metadata_dir="${run_dir}/metadata"
  local config_file="${metadata_dir}/run_config.env"

  mkdir -p "${metadata_dir}"

  {
    echo "# Auto-generated benchmark run config"
    printf "SCRIPT_NAME=%q\n" "${script_name}"
    for key in "${RUN_METADATA_KEYS[@]}"; do
      if [[ -n "${!key+x}" ]]; then
        printf "%s=%q\n" "${key}" "${!key}"
      fi
    done
  } > "${config_file}"
}

copy_latest_rocksdb_options() {
  local db_dir="$1"
  local run_dir="$2"
  local label="$3"
  local metadata_dir="${run_dir}/metadata"
  local latest_options

  mkdir -p "${metadata_dir}"
  latest_options="$(ls -1t "${db_dir}"/OPTIONS-* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_options}" ]]; then
    cp "${latest_options}" "${metadata_dir}/rocksdb_options_${label}.ini"
  fi
}

copy_rocksdb_log_file() {
  local db_dir="$1"
  local run_dir="$2"
  local label="$3"
  local metadata_dir="${run_dir}/metadata"
  local log_file="${db_dir}/LOG"

  mkdir -p "${metadata_dir}"
  if [[ -f "${log_file}" ]]; then
    cp "${log_file}" "${metadata_dir}/rocksdb_LOG_${label}.txt"
  fi
}

resolve_block_device_for_path() {
  local target_path="$1"
  local source_device
  local physical_device

  if [[ -n "${HOST_METRICS_DEVICE}" ]]; then
    printf "%s\n" "${HOST_METRICS_DEVICE}"
    return 0
  fi

  source_device="$(df --output=source "${target_path}" 2>/dev/null | tail -n 1 | tr -d ' ')"
  if [[ -z "${source_device}" ]]; then
    return 1
  fi

  if [[ "${source_device}" == /dev/* ]] && command -v lsblk >/dev/null 2>&1; then
    physical_device="$(lsblk -no pkname "${source_device}" 2>/dev/null | head -n 1 | tr -d ' ')"
    if [[ -n "${physical_device}" ]]; then
      printf "%s\n" "${physical_device}"
      return 0
    fi
  fi

  printf "%s\n" "$(basename "${source_device}")"
}

write_kv_metadata() {
  local target_file="$1"
  shift

  {
    echo "# Auto-generated metadata"
    while (( $# > 0 )); do
      local key="$1"
      local value="$2"
      printf "%s=%q\n" "${key}" "${value}"
      shift 2
    done
  } > "${target_file}"
}

start_host_metrics_collection() {
  local bench_pid="$1"
  local run_dir="$2"
  local metadata_dir="${run_dir}/metadata"
  local host_metrics_dir="${run_dir}/host_metrics"
  local collector_script="${BENCH_ROOT_DIR}/python/collect_host_metrics.py"
  local device
  local collector_pid_file="${metadata_dir}/host_metrics.pid"

  if [[ "${HOST_METRICS_INTERVAL_SEC}" == "0" || "${HOST_METRICS_INTERVAL_SEC}" == "0.0" ]]; then
    return 0
  fi

  if [[ ! -f "${collector_script}" ]]; then
    echo "warning: host metrics collector missing: ${collector_script}" >&2
    return 0
  fi

  device="$(resolve_block_device_for_path "${DB_BASE_DIR}" || true)"
  if [[ -z "${device}" ]]; then
    echo "warning: unable to resolve block device for ${DB_BASE_DIR}; skipping host metrics" >&2
    return 0
  fi

  mkdir -p "${metadata_dir}" "${host_metrics_dir}"
  write_kv_metadata "${metadata_dir}/host_metrics.env" \
    HOST_METRICS_PID "${bench_pid}" \
    HOST_METRICS_DEVICE "${device}" \
    HOST_METRICS_INTERVAL_SEC "${HOST_METRICS_INTERVAL_SEC}" \
    HOST_METRICS_DIR "${host_metrics_dir}"

  python3 "${collector_script}" \
    --pid "${bench_pid}" \
    --device "${device}" \
    --interval-sec "${HOST_METRICS_INTERVAL_SEC}" \
    --output-dir "${host_metrics_dir}" &
  local collector_pid=$!
  printf "%s\n" "${collector_pid}" > "${collector_pid_file}"
  HOST_METRICS_COLLECTOR_PID="${collector_pid}"
}

stop_host_metrics_collection() {
  local collector_pid="${HOST_METRICS_COLLECTOR_PID:-}"

  if [[ -z "${collector_pid}" ]]; then
    return 0
  fi

  if kill -0 "${collector_pid}" 2>/dev/null; then
    kill -TERM "${collector_pid}" 2>/dev/null || true
    wait "${collector_pid}" || true
  fi
  unset HOST_METRICS_COLLECTOR_PID
}

run_db_bench() {
  local log_file="$1"
  local run_dir
  local metadata_dir
  local cmd_file
  shift

  run_dir="$(dirname "${log_file}")"
  metadata_dir="${run_dir}/metadata"
  mkdir -p "${metadata_dir}"

  cmd_file="${metadata_dir}/$(basename "${log_file%.log}").cmd"
  {
    echo "# Auto-generated db_bench command"
    printf "timestamp=%q\n" "$(date +%Y-%m-%dT%H:%M:%S%z)"
    printf "command="
    printf "%q " "${DB_BENCH}" "$@"
    echo
  } > "${cmd_file}"

  if [[ "${RUN_QUIET}" == "1" ]]; then
    "${DB_BENCH}" "$@" > "${log_file}" 2>&1 &
  else
    "${DB_BENCH}" "$@" > >(tee "${log_file}") 2>&1 &
  fi
  local bench_pid=$!
  local bench_rc=0
  start_host_metrics_collection "${bench_pid}" "${run_dir}"

  if ! wait "${bench_pid}"; then
    bench_rc=$?
  fi

  stop_host_metrics_collection
  return "${bench_rc}"
}

safe_remove_dir() {
  local target_dir="$1"
  local base_dir="$2"

  if [[ -z "${target_dir}" || "${target_dir}" == "/" ]]; then
    echo "error: refusing to remove unsafe path: ${target_dir}" >&2
    exit 1
  fi

  if [[ -z "${base_dir}" || "${base_dir}" == "/" ]]; then
    echo "error: refusing to use unsafe base dir: ${base_dir}" >&2
    exit 1
  fi

  case "${target_dir}" in
    "${base_dir}"/*)
      rm -rf -- "${target_dir}"
      ;;
    *)
      echo "error: refusing to remove ${target_dir}; outside base dir ${base_dir}" >&2
      exit 1
      ;;
  esac
}

cleanup_db_wal_dirs() {
  local db_dir="$1"
  local wal_dir="$2"
  safe_remove_dir "${db_dir}" "${DB_BASE_DIR}"
  safe_remove_dir "${wal_dir}" "${WAL_BASE_DIR}"
}
