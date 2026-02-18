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
  # Required paths.
  DB_BENCH
  DB_BASE_DIR
  WAL_BASE_DIR
  OUTPUT_DIR

  # Run metadata.
  RUN_ID

  # Workload knobs.
  THREADS
  KEY_SIZE
  VALUE_SIZES
  SEED

  # CPU multithreading experimentation knob.
  COMP_THREADS_LIST

  # LSM tree config knobs (additional defaults).
  COMPRESSION_TYPE
  MAX_BACKGROUND_FLUSHES

  # OS knobs.
  DIRECT_IO
  OPEN_FILES

  # LSM tree config knobs (GP-Comp paper-specific defaults).
  WRITE_BUFFER_SIZE
  TARGET_FILE_SIZE_BASE
  MAX_BYTES_FOR_LEVEL_BASE
  BLOCK_RESTART_INTERVAL
  BLOCK_SIZE
  BLOOM_BITS
  LEVEL0_FILE_NUM_COMPACTION_TRIGGER
  LEVEL0_SLOWDOWN_WRITES_TRIGGER
  LEVEL0_STOP_WRITES_TRIGGER

  # Workload-specific knobs.
  NUM_KEYS
  NUM_LOADS
  WRITES
  WRITE_RATIOS
  MIX_READS
  LOAD_BENCH
  MIX_BENCH
)

# Preserve caller-provided env so inline overrides used over .env.local.
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

# CPU multithreading experimentation knob.
# Applied in workload-specific scripts via --subcompactions and
# --max_background_compactions so each workload phase can tune it independently
COMP_THREADS_LIST="${COMP_THREADS_LIST:-1 2 4 8}"

LSM_TREE_ADDITIONAL_FLAGS=(
  --compression_type="${COMPRESSION_TYPE}"
  --max_background_flushes="${MAX_BACKGROUND_FLUSHES}"
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
)

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
  COMP_THREADS_LIST

  # LSM tree config settings
  COMPRESSION_TYPE
  MAX_BACKGROUND_FLUSHES

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

  # Workload-specific settings
  NUM_KEYS
  NUM_LOADS
  WRITES
  WRITE_RATIOS
  MIX_READS
  LOAD_BENCH
  MIX_BENCH
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

  "${DB_BENCH}" "$@" 2>&1 | tee "${log_file}"
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
