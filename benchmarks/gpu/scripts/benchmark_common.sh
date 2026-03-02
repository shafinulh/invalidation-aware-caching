#!/usr/bin/env bash
set -euo pipefail

# Shared benchmark environment setup for GPU IO benchmarks.
# Source required machine-local config from benchmarks/gpu/config/.env.local.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_ENV_FILE="${BENCH_ROOT_DIR}/config/.env.local"

if [[ ! -f "${BENCH_ENV_FILE}" ]]; then
  echo "error: missing ${BENCH_ENV_FILE}" >&2
  echo "hint: copy ${BENCH_ROOT_DIR}/config/.env.example to ${BENCH_ENV_FILE} and edit it." >&2
  exit 1
fi

CONFIG_KEYS=(
  DATA_DIR
  OUTPUT_DIR
  GPU_DEVICE
  COMPAT_MODE
  L0_SIZES
  NUM_L0_READ
  NUM_L1_WRITE
  ALIGNMENT
  NUM_REPS
  DIRECT_IO
)

# Preserve caller-provided env so inline overrides win over .env.local
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

require_env DATA_DIR
require_env OUTPUT_DIR

# Defaults
GPU_DEVICE="${GPU_DEVICE:-0}"
COMPAT_MODE="${COMPAT_MODE:-true}"
L0_SIZES="${L0_SIZES:-8388608 67108864}"
NUM_L0_READ="${NUM_L0_READ:-4}"
NUM_L1_WRITE="${NUM_L1_WRITE:-3}"
ALIGNMENT="${ALIGNMENT:-4096}"
NUM_REPS="${NUM_REPS:-10}"
DIRECT_IO="${DIRECT_IO:-true}"

# Run metadata
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M)}"

# Path to the benchmark binary (built by Makefile).
GPU_IO_BENCH="${BENCH_ROOT_DIR}/gpu_io_bench"

if [[ ! -x "${GPU_IO_BENCH}" ]]; then
  echo "error: gpu_io_bench binary not found at ${GPU_IO_BENCH}" >&2
  echo "hint: run 'make' in ${BENCH_ROOT_DIR} first." >&2
  exit 1
fi

# ── helpers ──

write_run_config() {
  local run_dir="$1"
  local script_name="${2:-unknown}"
  local metadata_dir="${run_dir}/metadata"
  local config_file="${metadata_dir}/run_config.env"

  mkdir -p "${metadata_dir}"

  {
    echo "# Auto-generated GPU IO benchmark run config"
    printf "SCRIPT_NAME=%q\n" "${script_name}"
    printf "RUN_ID=%q\n" "${RUN_ID}"
    printf "TIMESTAMP=%q\n" "$(date +%Y-%m-%dT%H:%M:%S%z)"
    for key in "${CONFIG_KEYS[@]}"; do
      if [[ -n "${!key+x}" ]]; then
        printf "%s=%q\n" "${key}" "${!key}"
      fi
    done
  } > "${config_file}"
}

drop_caches_if_root() {
  if [[ $EUID -eq 0 ]]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches
    return 0
  fi
  if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    return 0
  fi
  echo "warning: cannot drop page cache (not root). Results may include cached reads." >&2
  return 1
}
