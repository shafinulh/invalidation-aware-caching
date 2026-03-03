#!/usr/bin/env bash
set -euo pipefail

# Shared machine-local benchmark environment setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_ENV_FILE="${BENCH_ROOT_DIR}/config/.env.local"

if [[ ! -f "${BENCH_ENV_FILE}" ]]; then
  echo "error: missing ${BENCH_ENV_FILE}" >&2
  echo "hint: copy ${BENCH_ROOT_DIR}/config/.env.example to ${BENCH_ENV_FILE} and edit it." >&2
  exit 1
fi

BASE_CONFIG_KEYS=(
  DATA_DIR
  OUTPUT_DIR
  GPU_DEVICE
  COMPAT_MODE
)

CONFIG_KEYS=()

load_bench_env() {
  local extra_keys=("$@")
  CONFIG_KEYS=("${BASE_CONFIG_KEYS[@]}" "${extra_keys[@]}")

  for key in "${CONFIG_KEYS[@]}"; do
    local cli_key="__CLI_${key}"
    if [[ -n "${!key+x}" ]]; then
      printf -v "${cli_key}" "%s" "${!key}"
    fi
  done

  set -a
  # shellcheck disable=SC1090
  source "${BENCH_ENV_FILE}"
  set +a

  for key in "${CONFIG_KEYS[@]}"; do
    local cli_key="__CLI_${key}"
    if [[ -n "${!cli_key+x}" ]]; then
      export "${key}=${!cli_key}"
      unset "${cli_key}"
    fi
  done

  require_env DATA_DIR
  require_env OUTPUT_DIR

  GPU_DEVICE="${GPU_DEVICE:-0}"
  COMPAT_MODE="${COMPAT_MODE:-true}"
  RUN_ID="${RUN_ID:-$(date +%m%d_%H%M)}"
}

require_env() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "error: ${var_name} must be set in ${BENCH_ENV_FILE}" >&2
    exit 1
  fi
}

write_run_config() {
  local run_dir="$1"
  local script_name="${2:-unknown}"
  local metadata_dir="${run_dir}/metadata"
  local config_file="${metadata_dir}/run_config.env"

  mkdir -p "${metadata_dir}"

  {
    echo "# Auto-generated GPU benchmark run config"
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
