#!/usr/bin/env bash
set -euo pipefail

# GPU compaction sweep entry point.
#
# Usage:
#   ./benchmarks/gpu_compaction/run_sweep.sh --config quick_test
#   ./benchmarks/gpu_compaction/run_sweep.sh --config full_sweep
#   ./benchmarks/gpu_compaction/run_sweep.sh                      # defaults to quick_test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="quick_test"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_NAME="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

DEFAULTS_FILE="${SCRIPT_DIR}/config/defaults.sh"
CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.sh"

if [[ ! -f "${DEFAULTS_FILE}" ]]; then
  echo "error: missing defaults: ${DEFAULTS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "error: unknown config '${CONFIG_NAME}' (looked for ${CONFIG_FILE})" >&2
  echo "available configs:"
  for f in "${SCRIPT_DIR}"/config/*.sh; do
    [[ "$(basename "$f")" == "defaults.sh" ]] && continue
    echo "  $(basename "${f}" .sh)"
  done
  exit 1
fi

# shellcheck disable=SC1090
source "${DEFAULTS_FILE}"
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

exec bash "${SCRIPT_DIR}/scripts/gpu_compaction_sweep.sh" \
  --values "${VALUES}" \
  --modes "${MODES}" \
  --runs "${RUNS}" \
  --label "${LABEL}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
