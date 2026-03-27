#!/usr/bin/env bash
set -euo pipefail

# CPU compaction sweep entry point.
#
# Usage:
#   ./benchmarks/cpu_compaction/run_sweep.sh --config quick_test
#   ./benchmarks/cpu_compaction/run_sweep.sh --config full_sweep
#   ./benchmarks/cpu_compaction/run_sweep.sh                      # defaults to quick_test
#
# Sources shared compaction defaults, then a named config, then runs
# the CPU compaction sweep with automatic analysis.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARED_DEFAULTS="${BENCH_ROOT}/common/compaction_defaults.sh"
CONFIG_NAME="quick_test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.sh"

if [[ ! -f "${SHARED_DEFAULTS}" ]]; then
  echo "error: missing shared defaults: ${SHARED_DEFAULTS}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "error: unknown config '${CONFIG_NAME}' (looked for ${CONFIG_FILE})" >&2
  echo "available configs:"
  for f in "${SCRIPT_DIR}"/config/*.sh; do
    echo "  $(basename "${f}" .sh)"
  done
  exit 1
fi

# shellcheck disable=SC1090
source "${SHARED_DEFAULTS}"
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

echo "========================================================="
echo " CPU Compaction Sweep"
echo " config: ${CONFIG_NAME}"
echo " run id: ${RUN_ID}"
echo " value sizes: ${VALUE_SIZES}"
echo " input data MB: ${INPUT_DATA_MB_LIST}"
echo " SST sizes MB: ${SST_SIZE_MB_LIST}"
echo " subcompaction threads: ${SUBCOMP_THREADS_LIST}"
echo " compaction runs: ${COMPACTION_RUNS}"
echo "========================================================="

cd "${BENCH_ROOT}/.."

KEY_SIZE="${KEY_SIZE:-${COMPACTION_DEFAULT_KEY_SIZE}}" \
VALUE_SIZES="${VALUE_SIZES}" \
INPUT_DATA_MB_LIST="${INPUT_DATA_MB_LIST}" \
SST_SIZE_MB_LIST="${SST_SIZE_MB_LIST}" \
SUBCOMP_THREADS_LIST="${SUBCOMP_THREADS_LIST}" \
COMPACTION_RUNS="${COMPACTION_RUNS}" \
RUN_ID="${RUN_ID}" \
THREADS="${THREADS:-${COMPACTION_DEFAULT_THREADS}}" \
PRELOAD_THREADS="${PRELOAD_THREADS:-${COMPACTION_DEFAULT_THREADS}}" \
PRELOAD_MIN_NUM_KEYS="${PRELOAD_MIN_NUM_KEYS:-${COMPACTION_DEFAULT_USER_KEY_SPACE}}" \
COMPACTION_BENCH="${COMPACTION_BENCH:-${COMPACTION_DEFAULT_COMPACTION_BENCH}}" \
COMPACTION_BG_THREADS="${COMPACTION_BG_THREADS:-${COMPACTION_DEFAULT_COMPACTION_BG_THREADS}}" \
COMPACTION_PERF_LEVEL="${COMPACTION_PERF_LEVEL:-${COMPACTION_DEFAULT_COMPACTION_PERF_LEVEL}}" \
OPEN_FILES="${OPEN_FILES:-${COMPACTION_DEFAULT_OPEN_FILES}}" \
DIRECT_IO="${DIRECT_IO:-${COMPACTION_DEFAULT_DIRECT_IO}}" \
METRICS_INTERVAL_MS="${METRICS_INTERVAL_MS:-${COMPACTION_DEFAULT_METRICS_INTERVAL_MS}}" \
HOST_METRICS_INTERVAL_SEC="${HOST_METRICS_INTERVAL_SEC:-${COMPACTION_DEFAULT_HOST_METRICS_INTERVAL_SEC}}" \
REPORT_BG_IO_STATS="${REPORT_BG_IO_STATS:-${COMPACTION_DEFAULT_REPORT_BG_IO_STATS}}" \
RUN_QUIET="${RUN_QUIET:-1}" \
RUN_PAUSE_SECONDS="${RUN_PAUSE_SECONDS:-0}" \
ANALYZE=1 \
./benchmarks/cpu_compaction/scripts/cpu_compaction_sweep.sh
