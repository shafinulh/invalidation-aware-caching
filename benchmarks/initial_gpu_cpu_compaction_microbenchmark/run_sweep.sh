#!/usr/bin/env bash
set -euo pipefail

# CPU/GPU compaction comparison entry point.
#
# Usage:
#   ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh --config quick_test
#   ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh --config full_sweep
#   ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh                      # defaults to quick_test
#
# Override SST/input geometry:
#   ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh --config quick_test --sst-size-mb 8 --input-ssts 32
#
# Skip phases:
#   SKIP_CPU=1 ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh --config quick_test
#   SKIP_GPU_BREAKDOWN=1 ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh --config quick_test
#   SKIP_GPU_HOST_METRICS=1 ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh --config quick_test
#   SKIP_GPU_DEVICE_METRICS=1 ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh --config quick_test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARED_DEFAULTS="${BENCH_ROOT}/common/compaction_defaults.sh"
CONFIG_NAME="quick_test"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_NAME="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
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

# Export all config variables so they're visible to the orchestrator subprocess
export RUN_ID SST_SIZE_MB INPUT_SSTS VALUE_SIZES SUBCOMP_THREADS_LIST
export GPU_MODES GPU_RUNS CPU_RUNS RUN_PAUSE_SECONDS

exec bash "${SCRIPT_DIR}/run_comparison.sh" \
  --sst-size-mb "${SST_SIZE_MB}" \
  --input-ssts "${INPUT_SSTS}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
