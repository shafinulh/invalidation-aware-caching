#!/usr/bin/env bash
set -euo pipefail

# Shared benchmark environment setup for GPU IO benchmarks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

load_bench_env \
  L0_SIZES \
  NUM_L0_READ \
  NUM_L1_WRITE \
  ALIGNMENT \
  NUM_REPS \
  DIRECT_IO

# Defaults
L0_SIZES="${L0_SIZES:-8388608 67108864}"
NUM_L0_READ="${NUM_L0_READ:-4}"
NUM_L1_WRITE="${NUM_L1_WRITE:-3}"
ALIGNMENT="${ALIGNMENT:-4096}"
NUM_REPS="${NUM_REPS:-10}"
DIRECT_IO="${DIRECT_IO:-true}"

# Path to the benchmark binary (built by Makefile).
GPU_IO_BENCH="${BENCH_ROOT_DIR}/gpu_io_bench"

if [[ ! -x "${GPU_IO_BENCH}" ]]; then
  echo "error: gpu_io_bench binary not found at ${GPU_IO_BENCH}" >&2
  echo "hint: run 'make' in ${BENCH_ROOT_DIR} first." >&2
  exit 1
fi
