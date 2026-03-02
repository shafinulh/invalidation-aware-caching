#!/usr/bin/env bash
set -euo pipefail

### Context ###
# GPU vs CPU compaction IO simulation — BOTH 8 MB and 64 MB L0 sizes
# in a single run for easy comparison.
#
# Compaction IO pattern per size:
#   Read  4 L0 files  →  Write 3 L1 files  (same size)
#
# This is the "full sweep" experiment that generates a complete
# comparison dataset for the IO overhead analysis.

### Run ###

RUN_ID=io-sweep \
L0_SIZES="8388608 67108864" \
NUM_L0_READ=4 \
NUM_L1_WRITE=3 \
ALIGNMENT=4096 \
NUM_REPS=10 \
DIRECT_IO=true \
./benchmarks/gpu/scripts/run_io_bench.sh
