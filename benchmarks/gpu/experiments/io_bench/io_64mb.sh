#!/usr/bin/env bash
set -euo pipefail

### Context ###
# GPU vs CPU compaction IO simulation with 64 MB L0 files.
# This matches a "large memtable" configuration:
#   WRITE_BUFFER_SIZE = TARGET_FILE_SIZE_BASE = 64 MB
#
# Compaction IO pattern:
#   Read  4 × 64 MB L0 files  (256 MB total read)
#   Write 3 × 64 MB L1 files  (192 MB total write)
#
# CPU path:  SSD ──read()──▸ RAM ──write()──▸ SSD   (direct IO)
# GPU path:  SSD ──cuFileRead()──▸ GPU ──cuFileWrite()──▸ SSD
#            (bounce-buffer on GeForce: SSD→RAM→GPU→RAM→SSD)
#
# Larger files amplify the IO overhead difference.  The GPU path
# must transfer 2× the data (SSD↔RAM + RAM↔GPU) compared to CPU.

### Run ###

RUN_ID=io-64mb \
L0_SIZES="67108864" \
NUM_L0_READ=4 \
NUM_L1_WRITE=3 \
ALIGNMENT=4096 \
NUM_REPS=10 \
DIRECT_IO=true \
./benchmarks/gpu/scripts/run_io_bench.sh
