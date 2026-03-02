#!/usr/bin/env bash
set -euo pipefail

### Context ###
# GPU vs CPU compaction IO simulation with 8 MB L0 files.
# This matches the default GPComp paper configuration:
#   WRITE_BUFFER_SIZE = TARGET_FILE_SIZE_BASE = 8 MB
#
# Compaction IO pattern:
#   Read  4 × 8 MB L0 files  (32 MB total read)
#   Write 3 × 8 MB L1 files  (24 MB total write)
#
# CPU path:  SSD ──read()──▸ RAM ──write()──▸ SSD   (direct IO)
# GPU path:  SSD ──cuFileRead()──▸ GPU ──cuFileWrite()──▸ SSD
#            (bounce-buffer on GeForce: SSD→RAM→GPU→RAM→SSD)
#
# We expect the GPU path to be slower due to extra PCIe hops.
# This quantifies the IO overhead that GPU compaction must overcome
# with faster merge/sort computation to break even.

### Run ###

RUN_ID=io-8mb \
L0_SIZES="8388608" \
NUM_L0_READ=4 \
NUM_L1_WRITE=3 \
ALIGNMENT=4096 \
NUM_REPS=10 \
DIRECT_IO=true \
./benchmarks/gpu/scripts/run_io_bench.sh
