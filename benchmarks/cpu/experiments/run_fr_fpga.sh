#!/usr/bin/env bash
set -euo pipefail

### Context ###
# FPGA demonstrates better throughput with faster compactions
# (multithreading or GPU) by reducing the build up of L0 files.
# I do not believe the FPGA paper has any limits on the build up of L0 files. 
# L0 files build up impacts read performance due to increased read amplification.

# Write stalls never occur thus we do not expect any significant improvements 
# in write performance with multithreaded compactions.

# If we consider only multithreading single compactions at a time (not running in parallel)
# then FillRandom performance should not change significantly

### GOAL ###
# This benchmark serves to demonstrate how the build up of L0 files changes with 
# faster compactions.

### Setup ###
# 10 threads to saturate i5-12900k cores with foreground writes 
# and have fastest theoretical write throughput

# 200M Key Space, 40M writes per thread = 400M total writes

RUN_ID=fpga-fr_l0-16mb \
NUM_KEYS=200000000 \
WRITES=40000000 \
THREADS=10 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 4 8 16 32" \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=1000000 \
LEVEL0_STOP_WRITES_TRIGGER=1000000 \
WRITE_BUFFER_SIZE=16777216 \
TARGET_FILE_SIZE_BASE=16777216 \
MAX_BYTES_FOR_LEVEL_BASE=100663296 \
MAX_BACKGROUND_FLUSHES=4 \
MAX_WRITE_BUFFER_NUMBER=4 \
./benchmarks/cpu/scripts/run_fillrandom.sh
