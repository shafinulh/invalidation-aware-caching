#!/usr/bin/env bash
set -euo pipefail

### Context ###
# FPGA demonstrates better throughput with faster compactions
# (multithreading or GPU) by reducing the build up of L0 files.
# I do not believe the FPGA paper has any limits on the build up of L0 files. 
# L0 files build up impacts read performance due to increased read amplification.

# Write stalls never occur thus we do not expect any significant improvements 
# in write performance with multithreaded compactions.

# If we consider running multiple parallel compactions (as is done in the FPGA paper)
# then FillRandom performance should change due to CPU contention, since more time is 
# spent on background compactions.

### GOAL ###
# This benchmark serves to demonstrate that with more BG compaction threads, we see more
# CPU contention and thus worse write performance. 

### Setup ###
# 10 threads to saturate i5-12900k cores with foreground writes 
# and have fastest theoretical write throughput

# 200M Key Space, 40M writes per thread = 400M total writes

RUN_ID=fpga-cpu-contention-fr_l0-16mb_ \
NUM_KEYS=200000000 \
WRITES=40000000 \
THREADS=10 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 4 8 16 32" \
BG_COMP_THREADS_LIST="16" \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=1000000 \
LEVEL0_STOP_WRITES_TRIGGER=1000000 \
WRITE_BUFFER_SIZE=16777216 \
TARGET_FILE_SIZE_BASE=16777216 \
MAX_BYTES_FOR_LEVEL_BASE=100663296 \
MAX_BACKGROUND_FLUSHES=4 \
MAX_WRITE_BUFFER_NUMBER=100 \
./benchmarks/cpu/scripts/run_fillrandom.sh
