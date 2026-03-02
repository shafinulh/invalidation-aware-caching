#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Isolated compaction profiling: load L0 files with compactions disabled,
# then trigger compactall to measure Read / Write / Computation breakdown.
#
# GPComp paper defaults (8 MB memtable / SSTs).

### GOAL ###
# Produce a compaction IO/CPU breakdown (GPComp Fig. 2 style) for 32B values
# under the default GPComp paper LSM config.

RUN_ID=compact-profile \
VALUE_SIZES=32 \
NUM_KEYS=200000000 \
NUM_LOADS=20000000 \
COMPACT_THREADS=1 \
COMPACT_SUBCOMP=1 \
OPEN_FILES=512 \
DIRECT_IO=true \
./benchmarks/cpu/scripts/run_compaction_profile.sh
