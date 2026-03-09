#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Profile compactions during a normal fillrandom workload.
# Every background compaction emits wall-clock, CPU, and write-IO timing.
#
# GPComp paper defaults (8 MB memtable / SSTs).

### GOAL ###
# Produce a per-compaction IO/CPU breakdown for compactions occurring
# naturally under write pressure with bounded L0.

# Also experiment with larger memtable. Larger memtables result in
# fewer but more expensive compactions.
# WRITE_BUFFER_SIZE=67108864 \
# TARGET_FILE_SIZE_BASE=67108864 \
# MAX_BYTES_FOR_LEVEL_BASE=268435456 \
# LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
# LEVEL0_SLOWDOWN_WRITES_TRIGGER=8 \
# LEVEL0_STOP_WRITES_TRIGGER=12 \

RUN_ID=fr-compact-profile-8mb_l0 \
VALUE_SIZES="32 64 128 256 512 1024" \
NUM_KEYS=200000000 \
WRITES=20000000 \
THREADS=1 \
COMPACT_SUBCOMP=1 \
COMPACT_BG_THREADS=1 \
OPEN_FILES=512 \
DIRECT_IO=true \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=8 \
LEVEL0_STOP_WRITES_TRIGGER=12 \
./benchmarks/cpu/scripts/run_fillrandom_compaction_profile.sh
