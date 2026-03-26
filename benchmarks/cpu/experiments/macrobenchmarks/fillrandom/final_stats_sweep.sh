#!/usr/bin/env bash
set -euo pipefail

### Final monitored FillRandom sweep ###
# Goals:
# - Sweep value sizes 32 / 64 / 128 / 256
# - Sweep subcompactions 1 / 2 / 4 / 8 / 16
# - Include a no-auto-compaction baseline via subcomp_threads=0
# - Collect host IO / CPU / per-thread CPU samples
# - Pause between runs to reduce thermal carry-over

RUN_ID=bounded-fr-l0-monitored-final \
NUM_KEYS=200000000 \
WRITES=-1 \
WRITES_BY_VALUE_SIZE="32:20000000 64:12000000 128:6500000 256:3500000" \
THREADS=10 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES="32 64 128 256" \
SUBCOMP_THREADS_LIST="0 1 2 4 8 16" \
BG_COMP_THREADS_LIST="1" \
DISABLE_AUTO_COMPACTIONS=0 \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=8 \
LEVEL0_STOP_WRITES_TRIGGER=12 \
MAX_BACKGROUND_FLUSHES=4 \
HOST_METRICS_INTERVAL_SEC=1 \
REPORT_BG_IO_STATS=1 \
RUN_PAUSE_SECONDS=60 \
./benchmarks/cpu/scripts/run_fillrandom.sh
