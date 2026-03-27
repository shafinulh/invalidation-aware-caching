#!/usr/bin/env bash

# Full GPU compaction sweep.
# All value sizes, all modes, multiple runs with warmup.

VALUES="${VALUES:-${COMPACTION_DEFAULT_VALUE_SIZES}}"
MODES="${MODES:-${COMPACTION_DEFAULT_GPU_MODES}}"
RUNS="${RUNS:-${COMPACTION_DEFAULT_RUNS}}"
LABEL="${LABEL:-full-sweep}"
