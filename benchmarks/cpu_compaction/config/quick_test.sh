#!/usr/bin/env bash

# Quick test config for fast end-to-end validation.
# Minimal matrix: single value size, single input size, few thread counts.

RUN_ID="${RUN_ID:-quick-test}"
VALUE_SIZES="${VALUE_SIZES:-32}"
INPUT_DATA_MB_LIST="${INPUT_DATA_MB_LIST:-32}"
SST_SIZE_MB_LIST="${SST_SIZE_MB_LIST:-8}"
SUBCOMP_THREADS_LIST="${SUBCOMP_THREADS_LIST:-1 4 16}"
COMPACTION_RUNS="${COMPACTION_RUNS:-1}"
