#!/usr/bin/env bash

# Quick test: minimal CPU+GPU comparison for end-to-end validation.

RUN_ID="${RUN_ID:-quick-test}"
SST_SIZE_MB="${SST_SIZE_MB:-8}"
INPUT_SSTS="${INPUT_SSTS:-32}"
VALUE_SIZES="${VALUE_SIZES:-32 1024}"
SUBCOMP_THREADS_LIST="${SUBCOMP_THREADS_LIST:-1 4 16}"
GPU_MODES="${GPU_MODES:-q_paper_with_plan c_paper_with_plan}"
GPU_RUNS="${GPU_RUNS:-2}"
CPU_RUNS="${CPU_RUNS:-2}"
RUN_PAUSE_SECONDS="${RUN_PAUSE_SECONDS:-0}"
