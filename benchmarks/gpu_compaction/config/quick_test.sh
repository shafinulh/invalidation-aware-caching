#!/usr/bin/env bash

# Quick test config for fast GPU compaction end-to-end validation.
# Minimal matrix: single value size, two modes, one run.

VALUES="${VALUES:-32 1024}"
MODES="${MODES:-q_paper_with_plan q_paper_with_plan_streaming_io c_paper_with_plan c_paper_with_plan_streaming_io}"
RUNS="${RUNS:-1}"
LABEL="${LABEL:-quick-test}"
