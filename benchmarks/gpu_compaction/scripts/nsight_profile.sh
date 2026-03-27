#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/gpu_sweep_common.sh"

# Usage examples:
#   ./nsight_profile.sh --tool nsys --dataset /abs/path/to/dataset_V32 --gpu_mode q_paper_with_plan --out_dir /abs/path/to/results/nsight
#   ./nsight_profile.sh --tool ncu  --dataset /abs/path/to/dataset_V32 --gpu_mode q_paper_without_plan --out_dir /abs/path/to/results/nsight

TOOL="nsys"
NSYS_BIN="${NSYS_BIN:-}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
DATASET="${DATASET:-${DATASET_ROOT%/}/${DATASET_PREFIX}32}"
GPU_MODE="q_paper_with_plan"
OUT_DIR="${OUT_DIR:-${OUT_ROOT%/}/single_run_nsight}"
RUNS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool) TOOL="$2"; shift 2 ;;
    --cpuctxsw) NSYS_CPUCTXSW="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu_mode) GPU_MODE="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"
STAMP="$(date '+%Y%m%d_%H%M%S')"

resolve_nsys_bin

if [[ "$TOOL" == "nsys" ]]; then
  if ! command -v "$NSYS_BIN" >/dev/null 2>&1; then
    echo "$NSYS_BIN not found in PATH"
    exit 1
  fi
  OUT_PREFIX="$OUT_DIR/nsys_${GPU_MODE}_${STAMP}"
  echo "Running Nsight Systems... binary: $NSYS_BIN"
  echo "Running Nsight Systems... output prefix: $OUT_PREFIX"
  echo "Running Nsight Systems... cpu context switches: $NSYS_CPUCTXSW"
  "$NSYS_BIN" profile \
    -o "$OUT_PREFIX" \
    --trace=cuda,osrt,nvtx \
    --sample=none \
    --cpuctxsw="$NSYS_CPUCTXSW" \
    "$BENCH_BIN" --dataset "$DATASET" --out_dir "$OUT_DIR/gpcomp_out_${STAMP}" --runs "$RUNS" --gpu_mode "$GPU_MODE"
  if [[ -f "${OUT_PREFIX}.nsys-rep" ]]; then
    echo "Nsight Systems report: ${OUT_PREFIX}.nsys-rep"
  elif [[ -f "${OUT_PREFIX}.qdstrm" ]]; then
    echo "Nsight Systems importer not available on this machine."
    echo "Trace generated as: ${OUT_PREFIX}.qdstrm"
    echo "Open/convert on a machine with full Nsight Systems install (GUI importer enabled)."
  else
    echo "Nsight Systems did not emit expected report files for prefix: ${OUT_PREFIX}"
    exit 1
  fi
elif [[ "$TOOL" == "ncu" ]]; then
  if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found in PATH"
    exit 1
  fi
  OUT_PREFIX="$OUT_DIR/ncu_${GPU_MODE}_${STAMP}"
  echo "Running Nsight Compute... output prefix: $OUT_PREFIX"
  ncu \
    --set full \
    --target-processes all \
    --export "$OUT_PREFIX" \
    "$BENCH_BIN" --dataset "$DATASET" --out_dir "$OUT_DIR/gpcomp_out_${STAMP}" --runs "$RUNS" --gpu_mode "$GPU_MODE"
  echo "Nsight Compute report: ${OUT_PREFIX}.ncu-rep"
else
  echo "Unsupported tool: $TOOL (use nsys or ncu)"
  exit 1
fi
