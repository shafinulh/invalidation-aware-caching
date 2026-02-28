#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test.sh  -  GPComp value-size / key-count sweep benchmark
#
# Runs gpcomp_bench for every (value_size x fillrandom_keys) combination
# and saves timestamped results in a per-run sub-folder.
#
# Usage:
#   bash test.sh [options]
#
# Options:
#   --dataset          DIR    dataset directory                [default: ./dataset]
#   --runs             N      timed repetitions per section   [default: 5]
#   --key_size         BYTES  fixed key size                  [default: 16]
#   --overhead         BYTES  per-entry SST overhead          [default: 20]
#   --outdir           DIR    parent results directory        [default: ./results]
#   --values           LIST   comma-separated value sizes (B) [default: 32,64,128]
#   --fillrandom_keys  LIST   comma-separated total key counts to simulate.
#                             Accepts K/M/B suffixes (10M, 200M, 1B).
#                             0 = skip.                  [default: 0]
#   --help / -h
#
# Each invocation saves into: <outdir>/run_YYYY-MM-DD_HH-MM-SS/
#   result_val<V>B[_keys<K>].txt  - full benchmark output per combination
#   result_metadata.txt           - parameters + combined summary table
#
# Examples:
#   bash test.sh --dataset dataset
#   bash test.sh --dataset dataset --values 32,64,128 --fillrandom_keys 1M,10M
#   bash test.sh --dataset dataset --runs 10 --fillrandom_keys 10M
#   bash test.sh --dataset dataset --values 32,64,128 --fillrandom_keys 1M,10M,200M
# ---------------------------------------------------------------------------
set -euo pipefail

# -- defaults ----------------------------------------------------------------
DATASET_DIR="dataset"
RUNS=5
KEY_SIZE=16
OVERHEAD=20
OUTDIR="./results"
VALUE_SIZES="32,64,128"
FILLRANDOM_KEYS="0"
BENCH="./gpcomp_bench"

# -- parse args ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)         DATASET_DIR="$2";     shift 2 ;;
        --runs)            RUNS="$2";            shift 2 ;;
        --key_size)        KEY_SIZE="$2";        shift 2 ;;
        --overhead)        OVERHEAD="$2";        shift 2 ;;
        --outdir)          OUTDIR="$2";          shift 2 ;;
        --values)          VALUE_SIZES="$2";     shift 2 ;;
        --fillrandom_keys) FILLRANDOM_KEYS="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | head -30
            exit 0 ;;
        *) echo "error: unknown option '$1'"; exit 1 ;;
    esac
done

# -- sanity checks ------------------------------------------------------------
if [[ ! -x "$BENCH" ]]; then
    echo "error: '$BENCH' not found or not executable. Run 'make gpcomp_bench' first."
    exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "error: dataset directory '$DATASET_DIR' does not exist."
    exit 1
fi

mkdir -p "$OUTDIR"

# -- timestamps + per-run subfolder ------------------------------------------
SWEEP_START=$(date '+%Y-%m-%d %H:%M:%S')
SWEEP_START_EPOCH=$(date +%s)
RUN_SLUG=$(date '+%Y-%m-%d_%H-%M-%S')
RUN_DIR="$OUTDIR/run_${RUN_SLUG}"
mkdir -p "$RUN_DIR"

# -- convert comma lists to arrays -------------------------------------------
IFS=',' read -ra VAL_ARR  <<< "$VALUE_SIZES"
IFS=',' read -ra KEYS_ARR <<< "$FILLRANDOM_KEYS"

# -- helper: parse K/M/B suffixes to plain integer: 10M -> 10000000 ----------
parse_count() {
    local s="$1"
    case "${s: -1}" in
        k|K) echo "$(echo "${s%?} * 1000"         | bc | cut -d. -f1)" ;;
        m|M) echo "$(echo "${s%?} * 1000000"       | bc | cut -d. -f1)" ;;
        b|B) echo "$(echo "${s%?} * 1000000000"    | bc | cut -d. -f1)" ;;
        *)   echo "$s" ;;
    esac
}

# Normalise KEYS_ARR entries to plain integers (so arithmetic comparisons work)
for i in "${!KEYS_ARR[@]}"; do
    KEYS_ARR[$i]=$(parse_count "${KEYS_ARR[$i]}")
done

# -- helper: human-readable key count: 1000000 -> "1M", 500000 -> "500K" ----
human_keys() {
    local n=$1
    if   (( n >= 1000000000 )); then printf '%gB' "$(echo "scale=1; $n/1000000000" | bc)"
    elif (( n >= 1000000    )); then printf '%gM' "$(echo "scale=1; $n/1000000"    | bc)"
    elif (( n >= 1000       )); then printf '%gK' "$(echo "scale=1; $n/1000"       | bc)"
    else printf '%d' "$n"
    fi
}

# -- header -------------------------------------------------------------------
FR_DISPLAY="$FILLRANDOM_KEYS"
[[ "$FILLRANDOM_KEYS" == "0" ]] && FR_DISPLAY="(disabled)"

echo ""
echo "================================================================"
echo "  GPComp Sweep Benchmark"
echo "  Dataset:          $DATASET_DIR"
echo "  Key size:         ${KEY_SIZE} B    Overhead: ${OVERHEAD} B"
echo "  Runs/section:     $RUNS"
echo "  Value sizes:      $VALUE_SIZES B"
echo "  fillrandom keys:  $FR_DISPLAY"
echo "  Output:           $RUN_DIR/"
echo "================================================================"
echo ""

# -- track all log files for the summary -------------------------------------
declare -a ALL_LOGS
declare -a ALL_VALS
declare -a ALL_KEYS

# -- 2D sweep: value_size x fillrandom_keys ----------------------------------
for VAL in "${VAL_ARR[@]}"; do
  for KEYS in "${KEYS_ARR[@]}"; do

    # choose log filename
    if [[ "$KEYS" == "0" ]]; then
        LOG="$RUN_DIR/result_val${VAL}B.txt"
        LABEL="value_size=${VAL}B"
    else
        HK=$(human_keys "$KEYS")
        LOG="$RUN_DIR/result_val${VAL}B_keys${HK}.txt"
        LABEL="value_size=${VAL}B  fillrandom_keys=${HK}"
    fi

    ALL_LOGS+=("$LOG")
    ALL_VALS+=("$VAL")
    ALL_KEYS+=("$KEYS")

    echo "------------------------------------------------------------"
    echo "  Running: $LABEL"
    echo "  Log:     $LOG"
    echo "------------------------------------------------------------"

    RUN_TS=$(date '+%Y-%m-%d %H:%M:%S')

    # write log header
    {
        printf '# Run started:  %s\n' "$RUN_TS"
        printf '# value_size=%sB  key_size=%sB  overhead=%sB  block_size=32768B  runs=%s' \
               "$VAL" "$KEY_SIZE" "$OVERHEAD" "$RUNS"
        [[ "$KEYS" != "0" ]] && printf '  fillrandom_keys=%s' "$KEYS"
        printf '\n# %s\n' "------------------------------------------------------------"
    } > "$LOG"

    # build bench command args
    BENCH_ARGS=(
        --dataset    "$DATASET_DIR"
        --key_size   "$KEY_SIZE"
        --value_size "$VAL"
        --overhead   "$OVERHEAD"
        --runs       "$RUNS"
    )
    [[ "$KEYS" != "0" ]] && BENCH_ARGS+=(--fillrandom_keys "$KEYS")

    "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee -a "$LOG"

    RUN_END=$(date '+%Y-%m-%d %H:%M:%S')
    printf '\n# Run finished: %s\n' "$RUN_END" >> "$LOG"

    echo "  Timestamp: $RUN_TS -> $RUN_END"
    echo ""

  done
done

# -- extract and print summary -----------------------------------------------
echo ""
echo "================================================================"

if [[ "$FILLRANDOM_KEYS" == "0" ]]; then
    echo "  SUMMARY  (best-of-${RUNS}, min latency, single compaction round)"
    echo "================================================================"
    printf "  %-12s  %-10s  %-14s  %-15s  %-18s\n" \
           "value_size" "keys/block" "CPU total(ms)" "GPU batched(ms)" "speedup(1-round)"
    printf "  %-12s  %-10s  %-14s  %-15s  %-18s\n" \
           "----------" "----------" "-------------" "---------------" "----------------"
    for i in "${!ALL_LOGS[@]}"; do
        L="${ALL_LOGS[$i]}"; V="${ALL_VALS[$i]}"
        kpb=$(grep -E "Keys/block:"                         "$L" | head -1 | awk '{print $2}' || echo "?")
        cpu=$(grep -E "CPU total.*I/O"                      "$L" | head -1 | awk '{print $8}' || echo "?")
        gpu=$(grep -E "GPU wall.*batched bloom"              "$L" | head -1 | awk '{print $7}' || echo "?")
        spd=$(grep -E "Speedup.*GPU full-batched.*CPU total" "$L" | head -1 | awk '{print $7}' || echo "?")
        printf "  %-12s  %-10s  %-14s  %-15s  %-18s\n" "${V}B" "$kpb" "$cpu" "$gpu" "$spd"
    done
else
    echo "  SUMMARY  (best-of-${RUNS} single-round + fillrandom aggregate mean)"
    echo "================================================================"
    printf "  %-12s  %-10s  %-10s  %-14s  %-15s  %-18s  %-16s\n" \
           "value_size" "fr_keys" "keys/block" "CPU/round(ms)" "GPU/round(ms)" "speedup(1-round)" "speedup(fr-agg)"
    printf "  %-12s  %-10s  %-10s  %-14s  %-15s  %-18s  %-16s\n" \
           "----------" "-------" "----------" "-------------" "-------------" "----------------" "---------------"
    for i in "${!ALL_LOGS[@]}"; do
        L="${ALL_LOGS[$i]}"; V="${ALL_VALS[$i]}"; K="${ALL_KEYS[$i]}"
        kpb=$(grep -E "Keys/block:"                          "$L" | head -1 | awk '{print $2}'  || echo "?")
        cpu=$(grep -E "CPU total.*I/O"                       "$L" | head -1 | awk '{print $8}'  || echo "?")
        gpu=$(grep -E "GPU wall.*batched bloom"               "$L" | head -1 | awk '{print $7}'  || echo "?")
        spd=$(grep -E "Speedup.*GPU full-batched.*CPU total"  "$L" | head -1 | awk '{print $7}'  || echo "?")
        frspd=$(grep -E "Speedup .mean.:"                     "$L" | head -1 | awk '{print $3}'  || echo "?")
        [[ "$K" == "0" ]] && frspd="N/A"
        HK="$K"; [[ "$K" != "0" ]] && HK=$(human_keys "$K")
        printf "  %-12s  %-10s  %-10s  %-14s  %-15s  %-18s  %-16s\n" \
               "${V}B" "$HK" "$kpb" "$cpu" "$gpu" "$spd" "$frspd"
    done
fi

# -- write metadata file -----------------------------------------------------
SWEEP_END=$(date '+%Y-%m-%d %H:%M:%S')
SWEEP_END_EPOCH=$(date +%s)
ELAPSED=$(( SWEEP_END_EPOCH - SWEEP_START_EPOCH ))

META="$RUN_DIR/result_metadata.txt"
{
    echo "==========================================================="
    echo "  GPComp Sweep Metadata"
    echo "==========================================================="
    echo ""
    echo "  Sweep started : $SWEEP_START"
    echo "  Sweep finished: $SWEEP_END"
    echo "  Elapsed       : ${ELAPSED}s"
    echo ""
    echo "  -- Parameters used --"
    echo "  dataset         = $DATASET_DIR"
    echo "  value_sizes     = $VALUE_SIZES B"
    echo "  fillrandom_keys = $FILLRANDOM_KEYS"
    echo "  key_size        = ${KEY_SIZE} B"
    echo "  overhead        = ${OVERHEAD} B"
    echo "  block_size      = 32768 B  (32 KB, fixed)"
    echo "  runs            = $RUNS  (timed repetitions per section)"
    echo "  bench binary    = $BENCH"
    echo ""
    echo "  -- Result files --"
    for L in "${ALL_LOGS[@]}"; do
        echo "  $(basename "$L")"
    done
    echo ""
} | tee "$META"

echo "  Full logs + metadata saved in: $RUN_DIR/"
echo ""
