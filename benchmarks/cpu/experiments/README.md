# Experiment Scripts

Pre-configured experiment scripts that invoke the shared runners in `scripts/`.
All scripts are run from the **repository root** (`invalidation-aware-caching/`).

## Folder layout

```
experiments/
  fillrandom/          # Write-heavy (FillRandom) workloads
    bounded_l0.sh        - GPComp-style bounded LSM (slowdown=8, stop=12)
    unbounded_l0.sh      - FPGA-style unbounded LSM (no write stalls)
    unbounded_l0_cpu_contention.sh  - Parallel BG compactions + CPU contention
  readwrite/           # Mixed read/write workloads
    bounded_l0_mix.sh    - Bounded LSM, readrandomwriterandom
    unbounded_l0_mix.sh  - Unbounded LSM, readrandomwriterandom
    unbounded_l0_readwhilewriting.sh  - Unbounded LSM, readwhilewriting
  cache_observation/   # Block-cache impact during compactions
    bounded_l0_cache_impact.sh   - Bounded LSM, 1s metrics polling
    unbounded_l0_cache_impact.sh - Unbounded LSM, 1s metrics polling
  compaction_profile/  # Read / write / compute breakdown for compactions
    fillrandom_compaction_profile.sh - Actual background compactions during fillrandom
    compaction_io_breakdown.sh       - Forced compactall after loading with compactions disabled
```

## Bounded vs Unbounded L0

| Setting | Bounded (GPComp) | Unbounded (FPGA) |
|---------|-------------------|-------------------|
| `level0_slowdown_writes_trigger` | 8 | 1 000 000 |
| `level0_stop_writes_trigger` | 12 | 1 000 000 |

**Bounded**: write stalls kick in → faster compactions directly improve write throughput.  
**Unbounded**: no write stalls → faster compactions reduce L0 accumulation / read amplification only.

## Running an experiment

```bash
cd /path/to/invalidation-aware-caching
./benchmarks/cpu/experiments/fillrandom/bounded_l0.sh
```

## Cache observation experiments

These experiments enable `METRICS_INTERVAL_MS=100`, which activates the
`MetricsCollectorAgent` in `db_bench`.  Every 100 ms it writes a CSV row with:

- **Block-cache** hit/miss deltas (total + data/index/filter breakdown)
- **Latency** histograms (Get P50/P95/P99, Write P50/P95/P99)
- **Compaction I/O** bytes read/written per interval
- **Write stall** microseconds per interval

After a run, plot the results:

```bash
python3 benchmarks/cpu/python/plot_cache_metrics.py \
    --metrics-csv /path/to/run_dir/metrics.csv

# Or compare multiple runs:
python3 benchmarks/cpu/python/plot_cache_metrics.py \
    --metrics-dir /path/to/bench_results/cpu/readwritemix/value_32 \
    --compare
```

## Compaction profile experiments

These experiments use RocksDB event logging plus background I/O timing to
break compaction time into:

- **Read IO**
- **Write IO**
- **Computation**

The parser is:

```bash
python3 benchmarks/cpu/python/parse_compaction_profile.py <RUN_DIR> --plot
```

It writes:

- `compaction_breakdown.png`
- `compaction_breakdown_summary.txt`

If you want one figure across all `value_*` directories for the same experiment:

```bash
python3 benchmarks/cpu/python/parse_compaction_profile.py \
    /path/to/bench_results/cpu/compaction_profile/value_32/fillrandom_inline/fr-compact-profile \
    --all-value-sizes --plot
```

This discovers sibling runs such as `value_64/.../fr-compact-profile`,
`value_128/.../fr-compact-profile`, and so on, then generates a combined plot.

### `fillrandom_compaction_profile.sh`

This is the more realistic experiment.

- Workload: `fillrandom`
- L0 policy: bounded
- Compactions: happen naturally in the background while writes are running
- Interpretation: shows the compaction cost seen during a write-only workload

Use this when you want to profile the compactions that RocksDB would actually
perform during a normal write-heavy experiment. The database is being filled,
L0 files accumulate, and background compactions are triggered as part of that
ongoing workload.

Run it from the repository root:

```bash
./benchmarks/cpu/experiments/compaction_profile/fillrandom_compaction_profile.sh
```

Then parse one run:

```bash
python3 benchmarks/cpu/python/parse_compaction_profile.py \
    /home/1755_project/bench_results/cpu/compaction_profile/value_32/fillrandom_inline/fr-compact-profile \
    --plot
```

Or parse the whole experiment family across value sizes:

```bash
python3 benchmarks/cpu/python/parse_compaction_profile.py \
    /home/1755_project/bench_results/cpu/compaction_profile/value_32/fillrandom_inline/fr-compact-profile \
    --all-value-sizes --plot
```

### `compaction_io_breakdown.sh`

This is the isolated compaction experiment.

- Workload structure: first load the database with compactions disabled
- L0 state: many files accumulate without being compacted
- Compactions: then trigger a large compact-all style cleanup phase
- Interpretation: measures an unbounded burst of compaction work happening at once

Use this when you want a cleaner breakdown of the compaction phase itself,
separate from the steady-state write workload. It is less representative of
normal fillrandom execution, but useful for studying the cost of a large batch
of pending L0 compactions once the database has already been filled.

Run it from the repository root:

```bash
./benchmarks/cpu/experiments/compaction_profile/compaction_io_breakdown.sh
```

Then parse the resulting run:

```bash
python3 benchmarks/cpu/python/parse_compaction_profile.py \
    /path/to/bench_results/cpu/compaction_profile/value_32/.../compact-profile \
    --plot
```