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
