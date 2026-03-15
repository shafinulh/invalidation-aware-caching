# Experiment Scripts

Pre-configured experiment scripts that invoke the shared runners in `scripts/`.
All scripts are run from the **repository root** (`invalidation-aware-caching/`).

## Folder layout

```
experiments/
  fillrandom/          # Write-heavy (FillRandom) workloads
    bounded_l0.sh        - GPComp-style bounded LSM (slowdown=8, stop=12)
    final_monitored_sweep.sh - Final monitored sweep + no-compaction baseline
    sample_monitored_pair.sh - Two-run monitored sample (32B, 8 vs 16 subcomp)
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
  compaction_parallelism/  # Isolated manual compaction scaling
    sample_compaction_sweep.sh - Smaller validation matrix
    final_compaction_sweep.sh  - Full isolated compaction scaling sweep
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

For the new monitored FillRandom runs:

```bash
./benchmarks/cpu/experiments/fillrandom/sample_monitored_pair.sh
./benchmarks/cpu/experiments/fillrandom/final_monitored_sweep.sh
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

## Host metrics on monitored runs

Monitored runs also write `host_metrics/` under each run directory:

- `device_io.csv`: per-interval device utilization and throughput
- `system_cpu.csv`: whole-system CPU busy/user/system/iowait
- `per_cpu.csv`: per-core CPU utilization
- `process_cpu.csv`: total `db_bench` CPU summed across its threads
- `thread_cpu.csv`: per-thread CPU usage for the target `db_bench` process
- `thread_role_cpu.csv`: per-interval CPU summed by thread role
- `summary.json`: per-run averages and maxima

The thread-role breakdown is approximate but useful in practice:
- `foreground`: `db_bench` worker / main threads
- `rocksdb_compaction`: RocksDB low/bottom priority background threads
- `rocksdb_flush`: RocksDB high priority flush threads

If you also set `THREAD_STATUS_PER_INTERVAL`, `db_bench.log` will contain
RocksDB thread snapshots with operation and stage labels for additional context.

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

## Compaction parallelism experiments

These experiments isolate a manual CPU compaction after a preload phase so you
can study how compaction parallelism scales with:

- input data size
- value size
- SST size
- requested subcompactions

The shared runner is:

```bash
./benchmarks/cpu/scripts/run_compaction_parallelism.sh
```

Default flow:

1. Preload with `fillrandom` and `--disable_auto_compactions=1`.
2. Keep L0 triggers effectively disabled (`1000000`) so background compactions
   do not interfere.
3. Copy the preload into an ephemeral DB per subcompaction setting.
4. Run `compactall,stats` by default with `--report_bg_io_stats=true`.

Why `compactall` is the shipped default:

- `db_bench compact0` is a no-op on a pure L0-only preload because it expects
  an existing lower level to compact into.
- `compactall` works on the preload shape produced by this flow and still lets
  the analyzer recover the actual L0-input compactions from RocksDB's event log.
- That keeps the sweep runnable while preserving the compaction-pattern detail
  needed for later CPU-vs-GPU reasoning.

You can still force `COMPACTION_BENCH=compact0`, but the runner will fail fast
if RocksDB reports that the benchmark was a no-op.

Recommended profiling settings for this study:

- `COMPACTION_PERF_LEVEL=5`
  This is RocksDB `kEnableTimeAndCPUTimeExceptForMutex`, which captures time
  and CPU timing without mutex timing noise.
- `REPORT_BG_IO_STATS=1`
  Needed for the write-IO timing fields in the RocksDB event log.
- `HOST_METRICS_INTERVAL_SEC=0.1`
  Helps capture short compactions that a 1s sampler would miss.
- `METRICS_INTERVAL_MS=100`
  Optional db_bench-side interval metrics for extra context.

Entry points:

```bash
./benchmarks/cpu/experiments/compaction_parallelism/sample_compaction_sweep.sh
./benchmarks/cpu/experiments/compaction_parallelism/final_compaction_sweep.sh
```

Analyze a finished sweep:

```bash
python3 benchmarks/cpu/python/analyze_compaction_parallelism.py \
    /path/to/bench_results/cpu/compaction_parallelism/<RUN_ID>
```

The analyzer writes:

- `analysis/summary_metrics.csv`
- `analysis/best_runs.csv`
- `analysis/gpu_candidate_runs.csv`
- `analysis/compaction_events.csv`
- `analysis/compaction_pattern_counts.csv`
- `analysis/analysis_summary.txt`
- overview heatmaps plus per-slice throughput / breakdown figures
