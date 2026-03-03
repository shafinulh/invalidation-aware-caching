# GPU Benchmarks

Benchmark suite for GPU-related storage experiments. It currently includes:

- **`io_bench`**: GPU IO (cuFile) vs CPU direct IO for an LSM-tree compaction
  IO pattern
- **`rocksdb_hook`**: a RocksDB dummy GPU compaction hook benchmark that
  replays ground-truth SST outputs through a cuFile-backed GPU pass-through
  helper before installing them back into RocksDB

## Motivation

GPU-accelerated compaction (e.g., GPComp) offloads merge/sort to the GPU, but
the data must travel a longer IO path:

```
CPU compaction IO:   SSD ──read()──▸ RAM ──write()──▸ SSD

GPU compaction IO:   SSD ──▸ RAM ──▸ GPU ──▸ RAM ──▸ SSD
                         (cuFileRead)    (cuFileWrite)
```

On GPUs **without** GPUDirect Storage (GDS) — such as GeForce cards — the cuFile
API falls back to a **bounce-buffer** through host memory, adding extra PCIe hops.
This benchmark quantifies exactly how much overhead those extra hops cost.

The key question: **How much faster must the GPU merge/sort be to offset the
additional IO transfer time?**

## Simulated Compaction Pattern

Each benchmark run simulates a single L0→L1 compaction:

| Phase | Files | Size each | Total data |
|-------|-------|-----------|------------|
| Read  | 4 L0  | 8 or 64 MB | 32 or 256 MB |
| Write | 3 L1  | 8 or 64 MB | 24 or 192 MB |

This matches the typical RocksDB compaction trigger of
`level0_file_num_compaction_trigger = 4`, writing back files of the same
`target_file_size_base`.

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (tested: RTX 3070)
- Fast NVMe SSD

### Software
- CUDA Toolkit ≥ 11.4 with cuFile headers & libraries
- `g++-12` or compatible host compiler
- Python 3.8+ with `pandas` (for analysis)
- Optional: `matplotlib` (for plots), `tabulate` (for tables)

## Quick Start

### 1. Build

```bash
cd /path/to/invalidation-aware-caching/benchmarks/gpu
make                      # builds gpu_io_bench
make gpu_file_replay_bench # builds the RocksDB hook replay helper
```

### 2. Configure

```bash
cp config/.env.example config/.env.local
# Edit config/.env.local — set DATA_DIR, OUTPUT_DIR, and any machine-specific GPU settings
```

### 3. Run `io_bench`

```bash
# From repository root:
./benchmarks/gpu/experiments/io_bench/io_sweep.sh    # both 8 MB and 64 MB
./benchmarks/gpu/experiments/io_bench/io_8mb.sh       # just 8 MB
./benchmarks/gpu/experiments/io_bench/io_64mb.sh      # just 64 MB
```

Or run the script directly with overrides:

```bash
L0_SIZES="8388608" NUM_REPS=5 ./benchmarks/gpu/scripts/run_io_bench.sh
```

Or invoke the binary directly:

```bash
./gpu_io_bench --l0_size 8388608 --reps 10 --csv results.csv
```

### 4. Analyse `io_bench`

```bash
python3 benchmarks/gpu/python/plot_io_bench.py <RUN_DIR> --plot
```

### 5. Run `rocksdb_hook`

```bash
# From repository root:
./benchmarks/gpu/experiments/rocksdb_hook/hook_replay.sh
```

This script incrementally rebuilds the CUDA replay helper and the RocksDB
benchmark target, then runs the dummy GPU compaction hook
benchmark and writes per-repetition CSV output.

### 6. Analyse `rocksdb_hook`

```bash
python3 benchmarks/gpu/python/plot_gpu_compaction_hook_bench.py <RUN_DIR> --plot
```

## Output Format

Each run produces CSV files with columns:

| Column | Description |
|--------|-------------|
| `path` | `cpu` or `gpu` |
| `l0_size_bytes` | Size of each L0 file |
| `l0_size_mb` | Size in MB |
| `num_l0_read` | Number of files read |
| `num_l1_write` | Number of files written |
| `rep` | Repetition index |
| `read_us` | Read phase time (microseconds) |
| `write_us` | Write phase time (microseconds) |
| `total_us` | Total IO time (microseconds) |
| `direct_io` | Whether O_DIRECT was used |

## Results Directory Structure

```
bench_results/gpu/
  io_bench/
    <run_id>/
      io_bench_8mb.csv          # raw data (8 MB L0 files)
      io_bench_64mb.csv         # raw data (64 MB L0 files)
      io_bench_8mb.log          # full console output
      io_bench_64mb.log
      io_bench_summary.csv      # mean ± std per path/size
      io_bench_overhead.csv     # GPU/CPU overhead ratios
      io_bench_comparison.png   # bar chart (if --plot)
      io_bench_boxplot.png      # box plot (if --plot)
      metadata/
        run_config.env           # full configuration snapshot
  rocksdb_hook/
    <run_id>/
      gpu_compaction_hook.csv        # per-repetition timing data
      gpu_compaction_hook.log        # full benchmark log
      gpu_compaction_hook_summary.csv
      gpu_compaction_hook_summary.png   # if --plot
      gpu_compaction_hook_per_rep.png   # if --plot
      metadata/
        run_config.env
```

## What to Expect

On an RTX 3070 (no GDS, bounce-buffer fallback):

- **GPU reads** are ~1.5–2× slower than CPU direct reads due to the extra
  `memcpy` from host RAM to GPU device memory.
- **GPU writes** are similarly ~1.2–1.5× slower.
- **Total overhead** is typically 1.3–1.8× depending on file size and SSD speed.

This overhead is the "IO tax" that GPU-accelerated compaction must recoup
through faster merge/sort computation.

## Configuration Reference

See [config/.env.example](config/.env.example) for machine-local settings.
Workload parameters are set in the experiment scripts under
`benchmarks/gpu/experiments/`.

Default experiment parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `L0_SIZES` | `8388608 67108864` | L0 file sizes (bytes, space-separated) |
| `NUM_L0_READ` | `4` | Input files per compaction |
| `NUM_L1_WRITE` | `3` | Output files per compaction |
| `NUM_REPS` | `10` | Repetitions for `io_bench` (the default `rocksdb_hook` wrapper uses `5`) |
| `ALIGNMENT` | `4096` | O_DIRECT alignment |
| `DIRECT_IO` | `true` | Use O_DIRECT for the CPU baseline in `io_bench` |
| `GPU_DEVICE` | `0` | CUDA device ordinal |
