# GPComp CUDA Testbench

GPU-accelerated RocksDB compaction benchmark.  
Measures and compares CPU vs GPU performance for the two compaction algorithms:
- **Algorithm 1 – Merge**: multi-way merge of SST key-value pairs (GPU `merge_kernel`)
- **Algorithm 2 – Bloom filter**: per-block Bloom filter construction (GPU `bloom_filter_kernel` / `bloom_filter_kernel_batched`)

---

## Directory Structure

```
cuda_test/
├── gpcomp_bench.cu          # main benchmark program (Benchmarks 1–6)
├── gpcomp_merge.cuh         # Algorithm 1: merge kernel + launcher
├── gpcomp_bloom.cuh         # Algorithm 2: bloom filter kernels (batched)
├── gpcomp_pack.cuh          # Fig. 7 & 8: Pack/Unpack GPU kernels
├── gpcomp_common.cuh        # shared types (KVPair), CPU helpers
├── gpcomp_cufile.cuh        # GDS / cuFile abstraction layer
├── gpcomp_datagen.cpp       # dataset generator
├── gpcomp_tests.cu          # unit tests (263 tests: merge + bloom + pack/unpack)
├── Makefile
├── test.sh                  # value-size sweep script
├── sync_to_repo.sh          # sync working files + git push to master
├── GPU_ACCELERATION_EXPLAINED.md  # in-depth algorithm + architecture guide
├── dataset/                 # generated dataset (SST .bin files + dataset.meta)
└── results/                 # all sweep outputs (created by test.sh)
    └── run_YYYY-MM-DD_HH-MM-SS/
        ├── result_val32B_keys200M.txt
        ├── result_val64B_keys200M.txt
        ├── result_val128B_keys200M.txt
        └── result_metadata.txt
```

---

## Build

```sh
make gpcomp_bench       # build benchmark binary
make gpcomp_datagen     # build dataset generator
make gpcomp_unit_tests  # build unit tests
make all                # build everything
```

Requires: CUDA 11+, `nvcc`, C++17.

---

## Generate a Dataset

```sh
./gpcomp_datagen --out dataset --num_sst 4 --keys_per_sst 81920
```

This produces `dataset/sst_0000.bin … sst_000N.bin` and `dataset/dataset.meta`.

---

## Run the Benchmark Directly

```sh
./gpcomp_bench [options]
```

| Option | Default | Description |
|---|---|---|
| `--dataset DIR` | `./dataset` | Path to the dataset directory |
| `--block_size BYTES` | `32768` | SST data block size (32 KB matches GP-Comp paper) |
| `--key_size BYTES` | `16` | Key size in bytes |
| `--value_size BYTES` | `64` | **Value size in bytes** — changes keys/block and bloom config |
| `--overhead BYTES` | `20` | Per-entry SST overhead bytes |
| `--restart_interval N` | `16` | Prefix-compression restart interval for Pack/Unpack (matches RocksDB default) |
| `--fpr_samples N` | `10000` | Non-member samples used to measure false positive rate |
| `--runs N` | `5` | Number of timed repetitions per section (reports min/mean/stddev) |
| `--fillrandom_keys N` | `0` | Simulate a fillrandom workload of N total keys. Auto-computes compaction rounds. Accepts K/M/B suffixes (e.g. `10M`, `200M`, `1B`). `0` = skip. |
| `--compaction_rounds N` | `0` | Manually set number of compaction rounds. Overridden by `--fillrandom_keys`. |
| `--gds` | off | Enable Benchmark 4: GDS I/O + Merge via cuFile |
| `--compat-mode` | off | Force cuFile compat mode (recommended on RTX consumer GPUs) |
| `--help` | — | Print usage |

**Examples:**
```sh
./gpcomp_bench --dataset dataset --value_size 128 --runs 10

# simulate 10M key fillrandom workload (auto-computes ~31 compaction rounds)
./gpcomp_bench --dataset dataset --fillrandom_keys 10M

# simulate 200M key fillrandom workload
./gpcomp_bench --dataset dataset --fillrandom_keys 200M

# simulate 1 billion keys
./gpcomp_bench --dataset dataset --fillrandom_keys 1B

# manually set compaction rounds
./gpcomp_bench --dataset dataset --compaction_rounds 50

# use a smaller restart interval for Pack/Unpack (more restart points, larger output)
./gpcomp_bench --dataset dataset --restart_interval 8
```

---

## Run the Value-Size Sweep (`test.sh`)

Runs `gpcomp_bench` once per value size and collects all outputs into a
timestamped results folder.

```sh
bash test.sh [options]
```

| Option | Default | Description |
|---|---|---|
| `--dataset DIR` | `./dataset` | Dataset directory |
| `--values LIST` | `32,64,128` | Comma-separated list of value sizes to sweep (bytes) |
| `--fillrandom_keys LIST` | `0` | Comma-separated total key counts to simulate. Accepts K/M/B suffixes. `0` = skip. |
| `--key_size BYTES` | `16` | Key size in bytes |
| `--overhead BYTES` | `20` | Per-entry SST overhead bytes |
| `--runs N` | `5` | Timed repetitions per section inside each benchmark run |
| `--outdir DIR` | `./results` | Parent directory for all results |
| `--help` / `-h` | — | Print usage |

The sweep is **2D**: every combination of `--values` × `--fillrandom_keys` gets its own result file.

**Examples:**
```sh
# default sweep: 32 / 64 / 128 B values, no fillrandom simulation
bash test.sh --dataset dataset

# sweep value sizes + simulate 10M key fillrandom workload
bash test.sh --dataset dataset --values 32,64,128 --fillrandom_keys 10M

# full 2D sweep: 3 value sizes x 3 key counts
bash test.sh --dataset dataset --values 32,64,128 --fillrandom_keys 1M,10M,200M

# wider value sweep with more repetitions
bash test.sh --dataset dataset --values 32,64,128,256 --runs 10

# quick sanity check (1 run each)
bash test.sh --dataset dataset --runs 1
```

Each invocation creates a timestamped sub-folder:
```
results/run_2026-02-28_17-07-13/
  result_val32B.txt                   # no fillrandom (--fillrandom_keys 0)
  result_val64B_keys1M.txt            # value_size=64B, fillrandom_keys=1M
  result_val64B_keys10M.txt           # value_size=64B, fillrandom_keys=10M
  result_metadata.txt                 # parameters + combined summary table
```

---

## Output Files

### `result_val<N>B.txt`

Full benchmark output for a single value size. Contains:

```
# Run started: 2026-02-28 17:07:13
# Parameters: value_size=64B  key_size=16B  overhead=20B  block_size=32768B  runs=5
# ─────────────────────────────────────────────────────────

BENCHMARK 1 – Merge kernel (Algorithm 1)
  ...
  CPU sort          min=  8.12  mean=  9.05 ± 0.75 ms  (40.3 M keys/s at min)
  GPU kernel-only   min=  0.12  mean=  0.12 ± 0.00 ms  (2689.8 M keys/s at min)
  GPU wall          min=  1.86  mean=  1.92 ± 0.08 ms  (176.6 M keys/s at min)
  Speedup kernel vs CPU sort (min): 66.69×
  Speedup wall   vs CPU+I/O  (min): 6.30×
  Validation: PASS ✓

BENCHMARK 2 – Bloom filter kernel (Algorithm 2)
  ...
  CPU bloom (per-block)                min= 14.99 ms
  GPU kernel-only (no xfer)            min=  2.94 ms
  GPU batched wall (1×H2D+grid+1×D2H) min=  0.90 ms
  Speedup kernel  vs CPU (min): 5.11×
  Speedup batched vs CPU (min): 16.62×
  No false negatives: PASS ✓
  FPR measured: 0.8189%  vs theoretical: 0.8194%

BENCHMARK 3 – fillrandom compaction simulation (Merge + Bloom)
  Compaction model:
    keys/compaction round = 327680  (4 SSTs flushed from MemTable)
    compaction rounds     = 4
    total simulated keys  = 1310720  (1.3 M)

  Per-round timing (best-of-5):
    I/O (disk read, per round)                   53.32 ms
    CPU compute/round  (sort+bloom)              25.01 ms
    GPU wall/round     (H2D+merge+bloom+D2H)      2.90 ms

  Aggregate over 4 compaction rounds:
    CPU total  (I/O + sort + bloom)              313.4 ms   4.18 M keys/s
    GPU total  (I/O + merge + batched bloom)     224.9 ms   5.83 M keys/s
    Speedup (min): 1.39×   Time saved: 88.5 ms

# Run finished: 2026-02-28 17:07:14
```

### `result_metadata.txt`

Summary file for the entire sweep. Contains:
- Sweep start/finish timestamps and total elapsed time
- All parameters used
- List of result files produced
- Combined summary table (all value sizes in one place)

```
═══════════════════════════════════════════════════════════
  GPComp Sweep Metadata
═══════════════════════════════════════════════════════════

  Sweep started : 2026-02-28 17:07:13
  Sweep finished: 2026-02-28 17:07:45
  Elapsed       : 32s

  ── Parameters used ─────────────────────────────────────
  dataset        = dataset
  value_sizes    = 32,64,128 B
  key_size       = 16 B
  overhead       = 20 B
  block_size     = 32768 B  (32 KB, fixed)
  runs           = 5
  bench binary   = ./gpcomp_bench

  ── Summary (best-of-5, min latency) ────────────────────
  value_size    keys/block  CPU total(ms)  GPU batched(ms)  speedup
  ----------    ----------  -------------  ---------------  -------
  32B           481         30.83          2.86             3.10×
  64B           327         26.36          2.81             5.13×
  128B          199         26.81          2.84             4.77×
```

---

## What Each Benchmark Measures

### Benchmark 1 – Merge (with I/O analysis)

In addition to the heap `cudaMemcpy` path, Benchmark 1 now runs a **pinned
+ async H2D variant**: each SST is loaded into a `cudaMallocHost` buffer and
transferred on a dedicated per-SST CUDA stream dispatched concurrently, then
waited with a single `cudaDeviceSynchronize`. This eliminates the driver's
pageable→staging bounce copy and enables true DMA from pinned memory.

```
CPU I/O (disk read):               2.37 ms
GPU wall (heap H2D+k+D2H):         1.92 ms
GPU wall (pinned async H2D+k+D2H): 1.70 ms   (1.13× improvement)
I/O breakdown:  disk 2.37 | H2D 0.72 | kernel 0.12 | D2H 0.86 ms
```

### Benchmark 2 – Bloom Filter

Measures the CPU vs GPU Bloom filter construction for the output SST's data
blocks. The GPU uses a two-phase shared-memory kernel: Phase 1 hashes each key
into a shared `ByteVector`; Phase 2 packs 8 bytes → 1 byte into the output
`BitVector`. The benchmark runs both a **per-block** variant (validation) and a
**batched** variant (one kernel launch for all data blocks — 1× H2D + 1 kernel
+ 1× D2H), and validates false-negative rate = 0.

### Benchmark 3 – fillrandom compaction simulation

Simulates an entire `db_bench fillrandom` workload spanning multiple compaction
rounds. Per-round timings (I/O + CPU compute or GPU wall) from Benchmarks 1 & 2
are scaled by the number of compaction rounds to produce aggregate CPU vs GPU
totals, speedup, and time saved.

### Benchmark 5 – Pack / Unpack

Measures the GPU Pack kernel (Algorithm Fig. 8 — sorted KVPair array → SST data
blocks with prefix-compressed keys and restart points) and Unpack kernel
(Fig. 7 — SST blocks → flat KVPair array). Both CPU and GPU paths are timed;
results are cross-validated in a round-trip test.

| Metric | val=32B | val=64B | val=128B |
|---|---|---|---|
| keys/block | 481 | 327 | 199 |
| Compression ratio | 0.736 | 0.737 | 0.740 |
| GPU Pack kernel speedup | 18× | 17× | 19× |
| GPU Unpack kernel speedup | 25× | 28× | 28× |

### Benchmark 6 – Full Pipeline

Runs the complete L0→L1 compaction pipeline end-to-end and reports:
- Per-stage CPU vs GPU wall breakdown
- I/O analysis table (disk / H2D / kernels / D2H as % of round)
- Pinned H2D improvement for the full pipeline
- Projected throughput with double-buffered pipelined I/O

```
Stage                                    CPU(ms)   GPU wall(ms)
──────────────────────────────────────   -------   ────────────
1. Unpack  (parse input blocks)             2.13         1.57
2. Merge   (sort/merge all keys)            8.88         1.92
3. Bloom   (build per-block filters)       15.44         0.93
4. Pack    (serialise output blocks)        4.01         1.10
TOTAL                                      30.46         5.52      → 5.5× speedup

With disk I/O (2.37 ms cached / up to 8.1 ms cold NFS):
  End-to-end speedup:  3.1–4.3×
  Pipelined projection (overlap disk+GPU): 1.43× round improvement
```

Requires `--fillrandom_keys N` (or `--compaction_rounds N`) to be set.

| Column | Meaning |
|---|---|
| CPU total | `rounds × (I/O + sort + bloom)` |
| GPU total | `rounds × (I/O + merge wall + batched bloom wall)` |
| Speedup | `CPU total / GPU total` |
| Time saved | `CPU total − GPU total` |

If `--fillrandom_keys` or `--compaction_rounds` is provided, Benchmark 3 also
appends a **fillrandom simulation** section that projects the single-round timings
across N compaction rounds, reporting aggregate CPU vs GPU time saved.
