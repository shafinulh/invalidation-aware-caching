# GPComp CUDA Testbench

GPU-accelerated RocksDB compaction benchmark.  
Measures and compares CPU vs GPU performance for the two compaction algorithms:
- **Algorithm 1 – Merge**: multi-way merge of SST key-value pairs (GPU `merge_kernel`)
- **Algorithm 2 – Bloom filter**: per-block Bloom filter construction (GPU `bloom_filter_kernel` / `bloom_filter_kernel_batched`)

---

## Directory Structure

```
cuda_test/
├── gpcomp_bench.cu          # main benchmark program
├── gpcomp_merge.cuh         # merge kernel + launcher
├── gpcomp_bloom.cuh         # bloom filter kernels (serial + batched)
├── gpcomp_common.cuh        # shared types (KVPair), CPU helpers
├── gpcomp_datagen.cpp       # dataset generator
├── gpcomp_tests.cu          # unit tests
├── Makefile
├── test.sh                  # value-size sweep script (this doc)
├── dataset/                 # generated dataset (SST .bin files + dataset.meta)
└── results/                 # all sweep outputs (created by test.sh)
    └── run_YYYY-MM-DD_HH-MM-SS/
        ├── result_val32B.txt
        ├── result_val64B.txt
        ├── result_val128B.txt
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
| `--fpr_samples N` | `10000` | Non-member samples used to measure false positive rate |
| `--runs N` | `5` | Number of timed repetitions per section (reports min/mean/stddev) |
| `--fillrandom_keys N` | `0` | Simulate a fillrandom workload of N total keys. Auto-computes number of compaction rounds. Accepts K/M/B suffixes (e.g. `10M`, `200M`, `1B`). `0` = skip. |
| `--compaction_rounds N` | `0` | Manually set number of compaction rounds. Overridden by `--fillrandom_keys`. |
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
  GPU serial wall (per-block)          min= 11.91 ms
  GPU batched wall (1×H2D+grid+1×D2H) min=  0.90 ms
  Speedup batched vs CPU (min): 16.62×
  Speedup batched vs serial (min): 13.20×
  No false negatives: PASS ✓
  FPR measured: 0.8189%  vs theoretical: 0.8194%

BENCHMARK 3 – Total compaction job (Merge + Bloom)
  Path                                      Time (ms)   M keys/s
  CPU compute  (sort + bloom)                  23.12       14.2
  CPU total    (I/O + sort + bloom)            26.69       12.3
  GPU kernel-only  (no transfers)               3.06      107.0
  GPU wall – serial bloom                      13.77       23.8
  GPU wall – batched bloom                      2.76      118.8
  GPU full – serial   (I/O + serial wall)      17.34       18.9
  GPU full – batched  (I/O + batched wall)      6.33       51.7
  Speedup GPU full-batched vs CPU total: 4.22×

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

### Benchmark 1 – Merge

| Label | What it times |
|---|---|
| **CPU sort** | `std::sort` over all merged keys (no I/O) |
| **GPU kernel-only** | CUDA events around `merge_kernel` only — pure compute time, no transfers |
| **GPU wall (H2D+k+D2H)** | Host→Device copy + kernel + Device→Host copy |

The merge kernel assigns one thread per output key and uses binary search
to find each key's source SST, so it scales with key count, not SST count.

### Benchmark 2 – Bloom Filter

| Label | What it times |
|---|---|
| **CPU bloom** | Pure CPU: `cpu_build_byte_vector` + `cpu_pack_bit_vector` for every block |
| **GPU kernel-only** | CUDA events around kernel only; data pre-transferred once before loop |
| **GPU serial wall** | Per-block: H2D → `bloom_filter_kernel<<<1,T>>>` → sync → D2H (1003 round-trips for 64B values) |
| **GPU batched wall** | 1× H2D for all blocks → `bloom_filter_kernel_batched<<<N,T>>>` → 1× D2H |

The batched kernel eliminates the `cudaDeviceSynchronize` overhead between blocks,
which is the dominant cost in serial mode (~11 ms saved for 64B values).

### Benchmark 3 – Total Compaction (+ fillrandom simulation)

Shows 7 end-to-end paths combining both algorithms.
All times use the minimum from the N-repetition run (best hardware performance).

| Path | Includes |
|---|---|
| CPU compute | sort + bloom (no I/O) |
| CPU total | I/O + sort + bloom |
| GPU kernel-only | merge kernel + bloom kernel (no transfers, no I/O) |
| GPU wall – serial bloom | GPU merge wall + GPU serial bloom wall |
| GPU wall – batched bloom | GPU merge wall + GPU batched bloom wall |
| GPU full – serial | I/O + GPU merge wall + GPU serial bloom wall |
| GPU full – batched | I/O + GPU merge wall + GPU batched bloom wall |

**Apples-to-apples speedup** is `GPU full-batched vs CPU total` — both include disk I/O.

If `--fillrandom_keys` or `--compaction_rounds` is provided, Benchmark 3 also
appends a **fillrandom simulation** section that projects the single-round timings
across N compaction rounds, reporting aggregate CPU vs GPU time saved.
