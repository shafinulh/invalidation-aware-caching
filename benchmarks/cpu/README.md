# CPU Benchmarks

## 1) Build `db_bench`

### macOS
```bash
cd /path/to/invalidation-aware-caching/rocksdb-gpu

rm -f make_config.mk

CPATH="$(brew --prefix)/include" \
LIBRARY_PATH="$(brew --prefix)/lib" \
make -B -j 8 DEBUG_LEVEL=0 DISABLE_WARNING_AS_ERROR=1 db_bench
```

### Ubuntu/Linux
```bash
sudo apt update && sudo apt install -y libgflags-dev 

cd /path/to/invalidation-aware-caching/rocksdb-gpu

make clean

make -j"$(nproc)" DEBUG_LEVEL=0 DISABLE_WARNING_AS_ERROR=1 db_bench
```

## 2) Configure machine-local paths

```bash
cd /path/to/invalidation-aware-caching
cp benchmarks/cpu/config/.env.example benchmarks/cpu/config/.env.local
```

Required in `benchmarks/cpu/config/.env.local`:
- `DB_BENCH` (absolute path)
- `DB_BASE_DIR`
- `WAL_BASE_DIR`
- `OUTPUT_DIR`

## 3) Run workloads

```bash
./benchmarks/cpu/scripts/run_fillrandom.sh
./benchmarks/cpu/scripts/run_readwritemix.sh
```

## Results organization

`OUTPUT_DIR` (from `.env.local`) is the root results directory.

For each run:
- benchmark data: `*.log`, `*.csv`
- metadata: `metadata/run_config.env`, `metadata/*.cmd`, `metadata/rocksdb_options_*.ini`

## Plotting

`plot_results.py` reads `OUTPUT_DIR` from `benchmarks/cpu/config/.env.local` and writes plots to:
- `<OUTPUT_DIR>/plots`

Run:
```bash
python3 benchmarks/cpu/python/plot_results.py
```

## Workload knobs

Quoted paper configuration text:
> "The tests for the Fillrandom workload and the mixed readwrite workload in Section IV-D are conducted using DBBench. The configuration for the KV system is as follows. The sizes of both the Memtable and SST file are set to 8 MB, while the L1 level is configured to 64 MB. The parameters related to the data block are: the block_restart_interval is 4, block_size is 32 KB, and the Bloom filter size (bloom_bits) is set to 10. Default settings are applied for the parameters governing the write rate: kL0_CompactionTrigger is set to 4, kL0_SlowdownWritesTrigger to 8, and kL0_StopWritesTrigger to 12."

Common knobs (`scripts/benchmark_common.sh`):

LSM tree config settings (GP-Comp paper-specific defaults):
- `WRITE_BUFFER_SIZE` (default: `8388608`)
- `TARGET_FILE_SIZE_BASE` (default: `8388608`)
- `MAX_BYTES_FOR_LEVEL_BASE` (default: `67108864`)
- `BLOCK_RESTART_INTERVAL` (default: `4`)
- `BLOCK_SIZE` (default: `32768`)
- `BLOOM_BITS` (default: `10`)
- `LEVEL0_FILE_NUM_COMPACTION_TRIGGER` (default: `4`)
- `LEVEL0_SLOWDOWN_WRITES_TRIGGER` (default: `8`)
- `LEVEL0_STOP_WRITES_TRIGGER` (default: `12`)

LSM tree config settings (additional defaults):
- `COMPRESSION_TYPE` (default: `none`)
- `MAX_BACKGROUND_FLUSHES` (default: `2`)

Workload settings:
- `THREADS` (default: `1`)
- `KEY_SIZE` (default: `16`)
- `VALUE_SIZES` (default: `"32 64 128 256"`)
- `SEED` (default: `23`)

OS settings:
- `DIRECT_IO` (default: `true`)
- `OPEN_FILES` (default: `-1`)

CPU multithreading experimentation settings:
- `COMP_THREADS_LIST` (default: `"1 2 4 8"`)
  Used to set num `--subcompactions` and `--max_background_compactions`

Run identifier:
- `RUN_ID` (default: `MMDD_HHMM`)

Statistics/reporting flags (fixed in `scripts/benchmark_common.sh`):
- `--statistics`
- `--stats_interval_seconds=60`
- `--stats_per_interval=1`
- `--stats_dump_period_sec=60`
- `--report_interval_seconds=1`

FillRandom-only settings (`scripts/run_fillrandom.sh`):
- `NUM_KEYS` (default: `20000000`)
- `WRITES` (default: `-1`, meaning `--writes` falls back to `--num`)

ReadWriteMix-only settings (`scripts/run_readwritemix.sh`):
- `WRITE_RATIOS` (default: `"25"`, write percentages)
- `LOAD_BENCH` (default: `filluniquerandomdeterministic`)
  If `LOAD_BENCH` is deterministic (`fillseqdeterministic` or `filluniquerandomdeterministic`),
  the script automatically sets `--disable_auto_compactions=1` during preload.
- `NUM_KEYS` (default: `200000000`, shared keyspace for preload + mixed phase)
- `NUM_LOADS` (default: `50000000`, number of preload writes)
- `MIX_BENCH` (default: `readrandomwriterandom`)
- `MIX_READS` (default: `20000000`, total mixed operations)

ReadWriteMix load/cleanup behavior:
- Preload DB is created once per `value_size` at `DB_BASE_DIR/readwritemix_preload/...` and reused across all `comp_threads`.
- If the preload DB already exists, the load phase is skipped automatically.
- Preload reuse is guarded by a `.preload_ready` marker created only after a successful load.
- Each mixed run uses a copy under `DB_BASE_DIR/readwritemix/...` and `WAL_BASE_DIR/readwritemix/...`.
- Mixed phase forces `--disable_auto_compactions=0`.
- Mixed DB/WAL directories are deleted after each run, and RocksDB `LOG` is copied to run metadata before deletion.
- To force a fresh load, remove the corresponding `readwritemix_preload` directories.
