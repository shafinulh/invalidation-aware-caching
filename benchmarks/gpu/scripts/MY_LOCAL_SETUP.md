# My Local Library Setup Notes
> Personal reference — machine: ug71

---

## Directory Layout

### gflags (hand-installed, NOT in system path)
```
GFLAGS_DIR=/nfs/ug/groups/ece1755_w26_group1/local/gflags

Headers  : $GFLAGS_DIR/usr/include/gflags/gflags.h
Libs     : $GFLAGS_DIR/usr/lib/x86_64-linux-gnu/
  libgflags.so.2.2.2          ← dynamic (need LD_LIBRARY_PATH)
  libgflags.a                 ← static  (no runtime dep)
  libgflags_nothreads.a
  libgflags_nothreads.so.2.2.2
```

### GDS / cuFile (NOT in system CUDA — /usr/local/cuda has no cufile.h)
```
GDS_LOCAL=/nfs/ug/groups/ece1755_w26_group1/cuda_test/gds/local

Header : $GDS_LOCAL/include/cufile.h
Libs   : $GDS_LOCAL/lib/
  libcufile.so.1.4.0
  libcufile_rdma.so.1.4.0
```

### System CUDA
```
nvcc   : /usr/bin/nvcc  (CUDA 11.8)
CUDA_HOME usually at /usr/local/cuda-11.8  or /usr/local/cuda
```

---

## How to Check: Dynamic vs Static Linking

**ldd** shows what shared libs a binary needs at runtime:
```bash
ldd rocksdb-gpu/db_bench
# "not found" = missing at runtime → need LD_LIBRARY_PATH or rebuild statically
```

**nm / objdump** to check if a symbol is linked in:
```bash
nm -D rocksdb-gpu/db_bench | grep gflags    # dynamic symbols
nm rocksdb-gpu/db_bench | grep gflags       # all symbols (static link shows them here)
```

**Current db_bench state** (as of cb7bacd):
- `libgflags.so.2.2 => not found` — it's dynamically linked to gflags but the
  .so is not in system path, so you MUST set LD_LIBRARY_PATH before running.
- Everything else (zlib, zstd, bz2, etc.) resolves from system `/lib/`.

---

## Commands: Build db_bench

Always run from `rocksdb-gpu/` directory.

### Static build (recommended — no runtime LD_LIBRARY_PATH needed after build)
```bash
cd /nfs/ug/groups/ece1755_w26_group1/rocksdb/rocksdb-gpu

GFLAGS_DIR=/nfs/ug/groups/ece1755_w26_group1/local/gflags
make -j8 db_bench \
  EXTRA_CXXFLAGS="-DGFLAGS -I${GFLAGS_DIR}/usr/include" \
  EXTRA_LDFLAGS="-L${GFLAGS_DIR}/usr/lib/x86_64-linux-gnu" \
  DEBUG_LEVEL=0 \
  LIB_MODE=static
```

### Dynamic build (faster link, but needs LD_LIBRARY_PATH every time you run)
```bash
GFLAGS_DIR=/nfs/ug/groups/ece1755_w26_group1/local/gflags
make -j8 db_bench \
  EXTRA_CXXFLAGS="-DGFLAGS -I${GFLAGS_DIR}/usr/include" \
  EXTRA_LDFLAGS="-L${GFLAGS_DIR}/usr/lib/x86_64-linux-gnu" \
  DEBUG_LEVEL=0 \
  LIB_MODE=shared
```

### Debug build (adds assertions, slower)
```bash
GFLAGS_DIR=/nfs/ug/groups/ece1755_w26_group1/local/gflags
make -j8 db_bench \
  EXTRA_CXXFLAGS="-DGFLAGS -I${GFLAGS_DIR}/usr/include" \
  EXTRA_LDFLAGS="-L${GFLAGS_DIR}/usr/lib/x86_64-linux-gnu" \
  DEBUG_LEVEL=1 \
  LIB_MODE=static
```

---

## Commands: Run db_bench

### tcsh (default shell on ug71)
```tcsh
setenv GFLAGS_DIR /nfs/ug/groups/ece1755_w26_group1/local/gflags
setenv LD_LIBRARY_PATH ${GFLAGS_DIR}/usr/lib/x86_64-linux-gnu

# quick sanity test (50k keys, no WAL, no compression)
./db_bench --benchmarks=fillrandom --num=50000 --value_size=32 \
  --db=/tmp/testdb --use_existing_db=0 \
  --compression_type=none --disable_wal=true --threads=1
```

### bash
```bash
export GFLAGS_DIR=/nfs/ug/groups/ece1755_w26_group1/local/gflags
export LD_LIBRARY_PATH="${GFLAGS_DIR}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

./db_bench --benchmarks=fillrandom --num=50000 --value_size=32 \
  --db=/tmp/testdb --use_existing_db=0 \
  --compression_type=none --disable_wal=true --threads=1
```

---

## Commands: Build GPU Benchmark Tools

From `benchmarks/gpu/`:
```bash
cd /nfs/ug/groups/ece1755_w26_group1/rocksdb/benchmarks/gpu

# build the cuFile I/O replay helper
make gpu_file_replay_bench

# build the GPU compaction worker (links librocksdb.a — slow first time)
make gpu_compaction_worker

# build all tools at once
make tools
```
GDS_LOCAL is auto-detected from `../../../cuda_test/gds/local` (see Makefile).

---

## Commands: Run GPU Benchmark Sweep

All scripts use `benchmarks/gpu/config/.env.local` for DATA_DIR / OUTPUT_DIR.
The file already exists on this machine.

```bash
cd /nfs/ug/groups/ece1755_w26_group1/rocksdb

# full 4-value-size GPU hook sweep (5 reps each, ~5 min total)
bash benchmarks/gpu/scripts/run_fillrandom_gpu.sh

# single hook run (one value size, custom reps)
NUM_REPS=3 INPUT_SST_MB=8 \
  bash benchmarks/gpu/scripts/run_rocksdb_gpu_hook_bench.sh
```

---

## Commands: Plot Results

```bash
cd /nfs/ug/groups/ece1755_w26_group1/rocksdb

# CPU + GPU comparison (throughput_avg.png, throughput_realtime.png)
python3 benchmarks/cpu/python/plot_results.py
# Output: benchmarks/cpu/results/plots/

# GPU-only hook bench plot
python3 benchmarks/gpu/python/plot_gpu_compaction_hook_bench.py \
  benchmarks/gpu/results/fillrandom_gpu/value_32/rocksdb_hook/<RUN_ID> --plot
```
