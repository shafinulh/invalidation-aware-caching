# GPU Benchmark Experiments

Pre-configured experiments for comparing GPU vs CPU IO paths
in a simulated LSM-tree compaction.

## Available Experiments

### `io_bench/`

| Script         | L0 Size | Description                     |
| -------------- | ------- | ------------------------------- |
| `io_8mb.sh`    | 8 MB    | Default GPComp paper config     |
| `io_64mb.sh`   | 64 MB   | Large memtable configuration    |
| `io_sweep.sh`  | Both    | Full sweep (8 MB + 64 MB)       |

All experiments simulate:
- **Read phase**: 4 L0 files (configurable via `NUM_L0_READ`)
- **Write phase**: 3 L1 files (configurable via `NUM_L1_WRITE`)
- **Workload parameters**: set inline in each script, not in `config/.env.local`

## Running

From the repository root:

```bash
./benchmarks/gpu/experiments/io_bench/io_sweep.sh
```

### `rocksdb_hook/`

| Script             | Description                                    |
| ------------------ | ---------------------------------------------- |
| `hook_replay.sh`   | Dummy GPU compaction hook benchmark for RocksDB |

This experiment drives a RocksDB manual compaction through the
`CompactionService` hook, stages a ground-truth compaction output, then
replays that output through the cuFile-backed GPU replay helper.

Run it from the repository root:

```bash
./benchmarks/gpu/experiments/rocksdb_hook/hook_replay.sh
```
