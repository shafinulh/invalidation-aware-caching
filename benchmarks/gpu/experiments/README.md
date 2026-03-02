# GPU IO Benchmark Experiments

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
