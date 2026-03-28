# CPU vs GPU Compaction Comparison

## CPU Single-Threaded: RocksDB vs Synthetic Baseline

Comparison of RocksDB compactall (subcomp=1) wall time vs the
synthetic CPU compaction baseline from gpcomp_bench.

| Value Size | RocksDB sub=1 (ms) | Throughput (Mrec/s) | Synthetic CPU (ms) | Throughput (Mrec/s) | Ratio |
|------------|-------------------|--------------------|-------------------|---------------------|-------|
| 32B | 1302.7 +/- 3.6 | 4.28 +/- 0.01 | 1140.4 +/- 0.0 | 4.90 | 0.88x |
| 1024B | 282.2 +/- 4.0 | 0.91 +/- 0.01 | 662.0 +/- 0.0 | 0.39 | 2.35x |

## CPU Multithreading Sweep

### Value Size: 32B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 4.28 +/- 0.01 | 1.00x | 85.7% | 11.1% |
| 4 | 13.93 +/- 0.14 | 3.25x | 283.6% | 32.5% |
| 16 | 15.12 +/- 0.25 | 3.53x | 365.4% | 40.2% |

### Value Size: 1024B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 0.91 +/- 0.01 | 1.00x | 54.7% | 47.2% |
| 4 | 1.87 +/- 0.02 | 2.05x | 134.3% | 80.6% |
| 16 | 1.81 +/- 0.02 | 1.98x | 134.1% | 78.5% |

## GPU Compaction Results

| Value Size | Mode | GPU Time (ms) | CPU Synthetic (ms) | Speedup | Avg CPU % | Avg IO % |
|------------|------|---------------|-------------------|---------|-----------|----------|
| 32B | c_paper_with_plan | 601.1 +/- 0.0 | 1140.4 +/- 0.0 | 1.90x | 75.8% | 12.2% |
| 32B | c_paper_with_plan_streaming_io | 463.7 +/- 0.0 | 1138.7 +/- 0.0 | 2.46x | 65.2% | 16.1% |
| 32B | q_paper_with_plan | 487.1 +/- 0.0 | 1133.9 +/- 0.0 | 2.33x | 74.0% | 15.6% |
| 32B | q_paper_with_plan_streaming_io | 350.0 +/- 0.0 | 1133.1 +/- 0.0 | 3.24x | 51.1% | 22.6% |
| 1024B | c_paper_with_plan | 446.7 +/- 0.0 | 662.0 +/- 0.0 | 1.48x | 75.7% | 12.8% |
| 1024B | c_paper_with_plan_streaming_io | 307.6 +/- 0.0 | 660.3 +/- 0.0 | 2.15x | 33.2% | 27.3% |
| 1024B | q_paper_with_plan | 451.8 +/- 0.0 | 672.5 +/- 0.0 | 1.49x | 75.1% | 13.7% |
| 1024B | q_paper_with_plan_streaming_io | 310.0 +/- 0.0 | 658.5 +/- 0.0 | 2.12x | 56.3% | 25.8% |

