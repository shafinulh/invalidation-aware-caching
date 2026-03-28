# CPU vs GPU Compaction Comparison

## CPU Single-Threaded: RocksDB vs Synthetic Baseline

Comparison of RocksDB compactall (subcomp=1) wall time vs the
synthetic CPU compaction baseline from gpcomp_bench.

| Value Size | RocksDB sub=1 (ms) | Throughput (Mrec/s) | Synthetic CPU (ms) | Throughput (Mrec/s) | Ratio |
|------------|-------------------|--------------------|-------------------|---------------------|-------|
| 32B | 1689.8 +/- 3.1 | 3.31 +/- 0.01 | 1914.6 +/- 0.0 | 2.92 | 1.13x |
| 1024B | 360.5 +/- 4.8 | 0.72 +/- 0.01 | 564.7 +/- 0.0 | 0.46 | 1.57x |

## CPU Multithreading Sweep

### Value Size: 32B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 3.31 +/- 0.01 | 1.00x | 88.3% | 13.8% |
| 4 | 10.15 +/- 0.03 | 3.07x | 291.2% | 31.0% |
| 16 | 10.65 +/- 0.71 | 3.22x | 564.9% | 58.8% |

### Value Size: 1024B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 0.72 +/- 0.01 | 1.00x | 54.7% | 47.6% |
| 4 | 1.30 +/- 0.03 | 1.82x | 102.0% | 66.4% |
| 16 | 0.91 +/- 0.00 | 1.27x | 391.9% | 72.9% |

## GPU Compaction Results

| Value Size | Mode | GPU Time (ms) | CPU Synthetic (ms) | Speedup | Avg CPU % | Avg IO % |
|------------|------|---------------|-------------------|---------|-----------|----------|
| 32B | c_paper_with_plan | 500.7 +/- 0.0 | 1914.6 +/- 0.0 | 3.82x | 65.4% | 13.1% |
| 32B | c_paper_with_plan_streaming_io | 390.5 +/- 0.0 | 1929.4 +/- 0.0 | 4.94x | 49.5% | 20.4% |
| 32B | c_paper_without_plan | 512.2 +/- 0.0 | 1908.5 +/- 0.0 | 3.73x | 76.9% | 16.1% |
| 32B | q_paper_with_plan | 387.4 +/- 0.0 | 1900.8 +/- 0.0 | 4.91x | 57.3% | 17.7% |
| 32B | q_paper_with_plan_streaming_io | 301.3 +/- 0.0 | 1904.0 +/- 0.0 | 6.32x | 41.3% | 30.1% |
| 32B | q_paper_without_plan | 407.7 +/- 0.0 | 1893.9 +/- 0.0 | 4.65x | 75.0% | 20.0% |
| 1024B | c_paper_with_plan | 318.0 +/- 0.0 | 564.7 +/- 0.0 | 1.78x | 67.1% | 14.5% |
| 1024B | c_paper_with_plan_streaming_io | 217.1 +/- 0.0 | 581.8 +/- 0.0 | 2.68x | 35.7% | 38.9% |
| 1024B | c_paper_without_plan | 316.8 +/- 0.0 | 586.7 +/- 0.0 | 1.85x | 69.6% | 23.6% |
| 1024B | q_paper_with_plan | 323.0 +/- 0.0 | 582.1 +/- 0.0 | 1.80x | 61.9% | 22.1% |
| 1024B | q_paper_with_plan_streaming_io | 227.0 +/- 0.0 | 565.0 +/- 0.0 | 2.49x | 41.6% | 38.0% |
| 1024B | q_paper_without_plan | 310.4 +/- 0.0 | 566.3 +/- 0.0 | 1.82x | 71.1% | 23.0% |

