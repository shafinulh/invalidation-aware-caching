# CPU vs GPU Compaction Comparison

## CPU Single-Threaded: RocksDB vs Synthetic Baseline

Comparison of RocksDB compactall (subcomp=1) wall time vs the
synthetic CPU compaction baseline from gpcomp_bench.

| Value Size | RocksDB sub=1 (ms) | Throughput (Mrec/s) | Synthetic CPU (ms) | Throughput (Mrec/s) | Ratio |
|------------|-------------------|--------------------|-------------------|---------------------|-------|
| 32B | 5935.8 +/- 24.0 | 3.76 +/- 0.02 | 5906.8 +/- 0.0 | 3.79 | 1.00x |
| 1024B | 1157.1 +/- 5.0 | 0.89 +/- 0.00 | 2736.7 +/- 0.0 | 0.38 | 2.37x |

## CPU Multithreading Sweep

### Value Size: 32B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 3.76 +/- 0.02 | 1.00x | 90.4% | 10.4% |
| 4 | 12.19 +/- 0.07 | 3.24x | 299.5% | 29.6% |
| 16 | 19.90 +/- 0.31 | 5.29x | 831.9% | 54.4% |

### Value Size: 1024B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 0.89 +/- 0.00 | 1.00x | 53.3% | 45.9% |
| 4 | 1.88 +/- 0.00 | 2.11x | 153.8% | 80.0% |
| 16 | 1.76 +/- 0.04 | 1.97x | 405.1% | 76.5% |

## GPU Compaction Results

| Value Size | Mode | GPU Time (ms) | CPU Synthetic (ms) | Speedup | Avg CPU % | Avg IO % |
|------------|------|---------------|-------------------|---------|-----------|----------|
| 32B | c_paper_with_plan | 2367.6 +/- 0.0 | 5906.8 +/- 0.0 | 2.49x | 74.1% | 11.7% |
| 32B | c_paper_with_plan_streaming_io | 1542.2 +/- 0.0 | 5855.1 +/- 0.0 | 3.80x | 71.5% | 20.8% |
| 32B | c_paper_without_plan | 2389.6 +/- 0.0 | 5870.3 +/- 0.0 | 2.46x | 84.1% | 11.7% |
| 32B | q_paper_with_plan | 1947.6 +/- 0.0 | 5893.8 +/- 0.0 | 3.03x | 70.1% | 17.0% |
| 32B | q_paper_with_plan_streaming_io | 1139.2 +/- 0.0 | 5836.3 +/- 0.0 | 5.12x | 70.9% | 30.7% |
| 32B | q_paper_without_plan | 2011.3 +/- 0.0 | 5883.8 +/- 0.0 | 2.93x | 85.3% | 14.2% |
| 1024B | c_paper_with_plan | 1796.2 +/- 0.0 | 2736.7 +/- 0.0 | 1.52x | 74.8% | 11.8% |
| 1024B | c_paper_with_plan_streaming_io | 974.5 +/- 0.0 | 2734.2 +/- 0.0 | 2.81x | 64.3% | 34.1% |
| 1024B | c_paper_without_plan | 1802.3 +/- 0.0 | 2747.9 +/- 0.0 | 1.52x | 83.7% | 16.0% |
| 1024B | q_paper_with_plan | 1755.7 +/- 0.0 | 2757.1 +/- 0.0 | 1.57x | 71.8% | 17.0% |
| 1024B | q_paper_with_plan_streaming_io | 937.9 +/- 0.0 | 2746.3 +/- 0.0 | 2.93x | 66.4% | 37.0% |
| 1024B | q_paper_without_plan | 1768.2 +/- 0.0 | 2760.6 +/- 0.0 | 1.56x | 83.1% | 16.9% |

