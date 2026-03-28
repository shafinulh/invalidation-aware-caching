# CPU vs GPU Compaction Comparison

## CPU Single-Threaded: RocksDB vs Synthetic Baseline

Comparison of RocksDB compactall (subcomp=1) wall time vs the
synthetic CPU compaction baseline from gpcomp_bench.

| Value Size | RocksDB sub=1 (ms) | Throughput (Mrec/s) | Synthetic CPU (ms) | Throughput (Mrec/s) | Ratio |
|------------|-------------------|--------------------|-------------------|---------------------|-------|
| 32B | 1697.5 +/- 4.1 | 3.29 +/- 0.01 | 1840.6 +/- 0.0 | 3.04 | 1.08x |
| 1024B | 363.0 +/- 2.0 | 0.71 +/- 0.00 | 566.3 +/- 0.0 | 0.46 | 1.56x |

## CPU Multithreading Sweep

### Value Size: 32B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 3.29 +/- 0.01 | 1.00x | 88.7% | 11.4% |
| 4 | 10.18 +/- 0.06 | 3.09x | 297.9% | 30.9% |
| 16 | 10.23 +/- 0.24 | 3.11x | 697.6% | 58.5% |

### Value Size: 1024B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 0.71 +/- 0.00 | 1.00x | 56.4% | 47.9% |
| 4 | 1.34 +/- 0.02 | 1.88x | 77.1% | 80.6% |
| 16 | 0.90 +/- 0.03 | 1.27x | 339.4% | 70.8% |

## GPU Compaction Results

| Value Size | Mode | GPU Time (ms) | CPU Synthetic (ms) | Speedup | Avg CPU % | Avg IO % |
|------------|------|---------------|-------------------|---------|-----------|----------|
| 32B | c_paper_with_plan | 552.8 +/- 0.0 | 1840.6 +/- 0.0 | 3.33x | 65.8% | 12.6% |
| 32B | c_paper_with_plan_streaming_io | 402.4 +/- 0.0 | 1852.9 +/- 0.0 | 4.60x | 58.4% | 24.4% |
| 32B | c_paper_without_plan | 556.0 +/- 0.0 | 1854.3 +/- 0.0 | 3.33x | 79.6% | 14.0% |
| 32B | q_paper_with_plan | 434.7 +/- 0.0 | 1859.3 +/- 0.0 | 4.28x | 58.4% | 18.7% |
| 32B | q_paper_with_plan_streaming_io | 287.1 +/- 0.0 | 1846.7 +/- 0.0 | 6.43x | 46.4% | 30.2% |
| 32B | q_paper_without_plan | 449.5 +/- 0.0 | 1846.9 +/- 0.0 | 4.11x | 78.6% | 18.2% |
| 1024B | c_paper_with_plan | 371.9 +/- 0.0 | 566.3 +/- 0.0 | 1.52x | 68.8% | 14.8% |
| 1024B | c_paper_with_plan_streaming_io | 226.5 +/- 0.0 | 568.6 +/- 0.0 | 2.51x | 39.7% | 44.4% |
| 1024B | c_paper_without_plan | 365.5 +/- 0.0 | 566.0 +/- 0.0 | 1.55x | 77.1% | 20.0% |
| 1024B | q_paper_with_plan | 370.5 +/- 0.0 | 565.1 +/- 0.0 | 1.53x | 62.2% | 21.3% |
| 1024B | q_paper_with_plan_streaming_io | 250.4 +/- 0.0 | 565.5 +/- 0.0 | 2.26x | 37.2% | 39.6% |
| 1024B | q_paper_without_plan | 375.7 +/- 0.0 | 566.5 +/- 0.0 | 1.51x | 74.6% | 22.5% |

