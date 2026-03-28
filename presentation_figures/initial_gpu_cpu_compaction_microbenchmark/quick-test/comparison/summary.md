# CPU vs GPU Compaction Comparison

## CPU Single-Threaded: RocksDB vs Synthetic Baseline

Comparison of RocksDB compactall (subcomp=1) wall time vs the
synthetic CPU compaction baseline from gpcomp_bench.

| Value Size | RocksDB sub=1 (ms) | Throughput (Mrec/s) | Synthetic CPU (ms) | Throughput (Mrec/s) | Ratio |
|------------|-------------------|--------------------|-------------------|---------------------|-------|
| 32B | 1700.7 +/- 8.5 | 3.29 +/- 0.02 | 1892.6 +/- 0.0 | 2.95 | 1.11x |
| 1024B | 346.9 +/- 10.9 | 0.74 +/- 0.02 | 569.7 +/- 0.0 | 0.45 | 1.64x |

## CPU Multithreading Sweep

### Value Size: 32B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 3.29 +/- 0.02 | 1.00x | 87.8% | 11.9% |
| 4 | 10.23 +/- 0.05 | 3.11x | 298.1% | 30.5% |
| 16 | 10.54 +/- 0.09 | 3.20x | 670.5% | 58.3% |

### Value Size: 1024B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 0.74 +/- 0.02 | 1.00x | 49.7% | 46.2% |
| 4 | 1.32 +/- 0.03 | 1.77x | 104.5% | 68.6% |
| 16 | 0.88 +/- 0.01 | 1.18x | 192.1% | 64.1% |

## GPU Compaction Results

| Value Size | Mode | GPU Time (ms) | CPU Synthetic (ms) | Speedup | Avg CPU % | Avg IO % |
|------------|------|---------------|-------------------|---------|-----------|----------|
| 32B | c_paper_with_plan | 545.5 +/- 0.0 | 1892.6 +/- 0.0 | 3.47x | 64.2% | 13.3% |
| 32B | c_paper_with_plan_streaming_io | 388.0 +/- 0.0 | 1932.3 +/- 0.0 | 4.98x | 55.5% | 17.4% |
| 32B | q_paper_with_plan | 436.4 +/- 0.0 | 1900.1 +/- 0.0 | 4.35x | 59.6% | 17.8% |
| 32B | q_paper_with_plan_streaming_io | 285.0 +/- 0.0 | 1902.8 +/- 0.0 | 6.68x | 45.9% | 23.3% |
| 1024B | c_paper_with_plan | 368.1 +/- 0.0 | 569.7 +/- 0.0 | 1.55x | 66.3% | 16.7% |
| 1024B | c_paper_with_plan_streaming_io | 229.9 +/- 0.0 | 572.3 +/- 0.0 | 2.49x | 53.0% | 31.9% |
| 1024B | q_paper_with_plan | 369.1 +/- 0.0 | 577.2 +/- 0.0 | 1.56x | 62.2% | 20.5% |
| 1024B | q_paper_with_plan_streaming_io | 261.9 +/- 0.0 | 561.8 +/- 0.0 | 2.14x | 51.2% | 33.3% |

