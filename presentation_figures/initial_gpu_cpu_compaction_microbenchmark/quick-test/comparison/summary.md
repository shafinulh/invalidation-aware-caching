# CPU vs GPU Compaction Comparison

## CPU Single-Threaded: RocksDB vs Synthetic Baseline

Comparison of RocksDB compactall (subcomp=1) wall time vs the
synthetic CPU compaction baseline from gpcomp_bench.

| Value Size | RocksDB sub=1 (ms) | Throughput (Mrec/s) | Synthetic CPU (ms) | Throughput (Mrec/s) | Ratio |
|------------|-------------------|--------------------|-------------------|---------------------|-------|
| 32B | 1718.5 +/- 1.2 | 3.25 +/- 0.00 | 1860.9 +/- 0.0 | 3.01 | 1.08x |
| 1024B | 342.8 +/- 1.0 | 0.75 +/- 0.00 | 604.2 +/- 0.0 | 0.43 | 1.76x |

## CPU Multithreading Sweep

### Value Size: 32B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 3.25 +/- 0.00 | 1.00x | 88.7% | 12.4% |
| 4 | 10.11 +/- 0.04 | 3.11x | 296.1% | 30.0% |
| 16 | 10.46 +/- 0.15 | 3.21x | 657.1% | 60.0% |

### Value Size: 1024B

| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 | Avg CPU % | Avg IO % |
|----------------|--------------------|-----------------|-----------|----------|
| 1 | 0.75 +/- 0.00 | 1.00x | 51.4% | 48.2% |
| 4 | 1.34 +/- 0.01 | 1.78x | 119.3% | 73.1% |
| 16 | 0.84 +/- 0.01 | 1.11x | 240.2% | 66.2% |

## GPU Compaction Results

| Value Size | Mode | GPU Time (ms) | CPU Synthetic (ms) | Speedup | Avg CPU % | Avg IO % |
|------------|------|---------------|-------------------|---------|-----------|----------|
| 32B | c_paper_with_plan | 568.5 +/- 0.0 | 1860.9 +/- 0.0 | 3.27x | 65.0% | 12.3% |
| 32B | q_paper_with_plan | 370.1 +/- 0.0 | 1851.2 +/- 0.0 | 5.00x | 58.5% | 17.2% |
| 1024B | c_paper_with_plan | 538.7 +/- 0.0 | 604.2 +/- 0.0 | 1.12x | 67.1% | 16.4% |
| 1024B | q_paper_with_plan | 328.9 +/- 0.0 | 601.2 +/- 0.0 | 1.83x | 60.7% | 20.8% |

