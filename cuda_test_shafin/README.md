# `cuda_test_shafin`

This directory is a narrowed GPComp reproduction focused only on the paper's Q-compaction path.

Fixed configuration:

- 4 input SSTables
- 8 MiB dummy SSTable target size per input file
- 16-byte keys
- 32-byte values
- 32 KiB data blocks
- restart interval = 4
- Q-compaction only
- no garbage collection
- CPU vs GPU timing measured from first input-file read to final output-file write

Implemented units:

- Unpack unit: parse SST data blocks into flat KV arrays
- Sort unit: Algorithm 1 merge kernel from the paper
- Pack unit:
  - data block generation with restart interval 4
  - Bloom filter generation with Algorithm 2
  - index block emission
  - SST footer / metadata emission

What was intentionally removed from the copied baseline:

- C-compaction paths
- mixed benchmark modes unrelated to the first Q-compaction milestone
- configurable restart interval and approximate keys-per-block modelling
- legacy merge+bloom-only simulations

Build:

```bash
make all
```

Generate the 4 input SSTables:

```bash
./gpcomp_datagen --out_dir dataset_shafin
```

Run unit tests:

```bash
./gpcomp_unit_tests
```

Run the end-to-end benchmark:

```bash
./gpcomp_bench --dataset dataset_shafin --out_dir results_shafin --runs 3
```

Run the stricter Q-compaction pipeline experiment that keeps the merged KV array on device and only materializes host output state near the end:

```bash
./gpcomp_bench --dataset dataset_shafin --out_dir results_shafin --runs 3 --gpu_mode q_pipeline
```

Run the paper-shaped Q-compaction mode that plans at restart-group granularity, packs one output SST span per CUDA stream, and preserves CPU/GPU output identity under that layout:

```bash
./gpcomp_bench --dataset dataset_shafin --out_dir results_shafin --runs 3 --gpu_mode q_paper
```

Run the experimental overlap variant that precompresses restart groups on a separate GPU stream while restart-group sizes are copied back for CPU planning:

```bash
./gpcomp_bench --dataset dataset_shafin --out_dir results_shafin --runs 3 --gpu_mode q_paper_overlap
```

Benchmark snapshots on this machine (`RTX 3070`, 4 x ~8 MiB input SSTs):

- legacy path: CPU `141.27 ms`, GPU `62.28 ms`, speedup `2.27x`, `PASS`
- strict Q-pipeline path: CPU `136.65 ms`, GPU `98.21 ms`, speedup `1.39x`, `PASS`
- paper-shaped Q path: CPU `133.83 ms`, GPU `43.77 ms`, speedup `3.06x`, `PASS`

Preserved baseline and current logs:

- baseline: `results_shafin/gpcomp_bench_baseline_20260315.log`
- current legacy benchmark: `results_shafin/gpcomp_bench.log`
- legacy snapshot: `results_shafin/gpcomp_bench_legacy_20260315.log`
- Q-pipeline snapshot: `results_shafin/gpcomp_bench_q_pipeline_20260315.log`
- Q-paper snapshot: `results_shafin/gpcomp_bench_q_paper_20260315.log`
- Q-paper overlap snapshot: `results_shafin/gpcomp_bench_q_paper_overlap_20260315.log`

Main remaining optimization gap relative to the paper:

- block planning and output-file partitioning are still CPU-side and sequential
- index/meta/footer assembly is still host-side rather than GPU-generated
- unpack still materializes full values instead of key + value references as described in the paper
- disk I/O is still staged through host memory; the pipeline/GDS path is not wired into the benchmarked Q-compaction flow yet
