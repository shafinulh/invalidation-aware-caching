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

Current benchmark result on this machine (`RTX 3070`, 4 x ~8 MiB input SSTs):

- CPU total: `135.29 ms` min
- GPU total: `117.97 ms` min
- speedup: `1.15x`
- output SST match: `PASS`

Main remaining optimization gap relative to the paper:

- block planning is still CPU-side and sequential
- SST assembly is still mostly host-side
- unpack is launched file-by-file instead of with streams
- GPU kernels are fast, but host orchestration now dominates the total wall time
