# `cuda_test_shafin`

This is the small GPComp sandbox for end-to-end GPU compaction experiments.

Right now the benchmark compares one CPU baseline against 4 main GPU modes:

- `q_paper_with_plan`
- `q_paper_without_plan`
- `c_paper_with_plan`
- `c_paper_without_plan`

There are also two extra experimental modes for C-compaction:

- `c_paper_keys_only_with_plan`
- `c_paper_keys_only_without_plan`

## CPU baseline used everywhere

Every benchmark run compares against the same CPU baseline:

- CPU unpack
- CPU merge/sort
- CPU full garbage collection
- CPU group-aligned planning
- CPU bloom generation
- CPU pack + SST assembly + write

## The 4 GPU modes

- `q_paper_with_plan`
  - GPU does unpack + merge.
  - No GPU GC.
  - GPU sends restart-group size metadata to CPU.
  - CPU does planning.
  - GPU packs and writes.
  - Overhead: planning round trip.

- `q_paper_without_plan`
  - GPU does unpack + merge.
  - No GPU GC.
  - No CPU planning round trip.
  - Uses static planning.
  - Overhead: smallest host/device overhead, but output layout can differ from CPU.

- `c_paper_with_plan`
  - GPU does unpack + merge.
  - Then data goes to CPU for garbage collection.
  - After GC, GPU computes restart-group sizes, CPU does planning, GPU packs and writes.
  - Overhead: GC round trip + planning round trip.

- `c_paper_without_plan`
  - GPU does unpack + merge.
  - Then data goes to CPU for garbage collection.
  - After GC, it skips CPU planning and uses static planning.
  - Overhead: GC round trip, but less metadata traffic than `c_paper_with_plan`.

- `q` vs `c` tells you how much the CPU-side GC costs.
- `with_plan` vs `without_plan` tells you how much the planning round trip costs.

## Build

```bash
cd /home/1755_project/invalidation-aware-caching/cuda_test_shafin
make all
```

## Synthetic data generation

Synthetic input SSTs are generated with `gpcomp_datagen`.

Basic example:

```bash
./gpcomp_datagen --out_dir dataset
```

Uniform datagen over a large key space:

```bash
./gpcomp_datagen --out_dir dataset_uniform20m --seed 23 --zipf_alpha 0.0 --user_key_space 20000000
```

Zipfian datagen example:

```bash
./gpcomp_datagen --out_dir dataset_zipf --seed 23 --zipf_alpha 0.6 --user_key_space 20000000
```

Useful knobs:

- `--zipf_alpha 0.0` means uniform in the current generator
- larger `--zipf_alpha` means more skew / more overlap
- `--user_key_space` controls how many user keys we draw from
- smaller key space means more overwritten versions and more work for GC

## Unit tests

```bash
./gpcomp_unit_tests
```

## Single benchmark runs

```bash
./gpcomp_bench --dataset dataset --out_dir results_tmp --runs 5 --gpu_mode q_paper_with_plan
./gpcomp_bench --dataset dataset --out_dir results_tmp --runs 5 --gpu_mode q_paper_without_plan
./gpcomp_bench --dataset dataset --out_dir results_tmp --runs 5 --gpu_mode c_paper_with_plan
./gpcomp_bench --dataset dataset --out_dir results_tmp --runs 5 --gpu_mode c_paper_without_plan
```

Keys-only GC experiment:

```bash
./gpcomp_bench --dataset dataset --out_dir results_tmp --runs 5 --gpu_mode c_paper_keys_only_with_plan
./gpcomp_bench --dataset dataset --out_dir results_tmp --runs 5 --gpu_mode c_paper_keys_only_without_plan
```

## Sweeps and plots

The sweep script rebuilds for each value size, regenerates datasets, runs the selected modes, saves logs under `sweep_results/`, and then calls `plot_results.py`.

4-SST sweep:

```bash
bash run_sweep.sh --num_ssts 4 --values "32 64 128 256 512 1024" --runs 5
```

24-SST sweep:

```bash
bash run_sweep.sh --num_ssts 24 --values "32 64 128 256 512 1024" --runs 5
```

Current defaults in `run_sweep.sh`:

- uniform data (`--zipf_alpha 0.0`)
- user key space `20000000`
- modes `q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan`

The plots currently generated are:

- throughput
- CPU utilization
- SSD read/write bandwidth

## Nsight sweeps

Use `sweep_with_nsight.sh` to rerun the same value-size sweeps under Nsight.

4-SST Nsight Systems sweep:

```bash
bash sweep_with_nsight.sh --tool nsys --num_ssts 4 --values "32 64 128 256 512 1024" --runs 5
```

24-SST Nsight Systems sweep:

```bash
bash sweep_with_nsight.sh --tool nsys --num_ssts 24 --values "32 64 128 256 512 1024" --runs 5
```

Default Nsight output directories:

- `sweep_results/sweep_nsight_nsys_8mb-sst_4sst`
- `sweep_results/sweep_nsight_nsys_8mb-sst_24sst`

Inside each Nsight sweep directory:

- `profiles/`: raw Nsight reports (`.nsys-rep`, `.qdstrm`, or `.ncu-rep`) plus Nsight stats CSVs
- `result_val*B_*.log`: benchmark stdout/stderr for each mode
- `nsight_manifest.csv`: manifest mapping each run to its profile files

Important: raw Nsight trace files can capture local environment metadata from the profiled session. For public sharing, commit the derived summary CSVs, manifests, and logs, but keep raw `.nsys-rep`, `.sqlite`, `.qdstrm`, and `.ncu-rep` files out of git unless you have scrubbed them or are using a private channel.

Plotting is disabled by default for Nsight sweeps. Pass `--plot` to also run `plot_nsight_sweep.py`. Pass `--timestamp_output` if you want a timestamp appended to the sweep directory name instead of overwriting the previous Nsight sweep directory.

## Main files

- `gpcomp_bench.cu`: benchmark driver and CPU/GPU comparison
- `gpcomp_pipeline.cuh`: CPU and GPU compaction pipelines
- `gpcomp_datagen.cpp`: synthetic SST generation entry point
- `gpcomp_dummy_data.cuh`: key/value generation and Zipf / key-space control
- `run_sweep.sh`: full sweep runner
- `plot_results.py`: plotting from benchmark logs
