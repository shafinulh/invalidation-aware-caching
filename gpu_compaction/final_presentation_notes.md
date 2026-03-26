## QUICK RUN

This is what I used to generate the current `sweep_results/` and `graphs/`.

```bash
cd /home/1755_project/invalidation-aware-caching/cuda_test_shafin

# 4 input SSTs, all value sizes
bash run_sweep.sh --num_ssts 4 --values "32 64 128 256 512 1024" --runs 5

# 24 input SSTs, all value sizes
bash run_sweep.sh --num_ssts 24 --values "32 64 128 256 512 1024" --runs 5
```

The sweep script already does the plotting at the end, so it fills in:

- `sweep_results/sweep_8mb-sst_4sst`
- `sweep_results/sweep_8mb-sst_24sst`
- `graphs/`

Current sweep defaults:

- dataset distribution: uniform (`--zipf_alpha 0.0`)
- user key space: `20000000`
- modes: `q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan`
- runs per mode: `5`

For Zipf instead of uniform:

```bash
bash run_sweep.sh --num_ssts 4 --values "32 64 128 256 512 1024" --runs 5 --zipf_alpha 0.6 --user_key_space 20000000
```

# Generate charts from sweep results
python3 plot_results.py --sweep_dir sweep_results/sweep_8mb-sst_4sst --graphs_dir graphs
python3 plot_results.py --sweep_dir sweep_results/sweep_8mb-sst_24sst --graphs_dir graphs

Optional Nsight profiling:

./nsight_profile.sh --tool nsys --dataset dataset_shafin_V32 --gpu_mode q_paper_with_plan --out_dir results/nsight

# TODO:
1. presentation slides for the GPCOMP compaction kernels + the modifications we made (PLAN for Q-compaction to properly pack blocks)
  - slides for the microbenchmarking results (comparing q_paper_with_plan, q_paper, c_paper)
  - will later make slides for gpu compaction vs cpu multithreading
2. implement q_paper with naive, static, KV grouping for data blocks
3. implement c_paper with CPU side garbage collection
4. do the microbenchmarks for different Input, Value, SSTable sizes as is done for the CPU microbenchmarking experiments: /invalidation-aware-caching/benchmarks/cpu/experiments/compaction_parallelism/final_compaction_sweep.sh)  
5. perform IO/GPU (NVIDIA Nsight) measurements on 


# NOTES:
KV pairs = (16B key + 32B value)

KV Groups (groups of KV pairs) = 4 KV pairs (restart interval)
--- uses prefix compression
--- each KV group will have a different final size based on how much prefix compression

Data blocks (groups of KV groups) = 32KB


************************Q-COMPACTION WITHOUT PLAN***************************************************

KV Pairs: AAAA0001 AAAA0002 AAAA0003 AAAA0004
KV groups = (AAAA0001 AAAA0002 AAAA0003 AAAA0004)
with prefix compression = (AAAA0001, +2(02), +2(03), +2(04))
------------------------------------
NO TRANSFER TO CPU
----------------------------
GPU can now split up KV groups statically into data blocks without needing to do any planning on CPU.
HOWEVER, to avoid index memory overflow/segmentation faults, we must enforce a safe static limit.
Current Implementation (updated) = Use a safe WORST-CASE block-capacity model based on pack encoding overhead,
then align to restart groups (4 KV per group).
Old rule (75% raw uncompressed cap) was too conservative and created too many tiny blocks.

THE PROBLEM WITH STATIC PLANNING (updated interpretation):
The previous static rule used an overly conservative cap (about 75% of block size from raw/uncompressed intuition).
When keys share long prefixes, real encoded entries are much smaller than that worst raw estimate.

Example intuition:
- Raw estimate can hit the cap early (for example, 300 keys * 80 B = 24,000 B).
- Actual prefix-compressed payload for those keys can be much smaller.
- That early cap creates underfilled data blocks, so we emit too many blocks.

Why this hurts performance:
- More data blocks means more index entries, filter metadata, and file assembly work.
- Even if planning is skipped on CPU, host-side pack/assemble metadata handling can increase.
- Net effect: WITHOUT PLAN can lose some expected gains if static packing over-fragments.

What we changed in the implementation:
- Replaced the old 75% static cap with a safer encoding-aware worst-case block-capacity rule.
- Kept restart-group alignment (4 KV per restart group) for correctness/safety.
- Result: reduced over-fragmentation and fewer tiny blocks than the old static heuristic.

Why speedup can still look modest after this fix:
- Fine-grained profiling shows storage I/O and final host-side assembly remain major costs.
- CPU<->GPU transfer is now measured explicitly and is not always the dominant term.
- CPU-Time still includes orchestration and file output, not just compaction kernels.
- So the optimization helps, but end-to-end wall time is still strongly bounded by write/assembly overhead.


*****************************************************************************************************

************************Q-COMPACTION WITH PLAN***************************************************

KV Pairs: AAAA AAAA AAAB AAAC SYBS MFLK SLDJ SDNK
KV groups = (AAAA AAAA AAAB AAAC) (SYBS MFLK DLDJ XDNK)
with prefix compression = (AAAA _ B C)=10 (SYBS MFLK DLDJ XDNK)=16
KV group metadata = 10, 16, 
------------------------------------
TRANSFER KV group metadata = 10, 16,  to CPU

CPU DOES PLAN
----------------------------
GPU actually packs the KV groups into blocks based on plan
1 data block is max of 12 Bytes: [(AAAA _ B C)] [(SYBS MFLK DLDJ XDNK)]

*****************************************************************************************************


*************************C-COMPACTION = TODO**************************************************

KV Pairs: AAAA AAAA AAAB AAAC SYBS MFLK SLDJ SDNK
GC: AAAA AAAB AAAC SYBS MFLK SLDJ SDNK
KV groups = (AAAA AAAB AAAC SYBS) (MFLK DLDJ XDNK)
with prefix compression = (AAAA B C SYBS)=13 (MFLK DLDJ XDNK)=12
PLAN: KV group metadata = 13, 12 CPU algorihtm 
1 data block is max of 13 Bytes: [(AAAA B C SYBS)] [(MFLK DLDJ XDNK)]

GPComp: Data block size/uncompressed KV group size = N KV groups per data block
E: 4/4 = 1 KV group per data block
[(1 2)] [(3 4 5 6)] [(1 2 3)] [(1)] [(2 3 4)]
THIS ALLOWS YOU TO NOT PLAN ON CPU. YOU JUST STATICALLY ASSIGN KV GROUPS PER DATA BLOCK.
TODO: IMPLEMENT THIS NAIVE APPROACH
*****************************************************************************************************


PLAN KV groups into data blocks properly
- RocksDB/LevelDB this is done sequentially, one by one

if you do PLAN on GPU = VERY SLOW because sucks at sequential algorithm
SO GPCOMP JUST SKIPS PLAN ALTOGHET AND DOES STATIC KV GROUP ASSIGNMENT FOR PARALLEL DATA BlOCK PACKING

Solution:
- transfer KV group metadata (less data than entire KV groups) to CPU
- CPU does the planning of which KV groups to which data blocks
- transfer PLAN back to GPU
- GPU does packing of kv groups in parallel 

SSTables 
-> unpack into kv groups 
-> unpack needs to prefix decompress kv groups into kv pairs
-> sort kv pairs into global array
-> pack kv pairs into kv groups
-> pack kv groups into data blocks
-> pack data blocks + bloom blocks + index blocks into output SSTables

Q-Compaction:
Unpack(SSTables) = split up raw KV pairs
Sort(raw KV pairs) = Global array of sorted KV pairs
Plan(global array of sorted KV pairs) = a Plan of which KV groups go into which data blocks
Pack(global array of sorted KV pairs + plan) = pack KV pairs into KV groups and data blocks AND returns SSTable

C-COMPACTION:
Unpack(SSTables) = split up raw KV pairs
Sort(raw KV pairs) = Global array of sorted KV pairs
Plan(global array of sorted KV pairs) = a Plan of which KV groups go into which data blocks ON CPU
  ALSO we need to do garbage collection on CPU to remove deleted KV pairs and expired KV pairs
Pack(global array of sorted KV pairs + plan) = pack KV pairs into KV groups and data blocks AND returns SSTable

Input: A: 1 B: 2 C: 3 D: 4 E: 5 E': 4 F: 6
Garbage Collection: A: 1 B: 2 C: 3 D: 4 E: 5 F: 6
GC is SEQUENTIAL == SLOW ON GPU == MUST TRANSFER AND DO ON CPU

TODO: IMPLEMENT FAST GC
- figure out how to send information to and from CPU