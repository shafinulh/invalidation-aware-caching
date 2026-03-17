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


************************Q-COMPACTION WITHOUT PLAN = TODO***************************************************

KV Pairs: AAAA AAAA AAAB AAAC SYBS MFLK SLDJ SDNK
KV groups = (AAAA AAAA AAAB AAAC) (SYBS MFLK DLDJ XDNK)
with prefix compression = (AAAA _ B C)=10 (SYBS MFLK DLDJ XDNK)=16
PLAN = Datablock size/(KV pair size * KV group len) = 1
GPU can now split up KV groups statically into data blocks without needing to do any planning on CPU
1 data block is max of 12 Bytes: [(AAAA _ B C)] [(SYBS MFLK DLDJ XDNK)]

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