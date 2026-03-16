# Q-Compaction vs C-Compaction

This repo implements the paper's compaction units, but the paper's terminology can be easy to misread:

- The paper's **Sort Unit** is implemented with **Algorithm 1**, which is a GPU **merge of multiple ordered arrays**.
- That means **"sort" in the paper is not separate from merge** for Q-compaction. The input SSTs are already individually sorted, so sorting is done by a multi-way merge.
- **Q-compaction** keeps all compaction units on the GPU:
  - unpack
  - sort (Algorithm 1 merge)
  - bloom generation
  - pack / SST generation
- **C-compaction** still uses the GPU for unpack and sort, but then transfers the sorted KV pairs back to the CPU for garbage collection before sending surviving KV pairs back to the GPU for SST generation.

## Paper-backed Q-compaction flow

`SSD -> Main Memory -> GPU -> Main Memory -> SSD`

1. CPU reads the picked SST files from SSD into host memory.
2. CUDA streams overlap `H2D` copies and `unpack` across SST files.
3. After unpack completes, the GPU runs the paper's Sort Unit.
   - This is Algorithm 1.
   - It is a parallel multi-way merge of already ordered SST arrays.
4. Q-compaction stays on the GPU after sorting.
   - no CPU garbage collection
   - no CPU-side merge stage
5. GPU generates bloom filters and packs new SST blocks.
6. The generated SST output is copied back to host memory.
7. CPU writes the new SST files to SSD.

## Paper-backed C-compaction flow

`SSD -> Main Memory -> GPU -> Main Memory -> GPU -> Main Memory -> SSD`

1. CPU reads SST files from SSD into host memory.
2. GPU unpacks and sorts the KV pairs.
3. Sorted KV pairs are transferred back to the CPU.
4. CPU performs garbage collection.
5. Surviving KV pairs are transferred back to the GPU.
6. GPU generates the new SST files.
7. CPU receives the output SST bytes and writes them to SSD.

## What Fig. 8 means

The `Merge&Garbage Collection in C-Compaction` line in Fig. 8 does **not** mean merge is only for C-compaction.

What it means is:

- Q-compaction: unpack -> sort/merge -> pack on GPU
- C-compaction: unpack -> sort/merge on GPU, then CPU garbage collection, then GPU pack

So the merge kernel belongs to the Sort Unit and is shared by both Q-compaction and C-compaction.
