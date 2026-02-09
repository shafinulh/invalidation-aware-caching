# Related Work: GPU-Accelerated Compaction (GPComp)

This project is guided primarily by the paper:

> *GPU-Accelerated Compaction for LSM-based Key-Value Stores*  
> (referred to as **GPComp** throughout this repository)

## Key Ideas

The GPComp paper proposes offloading LSM-tree compaction to GPUs by exploiting:
- The high parallelism of GPU architectures
- The sort/merge nature of compaction workloads
- Batched processing of key-value pairs

The paper argues that GPU compaction can:
- Improve compaction throughput
- Reduce write amplification
- Lower CPU utilization
- Reduce write stalls during heavy ingestion

## System Overview

GPComp is implemented on top of a modified key-value store and focuses on:
- Sorting and merging SSTable contents on the GPU
- Overlapping CPU and GPU work
- Managing data transfer overheads between CPU and GPU

The system primarily targets **write-heavy workloads** where compaction is a
dominant cost.

## Reported Results

The paper reports:
- Significant speedups over CPU-only compaction
- Improved write throughput under sustained load
- Reduced compaction-induced stalls

However, the evaluation environment differs substantially from modern
production systems in terms of:
- Storage engine maturity
- Hardware configuration
- Compaction pipeline complexity