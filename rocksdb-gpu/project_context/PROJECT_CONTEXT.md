# Project Context

This repository contains the implementation and evaluation artifacts for a
course project exploring **GPU-accelerated compaction in RocksDB**.

Modern key-value stores such as RocksDB rely heavily on background compaction
to maintain read performance and space efficiency. While compaction is
traditionally CPU-bound, recent work has proposed offloading parts of the
compaction pipeline to GPUs to improve throughput and reduce write stalls.

This project is motivated by two observations:

1. **Compaction is a dominant background cost** in write-heavy workloads.
2. **GPUs offer massive parallelism** that may be well-suited to the sort/merge
   operations at the heart of LSM-tree compaction.

The goals of this project are:
- To establish a **clean CPU-only compaction baseline** in RocksDB
- To re-evaluate claims made by prior GPU-compaction work
- To explore the feasibility and limitations of GPU-assisted compaction in a
  production-grade storage engine

This work is strictly experimental and intended for academic evaluation only.
It is not intended for upstream contribution to RocksDB.
