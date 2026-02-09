# Project Proposal Summary

## Objective

The objective of this project is to study and experimentally evaluate
GPU-accelerated compaction techniques within the context of a modern LSM-tree
based key-value store.

Specifically, we aim to:
- Profile and understand CPU-based compaction behavior in RocksDB
- Compare single-threaded and multi-threaded CPU compaction performance
- Reproduce and re-evaluate evaluation metrics from prior GPU compaction work
- Identify which components of compaction benefit most from GPU acceleration

## Planned Phases

### Phase 1: Baseline CPU Compaction
- Build and configure RocksDB on a modern Linux system
- Run `db_bench` with write-heavy and mixed workloads
- Profile compaction behavior under:
  - single-threaded compaction
  - multi-threaded compaction
- Collect throughput, latency, and compaction statistics

### Phase 2: GPU Compaction Exploration
- Study prior GPU-compaction designs
- Identify candidate compaction stages suitable for GPU offload
- Prototype or simulate GPU-assisted compaction paths
- Compare against CPU baselines

### Phase 3: Evaluation & Analysis
- Analyze performance trade-offs
- Discuss architectural challenges
- Compare results with prior work

## Non-Goals

- Production deployment
- Full feature parity with RocksDB CPU compaction
- Upstream integration
