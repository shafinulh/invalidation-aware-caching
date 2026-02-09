# Evaluation Plan

This project re-evaluates selected claims from the GPComp paper using RocksDB
as the baseline storage engine.

## Baseline Metrics

We first establish CPU-only baselines using RocksDB:
- Write throughput
- Read latency (where applicable)
- Compaction time
- Bytes read/written during compaction
- Write stall frequency

These measurements are collected using `db_bench` under controlled workloads.

## CPU Compaction Experiments

We evaluate:
- Single-threaded compaction
- Multi-threaded compaction

Key questions:
- How does compaction scale with CPU threads?
- Where does CPU compaction bottleneck?
- How much parallelism is already exploited?

## GPU Compaction Comparison

Using the GPComp paper as guidance, we focus on:
- Workloads similar to those used in the paper
- Comparable metrics (throughput, compaction cost)
- Qualitative comparison of trends rather than absolute numbers

We explicitly account for:
- Differences in storage engine design
- GPU data transfer overheads
- Changes in compaction architecture

## Limitations

This evaluation does not attempt to:
- Reproduce exact performance numbers from GPComp
- Match hardware configurations exactly
- Fully re-implement GPComp inside RocksDB

Instead, the goal is to understand **whether the claimed benefits still hold
in a modern storage engine**.
