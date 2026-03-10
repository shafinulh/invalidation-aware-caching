// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <cstdint>
#include "rocksdb/status.h"

// ─────────────────────────────────────────────────────────────────────────────
// GPU N-way merge for variable-length RocksDB internal keys.
//
// Algorithm: one GPU thread per input entry.  Each thread determines its
// output position by binary-searching all other sorted SST runs (Algorithm 1
// from the GPComp paper, lifted to variable-length byte-string keys with the
// RocksDB InternalKeyComparator ordering).
//
// Entry layout in the packed SoA:
//   - Keys are concatenated in h_keys[0..keys_bytes-1].
//   - h_key_offsets[i]..h_key_offsets[i+1] gives the byte range of entry i.
//   - Each SST run j occupies entries [h_run_offsets[j], h_run_offsets[j+1]).
//   - Entries within every run are already sorted (guaranteed by SST format).
//
// Output:
//   h_out_mapping[output_pos] = packed uint32_t:
//     bits [31:24] = source SST index j   (0-based, max 255 SSTs per compaction)
//     bits [23:0]  = local index within run j (max 16 M entries per SST)
//
// The caller uses this permutation to reorder key-value data into the
// final output SoA without a second GPU round-trip.
// ─────────────────────────────────────────────────────────────────────────────

namespace ROCKSDB_NAMESPACE {

// Launches the GPU merge kernel.  All host pointers must be CUDA pinned memory
// (cudaMallocHost) so that cudaMemcpyAsync can overlap H2D transfers with
// setup work.
//
// Parameters:
//   h_keys          – packed key bytes (pinned host)
//   h_key_offsets   – key byte-range offsets, length = total_entries + 1 (pinned)
//   total_entries   – sum of all run sizes
//   h_run_offsets   – SST run entry-index boundaries, length = num_runs + 1
//   num_runs        – number of sorted input runs (one per input SST file)
//   h_out_mapping   – caller-allocated host buffer (length = total_entries)
//                     filled with the sorted permutation on return
//
// Returns OK on success; non-OK if CUDA fails or CUDA is unavailable.
Status GpuVarlengthMerge(const char*     h_keys,
                          const uint32_t* h_key_offsets,
                          uint32_t        total_entries,
                          const uint32_t* h_run_offsets,
                          uint32_t        num_runs,
                          uint32_t*       h_out_mapping);

}  // namespace ROCKSDB_NAMESPACE
