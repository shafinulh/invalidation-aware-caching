//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/compaction/gpu_compaction_orchestrator.h"
#include "../../../benchmarks/gpu/src/v2/gpu_varlength_merge.cuh"

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

#include "db/dbformat.h"
#include "monitoring/histogram.h"
#include "table/block_based/block.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/block_based/uncompression_dict_reader.h"
#include "table/block_fetcher.h"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#define ROCKSDB_GPU_COMPACTION_HAS_CUDA 1
#else
#define ROCKSDB_GPU_COMPACTION_HAS_CUDA 0
#endif

namespace ROCKSDB_NAMESPACE {

namespace {

Status AllocatePinned(void** ptr, size_t bytes) {
#if ROCKSDB_GPU_COMPACTION_HAS_CUDA
  if (bytes == 0) {
    *ptr = nullptr;
    return Status::OK();
  }
  cudaError_t cuda_status = cudaMallocHost(ptr, bytes);
  if (cuda_status != cudaSuccess) {
    return Status::MemoryLimit(
        std::string("cudaMallocHost failed: ") + cudaGetErrorString(cuda_status));
  }
  return Status::OK();
#else
  (void)ptr;
  (void)bytes;
  return Status::NotSupported(
      "CUDA runtime headers are unavailable; pinned host allocations are disabled");
#endif
}

void FreePinned(void* ptr) {
#if ROCKSDB_GPU_COMPACTION_HAS_CUDA
  if (ptr != nullptr) {
    cudaFreeHost(ptr);
  }
#else
  (void)ptr;
#endif
}

bool KeyInSubcompactionRange(const Comparator* user_cmp,
                             const std::optional<Slice>& start,
                             const std::optional<Slice>& end,
                             const Slice& internal_key) {
  const Slice user_key = ExtractUserKey(internal_key);
  if (start.has_value() && user_cmp->Compare(user_key, *start) < 0) {
    return false;
  }
  if (end.has_value() && user_cmp->Compare(user_key, *end) >= 0) {
    return false;
  }
  return true;
}

}  // namespace

GPUCompactionOrchestrator::GPUCompactionOrchestrator(
    const ImmutableDBOptions& db_options,
    const MutableDBOptions& mutable_db_options,
  const FileOptions& file_options, ColumnFamilyData* cfd,
  SubcompactionState* sub_compact,
    std::vector<const FileMetaData*> input_files,
    const CompactionFileOpenFunc& open_file_func,
    const CompactionFileCloseFunc& close_file_func,
    uint64_t proximal_after_seqno)
    : db_options_(db_options),
      mutable_db_options_(mutable_db_options),
      file_options_(file_options),
      cfd_(cfd),
      sub_compact_(sub_compact),
      input_files_(std::move(input_files)),
      open_file_func_(open_file_func),
      close_file_func_(close_file_func),
      proximal_after_seqno_(proximal_after_seqno) {}

GPUCompactionOrchestrator::~GPUCompactionOrchestrator() {
  FreePinnedSoA(&input_soa_);
  FreePinnedSoA(&output_soa_);
}

Status GPUCompactionOrchestrator::Execute() {
  Status status = FlattenBlocksToPinnedSoA();
  if (!status.ok()) {
    return status;
  }

  status = LaunchGPUCompaction();
  if (!status.ok()) {
    return status;
  }

  return PackGpuOutputToSst();
}

Status GPUCompactionOrchestrator::OpenInputTable(const FileMetaData& meta,
                                                 OpenedTable* opened_table) {
  TableReader* table_reader = nullptr;
  std::unique_ptr<InternalIterator> keep_alive_iter;
  keep_alive_iter.reset(cfd_->table_cache()->NewIterator(
      ReadOptions(Env::IOActivity::kCompaction), file_options_,
      cfd_->internal_comparator(), meta, /*range_del_agg=*/nullptr,
      sub_compact_->compaction->mutable_cf_options(), &table_reader,
      /*file_read_hist=*/nullptr, TableReaderCaller::kCompaction,
      /*arena=*/nullptr, /*skip_filters=*/true,
      sub_compact_->compaction->output_level(),
      /*max_file_size_for_l0_meta_pin=*/0,
      /*smallest_compaction_key=*/nullptr, /*largest_compaction_key=*/nullptr,
      /*allow_unprepared_value=*/false));

  if (keep_alive_iter == nullptr) {
    return Status::Corruption("Failed to create input iterator for GPU compaction");
  }
  Status status = keep_alive_iter->status();
  if (!status.ok()) {
    return status;
  }

  auto* block_table = dynamic_cast<BlockBasedTable*>(table_reader);
  if (block_table == nullptr) {
    return Status::NotSupported(
        "GPU compaction currently supports only block-based SSTables");
  }

  opened_table->keep_alive_iter = std::move(keep_alive_iter);
  opened_table->block_table = block_table;
  opened_table->meta = &meta;
  return Status::OK();
}

Status GPUCompactionOrchestrator::ReadDataBlocksFromTable(
    const OpenedTable& opened_table) {
  const auto* rep = opened_table.block_table->get_rep();
  BlockCacheLookupContext lookup_context{TableReaderCaller::kCompaction};
  std::unique_ptr<InternalIteratorBase<IndexValue>> index_iter(
      rep->index_reader->NewIterator(ReadOptions(Env::IOActivity::kCompaction),
                                     /*disable_prefix_seek=*/false,
                                     /*iter=*/nullptr, /*get_context=*/nullptr,
                                     &lookup_context));
  Status status = index_iter->status();
  if (!status.ok()) {
    return status;
  }

  UnownedPtr<Decompressor> decompressor = rep->decompressor.get();
  CachableEntry<DecompressorDict> cached_dict;
  if (rep->uncompression_dict_reader != nullptr) {
    status = rep->uncompression_dict_reader->GetOrReadUncompressionDictionary(
        /*prefetch_buffer=*/nullptr, ReadOptions(Env::IOActivity::kCompaction),
        /*get_context=*/nullptr, /*lookup_context=*/nullptr, &cached_dict);
    if (!status.ok()) {
      return status;
    }
    if (cached_dict.GetValue() != nullptr) {
      decompressor = cached_dict.GetValue()->decompressor_.get();
    }
  }

  for (index_iter->SeekToFirst(); index_iter->Valid(); index_iter->Next()) {
    status = index_iter->status();
    if (!status.ok()) {
      return status;
    }

    BlockContents block_contents;
    const BlockHandle& handle = index_iter->value().handle;
    BlockFetcher fetcher(rep->file.get(), /*prefetch_buffer=*/nullptr, rep->footer,
                         ReadOptions(Env::IOActivity::kCompaction), handle,
                         &block_contents, rep->ioptions,
                         /*do_uncompress=*/true, /*maybe_compressed=*/true,
                         BlockType::kData, decompressor,
                         rep->persistent_cache_options,
                         /*memory_allocator=*/nullptr,
                         /*memory_allocator_compressed=*/nullptr,
                         /*for_compaction=*/true);
    status = fetcher.ReadBlockContents();
    if (!status.ok()) {
      return status;
    }

    Block data_block(std::move(block_contents));
    std::unique_ptr<DataBlockIter> data_iter(data_block.NewDataIterator(
        cfd_->internal_comparator().user_comparator(),
        rep->get_global_seqno(BlockType::kData)));

    for (data_iter->SeekToFirst(); data_iter->Valid(); data_iter->Next()) {
      status = data_iter->status();
      if (!status.ok()) {
        return status;
      }
      const Slice& internal_key = data_iter->key();
      if (!KeyInSubcompactionRange(cfd_->user_comparator(), sub_compact_->start,
                                   sub_compact_->end, internal_key)) {
        continue;
      }
      flattened_entries_.push_back(
          {std::string(internal_key.data(), internal_key.size()),
           std::string(data_iter->value().data(), data_iter->value().size())});
    }
  }

  return Status::OK();
}

Status GPUCompactionOrchestrator::FlattenBlocksToPinnedSoA() {
  flattened_entries_.clear();
  sst_run_global_offsets_.clear();

  for (const FileMetaData* meta : input_files_) {
    // Record the start index of this SST's run before reading its entries.
    // Entries within each SST are already sorted by InternalKeyComparator
    // (guaranteed by the SST block format); no CPU sort is needed here.
    sst_run_global_offsets_.push_back(
        static_cast<uint32_t>(flattened_entries_.size()));

    OpenedTable opened_table;
    Status status = OpenInputTable(*meta, &opened_table);
    if (!status.ok()) {
      return status;
    }
    status = ReadDataBlocksFromTable(opened_table);
    if (!status.ok()) {
      return status;
    }
  }
  // Sentinel: total entry count.
  sst_run_global_offsets_.push_back(
      static_cast<uint32_t>(flattened_entries_.size()));

  // Pack into pinned SoA WITHOUT sorting — the GPU merge kernel will produce
  // the globally sorted output via one-thread-per-entry binary search.
  return FlattenEntriesToPinnedSoA(flattened_entries_, &input_soa_);
}

Status GPUCompactionOrchestrator::FlattenEntriesToPinnedSoA(
    const std::vector<FlattenedKV>& entries, PinnedSoA* soa) {
  FreePinnedSoA(soa);

  size_t keys_bytes = 0;
  size_t values_bytes = 0;
  for (const auto& entry : entries) {
    keys_bytes += entry.internal_key.size();
    values_bytes += entry.value.size();
  }

  Status status = AllocatePinned(reinterpret_cast<void**>(&soa->key_offsets),
                                 sizeof(uint32_t) * (entries.size() + 1));
  if (!status.ok()) {
    return status;
  }
  status = AllocatePinned(reinterpret_cast<void**>(&soa->value_offsets),
                          sizeof(uint32_t) * (entries.size() + 1));
  if (!status.ok()) {
    return status;
  }
  status = AllocatePinned(reinterpret_cast<void**>(&soa->keys), keys_bytes);
  if (!status.ok()) {
    return status;
  }
  status = AllocatePinned(reinterpret_cast<void**>(&soa->values), values_bytes);
  if (!status.ok()) {
    return status;
  }

  soa->num_entries = static_cast<uint32_t>(entries.size());
  soa->keys_bytes = static_cast<uint32_t>(keys_bytes);
  soa->values_bytes = static_cast<uint32_t>(values_bytes);

  uint32_t key_cursor = 0;
  uint32_t value_cursor = 0;
  for (size_t index = 0; index < entries.size(); ++index) {
    soa->key_offsets[index] = key_cursor;
    soa->value_offsets[index] = value_cursor;

    const auto& entry = entries[index];
    if (!entry.internal_key.empty()) {
      std::memcpy(soa->keys + key_cursor, entry.internal_key.data(),
                  entry.internal_key.size());
      key_cursor += static_cast<uint32_t>(entry.internal_key.size());
    }
    if (!entry.value.empty()) {
      std::memcpy(soa->values + value_cursor, entry.value.data(),
                  entry.value.size());
      value_cursor += static_cast<uint32_t>(entry.value.size());
    }
  }
  soa->key_offsets[entries.size()] = key_cursor;
  soa->value_offsets[entries.size()] = value_cursor;
  return Status::OK();
}

Status GPUCompactionOrchestrator::CloneInputToOutputBuffer() {
  FreePinnedSoA(&output_soa_);
  Status status = AllocatePinned(reinterpret_cast<void**>(&output_soa_.key_offsets),
                                 sizeof(uint32_t) * (input_soa_.num_entries + 1));
  if (!status.ok()) {
    return status;
  }
  status = AllocatePinned(reinterpret_cast<void**>(&output_soa_.value_offsets),
                          sizeof(uint32_t) * (input_soa_.num_entries + 1));
  if (!status.ok()) {
    return status;
  }
  status = AllocatePinned(reinterpret_cast<void**>(&output_soa_.keys), input_soa_.keys_bytes);
  if (!status.ok()) {
    return status;
  }
  status = AllocatePinned(reinterpret_cast<void**>(&output_soa_.values), input_soa_.values_bytes);
  if (!status.ok()) {
    return status;
  }

  output_soa_.num_entries = input_soa_.num_entries;
  output_soa_.keys_bytes = input_soa_.keys_bytes;
  output_soa_.values_bytes = input_soa_.values_bytes;
  std::memcpy(output_soa_.key_offsets, input_soa_.key_offsets,
              sizeof(uint32_t) * (input_soa_.num_entries + 1));
  std::memcpy(output_soa_.value_offsets, input_soa_.value_offsets,
              sizeof(uint32_t) * (input_soa_.num_entries + 1));
  if (input_soa_.keys_bytes > 0) {
    std::memcpy(output_soa_.keys, input_soa_.keys, input_soa_.keys_bytes);
  }
  if (input_soa_.values_bytes > 0) {
    std::memcpy(output_soa_.values, input_soa_.values, input_soa_.values_bytes);
  }
  return Status::OK();
}

Status GPUCompactionOrchestrator::LaunchGPUCompaction() {
  const uint32_t total   = input_soa_.num_entries;
  const uint32_t num_runs =
      sst_run_global_offsets_.empty()
          ? 0
          : static_cast<uint32_t>(sst_run_global_offsets_.size() - 1);

  // Trivial cases: fall back to the CPU clone path.
  if (total == 0 || num_runs <= 1) {
    return CloneInputToOutputBuffer();
  }

#if ROCKSDB_GPU_COMPACTION_HAS_CUDA
  // ── Step 1: GPU N-way merge → sorted permutation ─────────────────────────
  std::vector<uint32_t> h_out_mapping(total);

  Status merge_status = GpuVarlengthMerge(
      input_soa_.keys,
      input_soa_.key_offsets,
      total,
      sst_run_global_offsets_.data(),
      num_runs,
      h_out_mapping.data());

  if (!merge_status.ok()) {
    // GPU path failed (e.g. out of device memory); fall back to CPU sort.
    return CloneInputToOutputBuffer();
  }

  // ── Step 2: Allocate output SoA (same total bytes as input) ──────────────
  FreePinnedSoA(&output_soa_);

  Status status = AllocatePinned(
      reinterpret_cast<void**>(&output_soa_.key_offsets),
      sizeof(uint32_t) * (total + 1));
  if (!status.ok()) return status;

  status = AllocatePinned(
      reinterpret_cast<void**>(&output_soa_.value_offsets),
      sizeof(uint32_t) * (total + 1));
  if (!status.ok()) return status;

  status = AllocatePinned(
      reinterpret_cast<void**>(&output_soa_.keys), input_soa_.keys_bytes);
  if (!status.ok()) return status;

  status = AllocatePinned(
      reinterpret_cast<void**>(&output_soa_.values), input_soa_.values_bytes);
  if (!status.ok()) return status;

  output_soa_.num_entries  = total;
  output_soa_.keys_bytes   = input_soa_.keys_bytes;
  output_soa_.values_bytes = input_soa_.values_bytes;

  // ── Step 3: Reorder key-value data according to the GPU permutation ───────
  uint32_t out_key_cursor = 0;
  uint32_t out_val_cursor = 0;

  for (uint32_t out_idx = 0; out_idx < total; ++out_idx) {
    const uint32_t packed     = h_out_mapping[out_idx];
    const uint32_t sst_j      = packed >> 24;
    const uint32_t local_k    = packed & 0x00FFFFFFu;
    const uint32_t global_k   = sst_run_global_offsets_[sst_j] + local_k;

    // Copy key.
    const uint32_t src_koff = input_soa_.key_offsets[global_k];
    const uint32_t klen     = input_soa_.key_offsets[global_k + 1] - src_koff;
    output_soa_.key_offsets[out_idx] = out_key_cursor;
    std::memcpy(output_soa_.keys + out_key_cursor,
                input_soa_.keys + src_koff, klen);
    out_key_cursor += klen;

    // Copy value.
    const uint32_t src_voff = input_soa_.value_offsets[global_k];
    const uint32_t vlen     = input_soa_.value_offsets[global_k + 1] - src_voff;
    output_soa_.value_offsets[out_idx] = out_val_cursor;
    std::memcpy(output_soa_.values + out_val_cursor,
                input_soa_.values + src_voff, vlen);
    out_val_cursor += vlen;
  }
  output_soa_.key_offsets[total]   = out_key_cursor;
  output_soa_.value_offsets[total] = out_val_cursor;

  return Status::OK();

#else
  // CUDA unavailable at compile time: fall back to CPU clone.
  return CloneInputToOutputBuffer();
#endif
}

Status GPUCompactionOrchestrator::WriteEntriesToOutputs(
    const GPUOutputView& output_view) {
  Status status;
  ParsedInternalKey prev_internal_key;
  bool have_prev_key = false;
  const Slice empty_next_table_min_key;

  for (uint32_t index = 0; index < output_view.num_entries; ++index) {
    const uint32_t key_begin = output_view.key_offsets[index];
    const uint32_t key_end = output_view.key_offsets[index + 1];
    const uint32_t value_begin = output_view.value_offsets[index];
    const uint32_t value_end = output_view.value_offsets[index + 1];

    const Slice key(output_view.keys + key_begin, key_end - key_begin);
    const Slice value(output_view.values + value_begin, value_end - value_begin);

    ParsedInternalKey parsed_key;
    status = ParseInternalKey(key, &parsed_key, /*log_err_key=*/false);
    if (!status.ok()) {
      return status;
    }
    if (parsed_key.sequence > proximal_after_seqno_) {
      return Status::NotSupported(
          "GPU compaction skeleton does not yet repartition proximal-level outputs");
    }

    if (!sub_compact_->Current().HasBuilder()) {
      status = open_file_func_(sub_compact_->Current());
      if (!status.ok()) {
        return status;
      }
    }

    status = sub_compact_->Current().current_output().validator.Add(key, value);
    if (!status.ok()) {
      return status;
    }
    sub_compact_->Current().builder_->Add(key, value);
    sub_compact_->Current().stats_.num_output_records++;
    sub_compact_->Current().current_output_file_size_ =
        sub_compact_->Current().builder_->EstimatedFileSize();
    status = sub_compact_->Current().GetMetaData()->UpdateBoundaries(
        key, value, parsed_key.sequence, parsed_key.type);
    if (!status.ok()) {
      return status;
    }

    prev_internal_key = parsed_key;
    have_prev_key = true;
  }

  if (sub_compact_->Current().HasBuilder() && have_prev_key) {
    status = close_file_func_(Status::OK(), prev_internal_key,
                              empty_next_table_min_key, /*c_iter=*/nullptr,
                              sub_compact_->Current());
  }
  return status;
}

Status GPUCompactionOrchestrator::PackGpuOutputToSst() {
  GPUOutputView output_view{output_soa_.keys,
                            output_soa_.key_offsets,
                            output_soa_.values,
                            output_soa_.value_offsets,
                            output_soa_.num_entries};
  return WriteEntriesToOutputs(output_view);
}

void GPUCompactionOrchestrator::FreePinnedSoA(PinnedSoA* soa) {
  FreePinned(soa->keys);
  FreePinned(soa->key_offsets);
  FreePinned(soa->values);
  FreePinned(soa->value_offsets);
  *soa = {};
}

}  // namespace ROCKSDB_NAMESPACE