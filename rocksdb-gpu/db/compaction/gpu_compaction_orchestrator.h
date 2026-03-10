//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "db/column_family.h"
#include "db/compaction/compaction_outputs.h"
#include "db/compaction/subcompaction_state.h"
#include "db/table_cache.h"
#include "options/db_options.h"

namespace ROCKSDB_NAMESPACE {

class BlockBasedTable;

class GPUCompactionOrchestrator {
 public:
  struct PinnedSoA {
    char* keys = nullptr;
    uint32_t* key_offsets = nullptr;
    char* values = nullptr;
    uint32_t* value_offsets = nullptr;
    uint32_t num_entries = 0;
    uint32_t keys_bytes = 0;
    uint32_t values_bytes = 0;
  };

  struct GPUOutputView {
    const char* keys = nullptr;
    const uint32_t* key_offsets = nullptr;
    const char* values = nullptr;
    const uint32_t* value_offsets = nullptr;
    uint32_t num_entries = 0;
  };

  GPUCompactionOrchestrator(
      const ImmutableDBOptions& db_options,
      const MutableDBOptions& mutable_db_options,
      const FileOptions& file_options, ColumnFamilyData* cfd,
      SubcompactionState* sub_compact,
      std::vector<const FileMetaData*> input_files,
      const CompactionFileOpenFunc& open_file_func,
      const CompactionFileCloseFunc& close_file_func,
      uint64_t proximal_after_seqno);
  ~GPUCompactionOrchestrator();

  GPUCompactionOrchestrator(const GPUCompactionOrchestrator&) = delete;
  GPUCompactionOrchestrator& operator=(const GPUCompactionOrchestrator&) = delete;

  Status Execute();
  Status FlattenBlocksToPinnedSoA();
  Status LaunchGPUCompaction();
  Status PackGpuOutputToSst();

  const PinnedSoA& input_soa() const { return input_soa_; }
  const PinnedSoA& output_soa() const { return output_soa_; }

 private:
  struct FlattenedKV {
    std::string internal_key;
    std::string value;
  };

  struct OpenedTable {
    std::unique_ptr<InternalIterator> keep_alive_iter;
    BlockBasedTable* block_table = nullptr;
    const FileMetaData* meta = nullptr;
  };

  Status OpenInputTable(const FileMetaData& meta, OpenedTable* opened_table);
  Status ReadDataBlocksFromTable(const OpenedTable& opened_table);
  Status FlattenEntriesToPinnedSoA(const std::vector<FlattenedKV>& entries,
                                   PinnedSoA* soa);
  Status CloneInputToOutputBuffer();
  Status WriteEntriesToOutputs(const GPUOutputView& output_view);
  void FreePinnedSoA(PinnedSoA* soa);

  const ImmutableDBOptions& db_options_;
  const MutableDBOptions& mutable_db_options_;
  const FileOptions& file_options_;
  ColumnFamilyData* cfd_;
  SubcompactionState* sub_compact_;
  std::vector<const FileMetaData*> input_files_;
  const CompactionFileOpenFunc& open_file_func_;
  const CompactionFileCloseFunc& close_file_func_;
  uint64_t proximal_after_seqno_;

  std::vector<FlattenedKV> flattened_entries_;
  // Global entry-index boundaries for each input SST run.
  // sst_run_global_offsets_[j]     = first entry index of SST j in the flat SoA.
  // sst_run_global_offsets_[j + 1] = first entry index of SST j+1 (= last + 1).
  // Length = num_input_ssts + 1.  Populated by FlattenBlocksToPinnedSoA before
  // the CPU sort is removed, so each run [j, j+1) is pre-sorted by SST format.
  std::vector<uint32_t> sst_run_global_offsets_;
  PinnedSoA input_soa_;
  PinnedSoA output_soa_;
};

}  // namespace ROCKSDB_NAMESPACE