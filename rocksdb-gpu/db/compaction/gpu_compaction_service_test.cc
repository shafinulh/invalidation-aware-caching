// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#include "db/db_test_util.h"
#include "rocksdb/sst_file_reader.h"
#include "rocksdb/statistics.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/gpu_compaction_service.h"

#include <functional>

namespace ROCKSDB_NAMESPACE {
namespace {

void StoreBigEndian64(uint64_t value, char* out) {
  out[0] = static_cast<char>(value >> 56);
  out[1] = static_cast<char>(value >> 48);
  out[2] = static_cast<char>(value >> 40);
  out[3] = static_cast<char>(value >> 32);
  out[4] = static_cast<char>(value >> 24);
  out[5] = static_cast<char>(value >> 16);
  out[6] = static_cast<char>(value >> 8);
  out[7] = static_cast<char>(value);
}

std::string MakeUserKey(uint64_t user_key) {
  std::string key(8, '\0');
  StoreBigEndian64(user_key, &key[0]);
  return key;
}

std::string MakePaddedUserKey(uint64_t user_key) {
  std::string key(16, '0');
  StoreBigEndian64(user_key, &key[0]);
  return key;
}

std::string MakeValue(uint64_t generation, uint64_t user_key) {
  std::string value(32, '\0');
  StoreBigEndian64(generation, &value[0]);
  StoreBigEndian64(user_key, &value[8]);
  StoreBigEndian64(generation ^ 0x9e3779b97f4a7c15ULL, &value[16]);
  StoreBigEndian64(user_key ^ 0xc2b2ae3d27d4eb4fULL, &value[24]);
  return value;
}

Options MakeSstReaderOptions(const Options& base_options) {
  Options reader_options = base_options;
  BlockBasedTableOptions table_options;
  table_options.no_block_cache = true;
  reader_options.table_factory.reset(NewBlockBasedTableFactory(table_options));
  return reader_options;
}

}  // namespace

class GpuCompactionServiceTest : public DBTestBase {
 public:
  GpuCompactionServiceTest() : DBTestBase("gpu_compaction_service_test", true) {}

 protected:
  void ReopenWithGpuCompactionService(Options* options) {
    options->env = env_;
    options->statistics = CreateDBStatistics();
    primary_statistics_ = options->statistics;

    GpuCompactionServiceOptions service_options;
    ASSERT_OK(
        NewGpuCompactionService(*options, service_options, &compaction_service_));
    gpu_service_ =
        std::dynamic_pointer_cast<GpuCompactionService>(compaction_service_);
    ASSERT_NE(gpu_service_, nullptr);

    options->compaction_service = compaction_service_;
    DestroyAndReopen(*options);
  }

  void WriteFile(uint64_t begin, uint64_t end, uint64_t generation,
                 std::map<uint64_t, std::string>* expected) {
    for (uint64_t key = begin; key < end; ++key) {
      const std::string user_key = MakeUserKey(key);
      const std::string value = MakeValue(generation, key);
      ASSERT_OK(db_->Put(WriteOptions(), user_key, value));
      (*expected)[key] = value;
    }
    ASSERT_OK(db_->Flush(FlushOptions()));
  }

  void WriteFileWithUserKeyMaker(
      uint64_t begin, uint64_t end, uint64_t generation,
      const std::function<std::string(uint64_t)>& make_user_key,
      std::map<uint64_t, std::string>* expected) {
    for (uint64_t key = begin; key < end; ++key) {
      const std::string user_key = make_user_key(key);
      const std::string value = MakeValue(generation, key);
      ASSERT_OK(db_->Put(WriteOptions(), user_key, value));
      (*expected)[key] = value;
    }
    ASSERT_OK(db_->Flush(FlushOptions()));
  }

  void AssertReads(const std::map<uint64_t, std::string>& expected) {
    for (const auto& entry : expected) {
      std::string value;
      ASSERT_OK(db_->Get(ReadOptions(), MakeUserKey(entry.first), &value));
      ASSERT_EQ(entry.second, value);
    }
  }

  void AssertReadsWithUserKeyMaker(
      const std::map<uint64_t, std::string>& expected,
      const std::function<std::string(uint64_t)>& make_user_key) {
    for (const auto& entry : expected) {
      std::string value;
      ASSERT_OK(db_->Get(ReadOptions(), make_user_key(entry.first), &value));
      ASSERT_EQ(entry.second, value);
    }
  }

  void VerifyLiveSstFiles(const Options& options) {
    const Options reader_options = MakeSstReaderOptions(options);
    std::vector<LiveFileMetaData> live_files;
    db_->GetLiveFilesMetaData(&live_files);
    ASSERT_FALSE(live_files.empty());

    for (const auto& file : live_files) {
      SstFileReader reader(reader_options);
      ASSERT_OK(reader.Open(file.db_path + file.name));
      ASSERT_OK(reader.VerifyChecksum(ReadOptions()));
      ASSERT_OK(reader.VerifyNumEntries(ReadOptions()));
    }
  }

  size_t CountFilesAtLevel(int level) {
    std::vector<LiveFileMetaData> live_files;
    db_->GetLiveFilesMetaData(&live_files);
    return static_cast<size_t>(std::count_if(
        live_files.begin(), live_files.end(),
        [level](const LiveFileMetaData& file) { return file.level == level; }));
  }

  std::shared_ptr<Statistics> primary_statistics_;
  std::shared_ptr<CompactionService> compaction_service_;
  std::shared_ptr<GpuCompactionService> gpu_service_;
};

TEST_F(GpuCompactionServiceTest, EndToEndGpuL0L1ThenCpuL1L2) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.num_levels = 3;
  options.max_subcompactions = 1;
  options.level_compaction_dynamic_level_bytes = false;
  options.target_file_size_base = 256 * 1024;
  options.compression = CompressionType::kNoCompression;
  options.compaction_verify_record_count = false;

  BlockBasedTableOptions table_options;
  table_options.block_restart_interval = 4;
  table_options.block_size = 32 * 1024;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  ReopenWithGpuCompactionService(&options);

  std::map<uint64_t, std::string> expected;
  WriteFile(0, 128, 1, &expected);
  WriteFile(64, 192, 2, &expected);
  WriteFile(32, 160, 3, &expected);
  ASSERT_EQ(CountFilesAtLevel(0), 3U);
  ASSERT_EQ(CountFilesAtLevel(1), 0U);
  ASSERT_EQ(CountFilesAtLevel(2), 0U);

  ASSERT_OK(db_->CompactRange(CompactRangeOptions(), nullptr, nullptr));
  AssertReads(expected);
  GpuCompactionServiceCounters counters;
  ASSERT_OK(gpu_service_->GetCounters(&counters));
  ASSERT_EQ(counters.accepted_gpu_jobs, 1);
  ASSERT_EQ(counters.completed_gpu_jobs, 1);
  ASSERT_EQ(counters.fallback_local_jobs, 0);
  ASSERT_EQ(counters.last_gpu_base_input_level, 0);
  ASSERT_EQ(counters.last_gpu_output_level, 1);
  ASSERT_EQ(CountFilesAtLevel(0), 0U);
  ASSERT_GT(CountFilesAtLevel(1), 0U);
  ASSERT_EQ(CountFilesAtLevel(2), 0U);
  VerifyLiveSstFiles(options);

  const uint64_t remote_write_after_first =
      primary_statistics_->getTickerCount(REMOTE_COMPACT_WRITE_BYTES);
  ASSERT_GT(remote_write_after_first, 0);

  WriteFile(16, 112, 4, &expected);
  WriteFile(96, 176, 5, &expected);
  ASSERT_EQ(CountFilesAtLevel(0), 2U);
  ASSERT_EQ(CountFilesAtLevel(1), 1U);
  ASSERT_EQ(CountFilesAtLevel(2), 0U);

  ASSERT_OK(db_->CompactRange(CompactRangeOptions(), nullptr, nullptr));
  AssertReads(expected);
  ASSERT_OK(gpu_service_->GetCounters(&counters));
  ASSERT_EQ(counters.accepted_gpu_jobs, 2);
  ASSERT_EQ(counters.completed_gpu_jobs, 2);
  ASSERT_EQ(counters.fallback_local_jobs, 0);
  ASSERT_EQ(counters.last_gpu_base_input_level, 0);
  ASSERT_EQ(counters.last_gpu_output_level, 1);
  ASSERT_EQ(CountFilesAtLevel(0), 0U);
  ASSERT_GT(CountFilesAtLevel(1), 0U);
  ASSERT_EQ(CountFilesAtLevel(2), 0U);
  VerifyLiveSstFiles(options);

  const uint64_t remote_write_before_cpu =
      primary_statistics_->getTickerCount(REMOTE_COMPACT_WRITE_BYTES);
  ASSERT_GT(remote_write_before_cpu, remote_write_after_first);

  CompactRangeOptions level2_options;
  level2_options.change_level = true;
  level2_options.target_level = 2;
  ASSERT_OK(db_->CompactRange(level2_options, nullptr, nullptr));
  AssertReads(expected);
  ASSERT_EQ(CountFilesAtLevel(0), 0U);
  ASSERT_EQ(CountFilesAtLevel(1), 0U);
  ASSERT_GT(CountFilesAtLevel(2), 0U);
  VerifyLiveSstFiles(options);

  ASSERT_OK(gpu_service_->GetCounters(&counters));
  ASSERT_EQ(counters.accepted_gpu_jobs, 2);
  ASSERT_EQ(counters.completed_gpu_jobs, 2);
  ASSERT_EQ(counters.last_gpu_base_input_level, 0);
  ASSERT_EQ(counters.last_gpu_output_level, 1);
  ASSERT_EQ(remote_write_before_cpu,
            primary_statistics_->getTickerCount(REMOTE_COMPACT_WRITE_BYTES));
}

TEST_F(GpuCompactionServiceTest, AcceptsDbBenchStylePadded16ByteUserKeys) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.num_levels = 3;
  options.max_subcompactions = 1;
  options.level_compaction_dynamic_level_bytes = false;
  options.target_file_size_base = 256 * 1024;
  options.compression = CompressionType::kNoCompression;
  options.compaction_verify_record_count = false;

  BlockBasedTableOptions table_options;
  table_options.block_restart_interval = 4;
  table_options.block_size = 32 * 1024;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  ReopenWithGpuCompactionService(&options);

  std::map<uint64_t, std::string> expected;
  WriteFileWithUserKeyMaker(0, 128, 1, MakePaddedUserKey, &expected);
  WriteFileWithUserKeyMaker(64, 192, 2, MakePaddedUserKey, &expected);
  ASSERT_EQ(CountFilesAtLevel(0), 2U);

  ASSERT_OK(db_->CompactRange(CompactRangeOptions(), nullptr, nullptr));
  AssertReadsWithUserKeyMaker(expected, MakePaddedUserKey);

  GpuCompactionServiceCounters counters;
  ASSERT_OK(gpu_service_->GetCounters(&counters));
  ASSERT_EQ(counters.accepted_gpu_jobs, 1);
  ASSERT_EQ(counters.completed_gpu_jobs, 1);
  ASSERT_EQ(counters.fallback_local_jobs, 0);
  ASSERT_EQ(counters.last_gpu_base_input_level, 0);
  ASSERT_EQ(counters.last_gpu_output_level, 1);
  ASSERT_EQ(CountFilesAtLevel(0), 0U);
  ASSERT_GT(CountFilesAtLevel(1), 0U);
  VerifyLiveSstFiles(options);
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
