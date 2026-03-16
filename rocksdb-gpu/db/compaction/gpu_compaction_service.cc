// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#include "rocksdb/utilities/gpu_compaction_service.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "db/builder.h"
#include "db/compaction/compaction_job.h"
#include "db/dbformat.h"
#include "file/file_util.h"
#include "file/filename.h"
#include "file/read_write_util.h"
#include "file/writable_file_writer.h"
#include "monitoring/histogram.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/sst_file_reader.h"
#include "table/table_builder.h"
#include "gpcomp_rocksdb_bridge.h"

namespace ROCKSDB_NAMESPACE {
namespace {

using BridgeHandle = gpcomp_rocksdb_bridge::Handle;
using BridgeInputRun = gpcomp_rocksdb_bridge::InputRun;
using BridgeMergeResult = gpcomp_rocksdb_bridge::MergeResult;
using BridgeOptions = gpcomp_rocksdb_bridge::Options;
using BridgeRecord = gpcomp_rocksdb_bridge::Record;

constexpr std::size_t kGpuUserKeyBytes = 8;
constexpr std::size_t kGpuInternalKeyBytes = 16;
constexpr std::size_t kGpuPaddedInternalKeyBytes = 24;
constexpr std::size_t kGpuUserKeySuffixBytes = 8;
constexpr std::size_t kGpuValueBytes = 32;
constexpr uint64_t kGpuOutputEpochNumber = 1;

std::string JoinPath(const std::string& dir, const std::string& child) {
  if (dir.empty()) {
    return child;
  }
  if (child.empty()) {
    return dir;
  }
  if (dir.back() == '/') {
    return dir + child;
  }
  return dir + "/" + child;
}

void StoreBigEndian64(uint64_t value, uint8_t* out) {
  out[0] = static_cast<uint8_t>(value >> 56);
  out[1] = static_cast<uint8_t>(value >> 48);
  out[2] = static_cast<uint8_t>(value >> 40);
  out[3] = static_cast<uint8_t>(value >> 32);
  out[4] = static_cast<uint8_t>(value >> 24);
  out[5] = static_cast<uint8_t>(value >> 16);
  out[6] = static_cast<uint8_t>(value >> 8);
  out[7] = static_cast<uint8_t>(value);
}

uint64_t LoadBigEndian64(const uint8_t* in) {
  return (static_cast<uint64_t>(in[0]) << 56) |
         (static_cast<uint64_t>(in[1]) << 48) |
         (static_cast<uint64_t>(in[2]) << 40) |
         (static_cast<uint64_t>(in[3]) << 32) |
         (static_cast<uint64_t>(in[4]) << 24) |
         (static_cast<uint64_t>(in[5]) << 16) |
         (static_cast<uint64_t>(in[6]) << 8) |
         static_cast<uint64_t>(in[7]);
}

bool SameUserKey(const BridgeRecord& lhs, const BridgeRecord& rhs) {
  return std::memcmp(lhs.key, rhs.key, kGpuUserKeyBytes) == 0;
}

struct KeyEncoding {
  std::size_t internal_key_bytes = 0;
  bool has_constant_user_key_suffix = false;
  std::array<uint8_t, kGpuUserKeySuffixBytes> constant_user_key_suffix{};
};

Status ValidateKeyEncoding(const Slice& internal_key, KeyEncoding* encoding) {
  if (encoding == nullptr) {
    return Status::InvalidArgument("key encoding output pointer is null");
  }

  if (internal_key.size() != kGpuInternalKeyBytes &&
      internal_key.size() != kGpuPaddedInternalKeyBytes) {
    return Status::NotSupported(
        "GpuCompactionService requires 8-byte or padded 16-byte user keys");
  }

  if (encoding->internal_key_bytes == 0) {
    encoding->internal_key_bytes = internal_key.size();
  } else if (encoding->internal_key_bytes != internal_key.size()) {
    return Status::NotSupported(
        "GpuCompactionService requires a consistent internal key size");
  }

  if (internal_key.size() == kGpuPaddedInternalKeyBytes) {
    std::array<uint8_t, kGpuUserKeySuffixBytes> suffix{};
    std::memcpy(suffix.data(), internal_key.data() + kGpuUserKeyBytes,
                suffix.size());
    if (!encoding->has_constant_user_key_suffix) {
      encoding->constant_user_key_suffix = suffix;
      encoding->has_constant_user_key_suffix = true;
    } else if (encoding->constant_user_key_suffix != suffix) {
      return Status::NotSupported(
          "GpuCompactionService requires padded user-key suffix bytes to be "
          "constant across the job");
    }
  }

  return Status::OK();
}

Status EncodeCanonicalKey(const Slice& internal_key, const KeyEncoding& encoding,
                          uint8_t out[16]) {
  if (internal_key.size() != encoding.internal_key_bytes) {
    return Status::Corruption(
        "internal key size does not match the detected key encoding");
  }
  std::memcpy(out, internal_key.data(), kGpuUserKeyBytes);
  const uint64_t footer = ExtractInternalKeyFooter(internal_key);
  StoreBigEndian64(~footer, out + kGpuUserKeyBytes);
  return Status::OK();
}

std::string DecodeCanonicalKey(const uint8_t key[16],
                               const KeyEncoding& encoding) {
  std::string internal_key(encoding.internal_key_bytes, '\0');
  std::memcpy(&internal_key[0], key, kGpuUserKeyBytes);
  if (encoding.has_constant_user_key_suffix) {
    std::memcpy(&internal_key[kGpuUserKeyBytes],
                encoding.constant_user_key_suffix.data(),
                encoding.constant_user_key_suffix.size());
  }
  EncodeFixed64(&internal_key[encoding.internal_key_bytes - kNumInternalBytes],
                ~LoadBigEndian64(key + kGpuUserKeyBytes));
  return internal_key;
}

std::string ExtractKeyPrefix(const std::string& internal_key) {
  return std::string(internal_key.data(),
                     std::min<std::size_t>(CompactionJobStats::kMaxPrefixLength,
                                           internal_key.size() -
                                               kNumInternalBytes));
}

Status MapBridgeStatus(const std::string& bridge_status) {
  if (bridge_status.empty()) {
    return Status::OK();
  }
  return Status::Incomplete(bridge_status);
}

Options MakeSstReaderOptions(const Options& base_options) {
  Options reader_options = base_options;
  BlockBasedTableOptions block_based_options;
  block_based_options.no_block_cache = true;
  reader_options.table_factory.reset(
      NewBlockBasedTableFactory(block_based_options));
  return reader_options;
}

}  // namespace

class GpuCompactionServiceImpl final : public GpuCompactionService {
 public:
  GpuCompactionServiceImpl(const Options& db_options,
                           const GpuCompactionServiceOptions& service_options,
                           BridgeHandle* gpu_handle)
      : options_(db_options),
        service_options_(service_options),
        env_options_(db_options),
        immutable_options_(db_options),
        mutable_cf_options_(ColumnFamilyOptions(db_options)),
        gpu_handle_(gpu_handle),
        service_id_(db_options.env->GenerateUniqueId()) {}

  ~GpuCompactionServiceImpl() override {
    gpcomp_rocksdb_bridge::Close(gpu_handle_);
    gpu_handle_ = nullptr;
  }

  static const char* kClassName() { return "GpuCompactionService"; }

  const char* Name() const override { return kClassName(); }

  CompactionServiceScheduleResponse Schedule(
      const CompactionServiceJobInfo& info,
      const std::string& compaction_service_input) override {
    if (canceled_.load()) {
      return CompactionServiceScheduleResponse(
          CompactionServiceJobStatus::kAborted);
    }

    CompactionServiceInput input;
    Status s = CompactionServiceInput::Read(compaction_service_input, &input);
    if (!s.ok()) {
      return CompactionServiceScheduleResponse(
          CompactionServiceJobStatus::kFailure);
    }
    if (!IsEligible(info, input)) {
      fallback_local_jobs_.fetch_add(1);
      return CompactionServiceScheduleResponse(
          CompactionServiceJobStatus::kUseLocal);
    }
    s = EnsureScratchRoot(info.db_name);
    if (!s.ok()) {
      return CompactionServiceScheduleResponse(
          CompactionServiceJobStatus::kFailure);
    }

    JobState job;
    job.info = info;
    job.input = std::move(input);

    const std::string job_id =
        "gpu-compaction-job-" +
        std::to_string(next_job_id_.fetch_add(1, std::memory_order_relaxed));
    {
      std::lock_guard<std::mutex> lock(mu_);
      jobs_.emplace(job_id, std::move(job));
      last_gpu_base_input_level_ = info.base_input_level;
      last_gpu_output_level_ = info.output_level;
    }
    accepted_gpu_jobs_.fetch_add(1);
    return CompactionServiceScheduleResponse(job_id,
                                             CompactionServiceJobStatus::kSuccess);
  }

  CompactionServiceJobStatus Wait(const std::string& scheduled_job_id,
                                  std::string* result) override {
    if (result == nullptr) {
      return CompactionServiceJobStatus::kFailure;
    }

    JobState job;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = jobs_.find(scheduled_job_id);
      if (it == jobs_.end()) {
        return CompactionServiceJobStatus::kFailure;
      }
      job = std::move(it->second);
      jobs_.erase(it);
    }

    if (canceled_.load()) {
      return CompactionServiceJobStatus::kAborted;
    }

    const std::string output_dir = JoinPath(scratch_root_, scheduled_job_id);
    Status s = options_.env->CreateDirIfMissing(output_dir);
    if (!s.ok()) {
      return CompactionServiceJobStatus::kFailure;
    }
    const uint64_t work_start_micros = options_.env->NowMicros();

    std::vector<std::vector<BridgeRecord>> runs;
    KeyEncoding key_encoding;
    uint64_t bytes_read = 0;
    uint64_t num_input_records = 0;
    uint64_t num_input_records_for_stats = 0;
    uint64_t total_input_raw_key_bytes = 0;
    uint64_t total_input_raw_value_bytes = 0;

    s = LoadInputRuns(job, &runs, &key_encoding, &bytes_read,
                      &num_input_records,
                      &num_input_records_for_stats,
                      &total_input_raw_key_bytes,
                      &total_input_raw_value_bytes);
    if (num_input_records_for_stats == 0) {
      num_input_records_for_stats = num_input_records;
    }
    if (s.IsNotSupported() && service_options_.fallback_on_unsupported_input) {
      CleanupPath(output_dir);
      fallback_local_jobs_.fetch_add(1);
      return CompactionServiceJobStatus::kUseLocal;
    }
    if (!s.ok()) {
      SerializeFailure(output_dir, job, s, result);
      CleanupPath(output_dir);
      return CompactionServiceJobStatus::kFailure;
    }

    std::vector<BridgeInputRun> bridge_runs;
    bridge_runs.reserve(runs.size());
    for (const auto& run : runs) {
      BridgeInputRun bridge_run;
      bridge_run.records = run.data();
      bridge_run.num_records = run.size();
      bridge_runs.push_back(bridge_run);
    }

    BridgeMergeResult merged_result;
    s = MapBridgeStatus(gpcomp_rocksdb_bridge::RunMerge(
        gpu_handle_, bridge_runs.data(), bridge_runs.size(), &merged_result));
    if (!s.ok()) {
      gpcomp_rocksdb_bridge::FreeResult(&merged_result);
      SerializeFailure(output_dir, job, s, result);
      CleanupPath(output_dir);
      return CompactionServiceJobStatus::kFailure;
    }

    CompactionServiceResult compaction_result;
    compaction_result.status = Status::OK();
    compaction_result.output_level = job.input.output_level;
    compaction_result.output_path = output_dir;
    compaction_result.bytes_read = bytes_read;

    s = BuildOutput(output_dir, job, num_input_records_for_stats,
                    total_input_raw_key_bytes, total_input_raw_value_bytes,
                    key_encoding, merged_result, &compaction_result);
    gpcomp_rocksdb_bridge::FreeResult(&merged_result);
    if (!s.ok()) {
      SerializeFailure(output_dir, job, s, result);
      CleanupPath(output_dir);
      return CompactionServiceJobStatus::kFailure;
    }

    compaction_result.stats.elapsed_micros =
        options_.env->NowMicros() - work_start_micros;
    compaction_result.stats.cpu_micros = compaction_result.stats.elapsed_micros;
    compaction_result.internal_stats.output_level_stats.micros =
        compaction_result.stats.elapsed_micros;
    compaction_result.internal_stats.output_level_stats.cpu_micros =
        compaction_result.stats.cpu_micros;

    s = compaction_result.Write(result);
    if (!s.ok()) {
      CleanupPath(output_dir);
      return CompactionServiceJobStatus::kFailure;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      completed_job_paths_[scheduled_job_id] = output_dir;
    }
    completed_gpu_jobs_.fetch_add(1);
    return CompactionServiceJobStatus::kSuccess;
  }

  void CancelAwaitingJobs() override { canceled_.store(true); }

  void OnInstallation(const std::string& scheduled_job_id,
                      CompactionServiceJobStatus /*status*/) override {
    std::string output_dir;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = completed_job_paths_.find(scheduled_job_id);
      if (it == completed_job_paths_.end()) {
        return;
      }
      output_dir = std::move(it->second);
      completed_job_paths_.erase(it);
    }
    CleanupPath(output_dir);
  }

  Status GetCounters(GpuCompactionServiceCounters* counters) const override {
    if (counters == nullptr) {
      return Status::InvalidArgument("counters output pointer is null");
    }
    counters->accepted_gpu_jobs = accepted_gpu_jobs_.load();
    counters->fallback_local_jobs = fallback_local_jobs_.load();
    counters->completed_gpu_jobs = completed_gpu_jobs_.load();
    std::lock_guard<std::mutex> lock(mu_);
    counters->last_gpu_base_input_level = last_gpu_base_input_level_;
    counters->last_gpu_output_level = last_gpu_output_level_;
    return Status::OK();
  }

 private:
  struct JobState {
    CompactionServiceJobInfo info;
    CompactionServiceInput input;
  };

  struct ActiveOutput {
    uint64_t file_number = 0;
    std::string file_name;
    std::string file_path;
    std::unique_ptr<WritableFileWriter> file_writer;
    std::unique_ptr<TableBuilder> builder;
    std::string smallest_internal_key;
    std::string largest_internal_key;
    SequenceNumber smallest_seqno = kMaxSequenceNumber;
    SequenceNumber largest_seqno = 0;
  };

  bool IsEligible(const CompactionServiceJobInfo& info,
                  const CompactionServiceInput& input) const {
    if (info.cf_id != 0 || info.cf_name != kDefaultColumnFamilyName) {
      return false;
    }
    if (info.base_input_level != 0 || info.output_level != 1 ||
        input.output_level != 1) {
      return false;
    }
    if (input.has_begin || input.has_end || input.input_files.empty()) {
      return false;
    }
    return true;
  }

  Status EnsureScratchRoot(const std::string& db_name) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!scratch_root_.empty()) {
      return Status::OK();
    }
    scratch_root_ = service_options_.scratch_root.empty()
                        ? JoinPath(db_name, ".gpu_compaction_service_" + service_id_)
                        : service_options_.scratch_root;
    return options_.env->CreateDirIfMissing(scratch_root_);
  }

  void CleanupPath(const std::string& path) const {
    if (path.empty()) {
      return;
    }
    Status exists = options_.env->FileExists(path);
    if (exists.ok()) {
      DestroyDir(options_.env, path).PermitUncheckedError();
    }
  }

  std::string ResolveInputPath(const JobState& job,
                               const std::string& input_file) const {
    if (!input_file.empty() && input_file.front() == '/') {
      return input_file;
    }
    return JoinPath(job.info.db_name, input_file);
  }

  Status LoadInputRuns(const JobState& job,
                       std::vector<std::vector<BridgeRecord>>* runs,
                       KeyEncoding* key_encoding,
                       uint64_t* bytes_read, uint64_t* num_input_records,
                       uint64_t* num_input_records_for_stats,
                       uint64_t* total_input_raw_key_bytes,
                       uint64_t* total_input_raw_value_bytes) const {
    if (key_encoding == nullptr) {
      return Status::InvalidArgument("key encoding output pointer is null");
    }
    *key_encoding = KeyEncoding();
    runs->clear();
    runs->reserve(job.input.input_files.size());

    for (const auto& input_file : job.input.input_files) {
      const std::string input_path = ResolveInputPath(job, input_file);

      uint64_t file_size = 0;
      Status s = options_.env->GetFileSize(input_path, &file_size);
      if (!s.ok()) {
        return s;
      }
      *bytes_read += file_size;

      const Options reader_options = MakeSstReaderOptions(options_);
      SstFileReader reader(reader_options);
      s = reader.Open(input_path);
      if (!s.ok()) {
        return s;
      }

      const auto props = reader.GetTableProperties();
      if (props != nullptr && props->num_range_deletions > 0) {
        return Status::NotSupported(
            "GpuCompactionService does not support range tombstones");
      }
      if (props != nullptr) {
        *num_input_records_for_stats +=
            props->num_entries - props->num_range_deletions;
      }

      std::unique_ptr<Iterator> table_iter = reader.NewTableIterator();
      if (table_iter == nullptr) {
        return Status::Corruption("failed to create SST table iterator");
      }

      std::vector<BridgeRecord> run;
      for (table_iter->SeekToFirst(); table_iter->Valid(); table_iter->Next()) {
        const Slice internal_key = table_iter->key();
        const Slice value = table_iter->value();

        Status key_status = ValidateKeyEncoding(internal_key, key_encoding);
        if (!key_status.ok()) {
          return key_status;
        }
        if (value.size() != kGpuValueBytes) {
          return Status::NotSupported(
              "GpuCompactionService requires 32-byte values");
        }
        if (ExtractValueType(internal_key) != kTypeValue) {
          return Status::NotSupported(
              "GpuCompactionService supports only point Put entries");
        }

        BridgeRecord record{};
        s = EncodeCanonicalKey(internal_key, *key_encoding, record.key);
        if (!s.ok()) {
          return s;
        }
        std::memcpy(record.value, value.data(), kGpuValueBytes);
        run.push_back(record);
        ++(*num_input_records);
        *total_input_raw_key_bytes += internal_key.size();
        *total_input_raw_value_bytes += value.size();
      }

      s = table_iter->status();
      if (!s.ok()) {
        return s;
      }
      runs->push_back(std::move(run));
    }

    return Status::OK();
  }

  Status OpenOutputFile(const std::string& output_dir, uint64_t file_number,
                        ActiveOutput* output) const {
    output->file_number = file_number;
    output->file_name = MakeTableFileName(file_number);
    output->file_path = JoinPath(output_dir, output->file_name);
    output->smallest_internal_key.clear();
    output->largest_internal_key.clear();
    output->smallest_seqno = kMaxSequenceNumber;
    output->largest_seqno = 0;

    const std::string file_path = output->file_path;
    std::unique_ptr<FSWritableFile> writable_file;
    FileOptions file_options(env_options_);
    IOStatus io_s = NewWritableFile(options_.env->GetFileSystem().get(), file_path,
                                    &writable_file, file_options);
    if (!io_s.ok()) {
      return io_s;
    }

    FileTypeSet checksum_handoff = immutable_options_.checksum_handoff_file_types;
    output->file_writer.reset(new WritableFileWriter(
        std::move(writable_file), file_path, file_options, immutable_options_.clock,
        nullptr /* io_tracer */, immutable_options_.stats,
        Histograms::SST_WRITE_MICROS, immutable_options_.listeners,
        immutable_options_.file_checksum_gen_factory.get(),
        checksum_handoff.Contains(FileType::kTableFile), false));

    InternalTblPropCollFactories internal_tbl_prop_coll_factories;
    const ReadOptions read_options(Env::IOActivity::kCompaction);
    const WriteOptions write_options(Env::IOActivity::kCompaction);
    const uint64_t target_file_size =
        std::max<uint64_t>(mutable_cf_options_.target_file_size_base, 1);
    TableBuilderOptions table_builder_options(
        immutable_options_, mutable_cf_options_, read_options, write_options,
        immutable_options_.internal_comparator, &internal_tbl_prop_coll_factories,
        mutable_cf_options_.compression, mutable_cf_options_.compression_opts,
        0 /* column_family_id */, kDefaultColumnFamilyName, 1 /* level */,
        kUnknownNewestKeyTime, false /* is_bottommost */,
        TableFileCreationReason::kCompaction, 0 /* oldest_key_time */,
        0 /* file_creation_time */, "" /* db_id */, "" /* db_session_id */,
        target_file_size, file_number);
    output->builder.reset(
        NewTableBuilder(table_builder_options, output->file_writer.get()));
    return Status::OK();
  }

  Status FinishOutputFile(ActiveOutput* output,
                          CompactionServiceResult* result) const {
    if (output->builder == nullptr) {
      return Status::OK();
    }

    Status s = output->builder->Finish();
    if (s.ok()) {
      IOStatus io_s = output->builder->io_status();
      if (!io_s.ok()) {
        s = io_s;
      }
    }

    IOOptions io_options;
    IOStatus io_s = WritableFileWriter::PrepareIOOptions(
        WriteOptions(Env::IOActivity::kCompaction), io_options);
    if (s.ok() && io_s.ok()) {
      io_s = output->file_writer->Sync(io_options, options_.use_fsync);
    }
    if (s.ok() && io_s.ok()) {
      io_s = output->file_writer->Close(io_options);
    }
    if (s.ok() && io_s.ok()) {
      s = io_s;
    }

    if (!s.ok()) {
      output->builder->Abandon();
      output->builder.reset();
      output->file_writer.reset();
      options_.env->DeleteFile(output->file_path).PermitUncheckedError();
      return s;
    }

    const uint64_t file_size = output->builder->FileSize();
    const TableProperties table_properties = output->builder->GetTableProperties();
    const bool marked_for_compaction = output->builder->NeedCompact();
    const std::string file_checksum = output->file_writer->GetFileChecksum();
    const std::string file_checksum_func_name =
        output->file_writer->GetFileChecksumFuncName();

    result->bytes_written += file_size;
    result->output_files.emplace_back(
        output->file_name, file_size, output->smallest_seqno,
        output->largest_seqno, output->smallest_internal_key,
        output->largest_internal_key, kUnknownOldestAncesterTime,
        kUnknownFileCreationTime, kGpuOutputEpochNumber, file_checksum,
        file_checksum_func_name, 0 /* paranoid_hash */,
        marked_for_compaction, kNullUniqueId64x2, table_properties,
        false /* is_proximal_level_output */, Temperature::kUnknown);

    output->builder.reset();
    output->file_writer.reset();
    return Status::OK();
  }

  Status BuildOutput(const std::string& output_dir, const JobState& job,
                     uint64_t num_input_records,
                     uint64_t total_input_raw_key_bytes,
                     uint64_t total_input_raw_value_bytes,
                     const KeyEncoding& key_encoding,
                     const BridgeMergeResult& merged,
                     CompactionServiceResult* result) const {
    const uint64_t target_file_size =
        std::max<uint64_t>(mutable_cf_options_.target_file_size_base, 1);

    ActiveOutput output;
    uint64_t next_output_file_number = 1;
    BridgeRecord previous_record{};

    for (std::size_t i = 0; i < merged.num_records; ++i) {
      const BridgeRecord& record = merged.records[i];

      if (output.builder != nullptr && output.builder->NumEntries() > 0 &&
          output.builder->EstimatedFileSize() >= target_file_size &&
          !SameUserKey(previous_record, record)) {
        Status s = FinishOutputFile(&output, result);
        if (!s.ok()) {
          return s;
        }
      }

      if (output.builder == nullptr) {
        Status s =
            OpenOutputFile(output_dir, next_output_file_number++, &output);
        if (!s.ok()) {
          return s;
        }
      }

      std::string internal_key = DecodeCanonicalKey(record.key, key_encoding);
      const Slice key(internal_key);
      const Slice value(reinterpret_cast<const char*>(record.value), kGpuValueBytes);
      const SequenceNumber seqno = ExtractInternalKeyFooter(key) >> 8;

      output.builder->Add(key, value);
      if (!output.builder->status().ok()) {
        return output.builder->status();
      }
      if (!output.builder->io_status().ok()) {
        return output.builder->io_status();
      }

      if (output.smallest_internal_key.empty()) {
        output.smallest_internal_key = internal_key;
      }
      output.largest_internal_key = internal_key;
      output.smallest_seqno = std::min(output.smallest_seqno, seqno);
      output.largest_seqno = std::max(output.largest_seqno, seqno);
      previous_record = record;
    }

    Status s = FinishOutputFile(&output, result);
    if (!s.ok()) {
      return s;
    }

    result->stats.has_accurate_num_input_records = true;
    result->stats.num_input_records = num_input_records;
    result->stats.num_output_records = merged.num_records;
    result->stats.num_input_files = job.input.input_files.size();
    result->stats.num_output_files = result->output_files.size();
    result->stats.is_full_compaction = job.info.is_full_compaction;
    result->stats.is_manual_compaction = job.info.is_manual_compaction;
    result->stats.is_remote_compaction = true;
    result->stats.total_input_bytes = result->bytes_read;
    result->stats.total_output_bytes = result->bytes_written;
    result->stats.total_input_raw_key_bytes = total_input_raw_key_bytes;
    result->stats.total_input_raw_value_bytes = total_input_raw_value_bytes;
    if (!result->output_files.empty()) {
      result->stats.smallest_output_key_prefix =
          ExtractKeyPrefix(result->output_files.front().smallest_internal_key);
      result->stats.largest_output_key_prefix =
          ExtractKeyPrefix(result->output_files.back().largest_internal_key);
    }

    result->internal_stats.output_level_stats =
        InternalStats::CompactionStats(job.info.compaction_reason, 1);
    result->internal_stats.output_level_stats.bytes_read_non_output_levels =
        result->bytes_read;
    result->internal_stats.output_level_stats.bytes_written =
        result->bytes_written;
    result->internal_stats.output_level_stats.num_input_files_in_non_output_levels =
        static_cast<int>(job.input.input_files.size());
    result->internal_stats.output_level_stats.num_output_files =
        static_cast<int>(result->output_files.size());
    result->internal_stats.output_level_stats.num_input_records =
        num_input_records;
    result->internal_stats.output_level_stats.num_output_records =
        merged.num_records;

    return Status::OK();
  }

  void SerializeFailure(const std::string& output_dir, const JobState& job,
                        const Status& status, std::string* result) const {
    if (result == nullptr) {
      return;
    }
    CompactionServiceResult failed_result;
    failed_result.status = status;
    failed_result.output_level = job.input.output_level;
    failed_result.output_path = output_dir;
    failed_result.Write(result).PermitUncheckedError();
  }

  Options options_;
  GpuCompactionServiceOptions service_options_;
  EnvOptions env_options_;
  ImmutableOptions immutable_options_;
  MutableCFOptions mutable_cf_options_;
  BridgeHandle* gpu_handle_ = nullptr;

  mutable std::mutex mu_;
  std::map<std::string, JobState> jobs_;
  std::map<std::string, std::string> completed_job_paths_;
  std::string scratch_root_;
  std::string service_id_;
  int last_gpu_base_input_level_ = -1;
  int last_gpu_output_level_ = -1;

  std::atomic<uint64_t> next_job_id_{1};
  std::atomic<uint64_t> accepted_gpu_jobs_{0};
  std::atomic<uint64_t> fallback_local_jobs_{0};
  std::atomic<uint64_t> completed_gpu_jobs_{0};
  std::atomic<bool> canceled_{false};
};

Status NewGpuCompactionService(
    const Options& db_options,
    const GpuCompactionServiceOptions& service_options,
    std::shared_ptr<CompactionService>* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("output compaction service pointer is null");
  }
  *out = nullptr;

  BridgeHandle* gpu_handle = nullptr;
  const BridgeOptions bridge_options{service_options.cuda_device};
  Status s = MapBridgeStatus(
      gpcomp_rocksdb_bridge::Open(bridge_options, &gpu_handle));
  if (!s.ok()) {
    return s;
  }

  out->reset(new GpuCompactionServiceImpl(db_options, service_options,
                                          gpu_handle));
  return Status::OK();
}

}  // namespace ROCKSDB_NAMESPACE
