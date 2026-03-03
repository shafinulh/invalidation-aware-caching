//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/compaction/compaction_job.h"
#include "db/db_test_util.h"
#include "port/stack_trace.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace ROCKSDB_NAMESPACE {

namespace {

double NowMicros() {
  using Clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::micro>(
             Clock::now().time_since_epoch())
      .count();
}

int GetEnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }

  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == nullptr || *end != '\0') {
    return fallback;
  }
  return static_cast<int>(parsed);
}

std::string GetEnvString(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  return value;
}

std::string ShellEscape(const std::string& input) {
  std::string escaped = "'";
  for (char c : input) {
    if (c == '\'') {
      escaped += "'\"'\"'";
    } else {
      escaped.push_back(c);
    }
  }
  escaped.push_back('\'');
  return escaped;
}

uint64_t NextScratchRootId() {
  static std::atomic<uint64_t> next_id{0};
  return next_id.fetch_add(1);
}

struct GpuReplayMetrics {
  uint64_t files = 0;
  uint64_t bytes = 0;
  double read_us = 0.0;
  double host_to_device_us = 0.0;
  double kernel_us = 0.0;
  double device_to_host_us = 0.0;
  double write_us = 0.0;
  double sync_us = 0.0;
  double total_us = 0.0;
};

struct HookBenchmarkRunMetrics {
  GpuReplayMetrics input_stage;
  GpuReplayMetrics replay;
  double gpu_pipeline_total_us = 0.0;
  double stage_ground_truth_us = 0.0;
  double wait_total_us = 0.0;
};

bool ParseSingleGpuReplayMetric(const std::string& key, const std::string& value,
                                GpuReplayMetrics* metrics) {
  if (key == "files") {
    metrics->files = static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
  } else if (key == "bytes") {
    metrics->bytes = static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
  } else if (key == "total_us") {
    metrics->total_us = std::strtod(value.c_str(), nullptr);
  } else if (key == "read_us") {
    metrics->read_us = std::strtod(value.c_str(), nullptr);
  } else if (key == "h2d_us") {
    metrics->host_to_device_us = std::strtod(value.c_str(), nullptr);
  } else if (key == "kernel_us") {
    metrics->kernel_us = std::strtod(value.c_str(), nullptr);
  } else if (key == "d2h_us") {
    metrics->device_to_host_us = std::strtod(value.c_str(), nullptr);
  } else if (key == "write_us") {
    metrics->write_us = std::strtod(value.c_str(), nullptr);
  } else if (key == "sync_us") {
    metrics->sync_us = std::strtod(value.c_str(), nullptr);
  } else {
    return false;
  }
  return true;
}

bool ParseGpuReplayMetrics(const std::string& output,
                           HookBenchmarkRunMetrics* metrics) {
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    static const std::string kPrefix = "GPU_FILE_REPLAY_METRICS ";
    if (line.rfind(kPrefix, 0) != 0) {
      continue;
    }

    std::istringstream token_stream(line.substr(kPrefix.size()));
    std::string token;
    while (token_stream >> token) {
      const auto eq = token.find('=');
      if (eq == std::string::npos) {
        continue;
      }

      const std::string key = token.substr(0, eq);
      const std::string value = token.substr(eq + 1);
      if (key == "total_us") {
        metrics->gpu_pipeline_total_us = std::strtod(value.c_str(), nullptr);
      } else if (key.rfind("stage_", 0) == 0) {
        ParseSingleGpuReplayMetric(key.substr(6), value, &metrics->input_stage);
      } else if (key.rfind("copy_", 0) == 0) {
        ParseSingleGpuReplayMetric(key.substr(5), value, &metrics->replay);
      }
    }
    return true;
  }
  return false;
}

double OutputPersistUs(const HookBenchmarkRunMetrics& metrics) {
  return metrics.replay.device_to_host_us + metrics.replay.write_us +
         metrics.replay.sync_us;
}

double GpuHookE2eUs(double compact_range_us,
                    const HookBenchmarkRunMetrics& metrics) {
  return std::max(0.0, compact_range_us - metrics.stage_ground_truth_us);
}

double PostWriteCompletionUs(double compact_range_us,
                             const HookBenchmarkRunMetrics& metrics) {
  const double accounted_us = metrics.stage_ground_truth_us +
                              metrics.gpu_pipeline_total_us;
  return std::max(0.0, compact_range_us - accounted_us);
}

void AppendBenchmarkCsv(const std::string& csv_path, int rep,
                        double compact_range_us, double verify_us,
                        const HookBenchmarkRunMetrics& metrics) {
  bool write_header = true;
  {
    std::ifstream in(csv_path);
    if (in.good()) {
      write_header = (in.peek() == std::ifstream::traits_type::eof());
    }
  }

  std::ofstream out(csv_path, std::ios::app);
  if (!out) {
    return;
  }

  if (write_header) {
    out << "rep,compact_range_us,gpu_hook_e2e_us,verify_us,wait_total_us,"
        << "stage_ground_truth_us,gpu_pipeline_total_us,"
        << "input_file_reads_us,dummy_compaction_us,output_persist_to_disk_us,"
        << "post_write_completion_us,"
        << "input_stage_us,input_read_us,input_h2d_us,input_kernel_us,"
        << "input_files,input_bytes,output_replay_us,output_read_us,"
        << "output_h2d_us,output_kernel_us,output_d2h_us,output_write_us,"
        << "output_sync_us,"
        << "output_files,output_bytes\n";
  }

  out << rep << ','
      << compact_range_us << ','
      << GpuHookE2eUs(compact_range_us, metrics) << ','
      << verify_us << ','
      << metrics.wait_total_us << ','
      << metrics.stage_ground_truth_us << ','
      << metrics.gpu_pipeline_total_us << ','
      << metrics.input_stage.total_us << ','
      << metrics.replay.kernel_us << ','
      << OutputPersistUs(metrics) << ','
      << PostWriteCompletionUs(compact_range_us, metrics) << ','
      << metrics.input_stage.total_us << ','
      << metrics.input_stage.read_us << ','
      << metrics.input_stage.host_to_device_us << ','
      << metrics.input_stage.kernel_us << ','
      << metrics.input_stage.files << ','
      << metrics.input_stage.bytes << ','
      << metrics.replay.total_us << ','
      << metrics.replay.read_us << ','
      << metrics.replay.host_to_device_us << ','
      << metrics.replay.kernel_us << ','
      << metrics.replay.device_to_host_us << ','
      << metrics.replay.write_us << ','
      << metrics.replay.sync_us << ','
      << metrics.replay.files << ','
      << metrics.replay.bytes << '\n';
}

}  // namespace

class GroundTruthReplayCompactionService : public CompactionService {
 public:
  GroundTruthReplayCompactionService(std::string db_path, const Options& options)
      : db_path_(std::move(db_path)),
        options_(options),
        scratch_id_(std::to_string(NextScratchRootId())),
        ground_truth_root_(db_path_ + "_gpu_compaction_ground_truth_" +
                           scratch_id_),
        replay_root_(db_path_ + "_gpu_compaction_replay_" + scratch_id_) {
    CleanupPath(ground_truth_root_);
    CleanupPath(replay_root_);
    options_.env->CreateDirIfMissing(ground_truth_root_).PermitUncheckedError();
    options_.env->CreateDirIfMissing(replay_root_).PermitUncheckedError();
  }

  ~GroundTruthReplayCompactionService() override = default;

  static const char* kClassName() {
    return "GroundTruthReplayCompactionService";
  }

  const char* Name() const override { return kClassName(); }

  CompactionServiceScheduleResponse Schedule(
      const CompactionServiceJobInfo& info,
      const std::string& compaction_service_input) override {
    JobState job_state;
    job_state.info = info;
    job_state.compaction_input_binary = compaction_service_input;

    Status s = CompactionServiceInput::Read(compaction_service_input,
                                            &job_state.compaction_input);
    if (!s.ok()) {
      return CompactionServiceScheduleResponse(
          CompactionServiceJobStatus::kFailure);
    }

    const std::string job_id =
        "gpu-hook-job-" + std::to_string(next_job_id_.fetch_add(1));
    {
      std::lock_guard<std::mutex> lock(mu_);
      jobs_.emplace(job_id, std::move(job_state));
    }
    return CompactionServiceScheduleResponse(
        job_id, CompactionServiceJobStatus::kSuccess);
  }

  CompactionServiceJobStatus Wait(const std::string& scheduled_job_id,
                                  std::string* result) override {
    JobState job_state;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto iter = jobs_.find(scheduled_job_id);
      if (iter == jobs_.end()) {
        return CompactionServiceJobStatus::kFailure;
      }
      job_state = std::move(iter->second);
      jobs_.erase(iter);
      last_metrics_ = HookBenchmarkRunMetrics();
    }

    const double wait_start = NowMicros();

    Status s = StageGroundTruthOutput(scheduled_job_id, &job_state);
    if (!s.ok()) {
      return CompactionServiceJobStatus::kFailure;
    }

    s = RunDummyGpuCompactionKernel(scheduled_job_id, job_state, result);
    if (!s.ok()) {
      return CompactionServiceJobStatus::kFailure;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      last_metrics_.wait_total_us = NowMicros() - wait_start;
    }

    replay_runs_.fetch_add(1);
    return CompactionServiceJobStatus::kSuccess;
  }

  void OnInstallation(const std::string& /*scheduled_job_id*/,
                      CompactionServiceJobStatus status) override {
    final_install_status_.store(status);
  }

  uint64_t GetGroundTruthRuns() const { return ground_truth_runs_.load(); }
  uint64_t GetReplayRuns() const { return replay_runs_.load(); }
  uint64_t GetLastInputBytes() const { return last_input_bytes_.load(); }
  size_t GetLastInputFileCount() const { return last_input_file_count_.load(); }
  size_t GetLastOutputFileCount() const {
    return last_output_file_count_.load();
  }
  CompactionServiceJobStatus GetFinalInstallStatus() const {
    return final_install_status_.load();
  }

  std::string GetLastGroundTruthDir() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_ground_truth_dir_;
  }

  HookBenchmarkRunMetrics GetLastRunMetrics() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_metrics_;
  }

 private:
  struct JobState {
    CompactionServiceJobInfo info;
    std::string compaction_input_binary;
    CompactionServiceInput compaction_input;
    CompactionServiceResult ground_truth_result;
  };

  void CleanupPath(const std::string& path) {
    Status s = options_.env->FileExists(path);
    if (s.ok()) {
      DestroyDir(options_.env, path).PermitUncheckedError();
    }
  }

  CompactionServiceOptionsOverride BuildOptionsOverride() const {
    CompactionServiceOptionsOverride options_override;
    options_override.env = options_.env;
    options_override.file_checksum_gen_factory =
        options_.file_checksum_gen_factory;
    options_override.comparator = options_.comparator;
    options_override.merge_operator = options_.merge_operator;
    options_override.compaction_filter = options_.compaction_filter;
    options_override.compaction_filter_factory =
        options_.compaction_filter_factory;
    options_override.prefix_extractor = options_.prefix_extractor;
    options_override.table_factory = options_.table_factory;
    options_override.sst_partitioner_factory = options_.sst_partitioner_factory;
    options_override.statistics = options_.statistics;
    options_override.info_log = options_.info_log;
    return options_override;
  }

  Status StageGroundTruthOutput(const std::string& scheduled_job_id,
                                JobState* job_state) {
    const std::string output_dir = ground_truth_root_ + "/" + scheduled_job_id;
    Status s = options_.env->CreateDirIfMissing(output_dir);
    if (!s.ok()) {
      return s;
    }

    const double t0 = NowMicros();
    std::string ground_truth_binary;
    s = DB::OpenAndCompact(db_path_, output_dir, job_state->compaction_input_binary,
                           &ground_truth_binary, BuildOptionsOverride());
    if (!s.ok()) {
      return s;
    }

    s = CompactionServiceResult::Read(ground_truth_binary,
                                      &job_state->ground_truth_result);
    if (!s.ok()) {
      return s;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      last_ground_truth_dir_ = job_state->ground_truth_result.output_path;
      last_metrics_.stage_ground_truth_us = NowMicros() - t0;
    }
    last_output_file_count_.store(job_state->ground_truth_result.output_files.size());
    ground_truth_runs_.fetch_add(1);
    return Status::OK();
  }

  Status RunGpuReplayHelper(const std::string& manifest_path,
                            HookBenchmarkRunMetrics* metrics) const {
    const std::string helper_path = GetEnvString(
        "GPU_FILE_REPLAY_BENCH", "../benchmarks/gpu/gpu_file_replay_bench");
    if (access(helper_path.c_str(), X_OK) != 0) {
      return Status::NotFound("Missing GPU replay helper at " + helper_path);
    }

    const int gpu_device = GetEnvInt("GPU_DEVICE", 0);
    const int alignment = GetEnvInt("ALIGNMENT", 4096);

    std::ostringstream cmd;
    cmd << ShellEscape(helper_path)
        << " --manifest " << ShellEscape(manifest_path)
        << " --gpu_device " << gpu_device
        << " --alignment " << alignment;

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (pipe == nullptr) {
      return Status::IOError("popen() failed for GPU replay helper");
    }

    std::string output;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      output += buffer;
    }

    const int status = pclose(pipe);
    if (status == -1) {
      return Status::IOError("pclose() failed for GPU replay helper");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      return Status::IOError("GPU replay helper failed: " + output);
    }
    if (!ParseGpuReplayMetrics(output, metrics)) {
      return Status::Corruption("Unable to parse GPU replay helper metrics");
    }
    return Status::OK();
  }

  Status RunDummyGpuCompactionKernel(const std::string& scheduled_job_id,
                                     const JobState& job_state,
                                     std::string* result) {
    CompactionServiceResult replay_result = job_state.ground_truth_result;
    replay_result.output_path = replay_root_ + "/" + scheduled_job_id;

    Status s = options_.env->CreateDirIfMissing(replay_result.output_path);
    if (!s.ok()) {
      return s;
    }

    const std::string manifest_path =
        replay_root_ + "/" + scheduled_job_id + ".manifest.tsv";
    {
      std::ofstream manifest(manifest_path);
      if (!manifest) {
        return Status::IOError("Unable to create manifest " + manifest_path);
      }
      uint64_t input_bytes = 0;
      last_input_file_count_.store(job_state.compaction_input.input_files.size());
      for (const auto& input_file : job_state.compaction_input.input_files) {
        const std::string input_path = db_path_ + "/" + input_file;
        uint64_t file_size = 0;
        s = options_.env->GetFileSize(input_path, &file_size);
        if (!s.ok()) {
          return s;
        }
        manifest << input_path << '\t' << '\t' << file_size << '\n';
        input_bytes += file_size;
      }
      last_input_bytes_.store(input_bytes);
      for (const auto& output_file : replay_result.output_files) {
        manifest << job_state.ground_truth_result.output_path << "/"
                 << output_file.file_name << '\t'
                 << replay_result.output_path << "/"
                 << output_file.file_name << '\t'
                 << output_file.file_size << '\n';
      }
    }

    HookBenchmarkRunMetrics run_metrics;
    s = RunGpuReplayHelper(manifest_path, &run_metrics);
    options_.env->DeleteFile(manifest_path).PermitUncheckedError();
    if (!s.ok()) {
      return s;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      last_metrics_.input_stage = run_metrics.input_stage;
      last_metrics_.replay = run_metrics.replay;
      last_metrics_.gpu_pipeline_total_us = run_metrics.gpu_pipeline_total_us;
    }

    return replay_result.Write(result);
  }

  mutable std::mutex mu_;
  const std::string db_path_;
  const Options options_;
  const std::string scratch_id_;
  const std::string ground_truth_root_;
  const std::string replay_root_;
  std::map<std::string, JobState> jobs_;
  std::string last_ground_truth_dir_;
  HookBenchmarkRunMetrics last_metrics_;

  std::atomic<uint64_t> next_job_id_{0};
  std::atomic<uint64_t> ground_truth_runs_{0};
  std::atomic<uint64_t> replay_runs_{0};
  std::atomic<uint64_t> last_input_bytes_{0};
  std::atomic<size_t> last_input_file_count_{0};
  std::atomic<size_t> last_output_file_count_{0};
  std::atomic<CompactionServiceJobStatus> final_install_status_{
      CompactionServiceJobStatus::kUseLocal};
};

class GpuCompactionHookBenchmarkTest : public DBTestBase {
 public:
  GpuCompactionHookBenchmarkTest()
      : DBTestBase("gpu_compaction_hook_benchmark_test", true) {}

 protected:
  static constexpr int kBaseInputFiles = 4;
  static constexpr int kValuesPerInputFile = 32;

  size_t ClampInputSstBytes() const {
    const int requested_mb = GetEnvInt("INPUT_SST_MB", 8);
    const int clamped_mb = std::max(8, std::min(64, requested_mb));
    return static_cast<size_t>(clamped_mb) * 1024 * 1024;
  }

  std::string BuildValuePayload(int key_id, bool gpu_version) const {
    const char fill = gpu_version ? static_cast<char>('a' + (key_id % 26))
                                  : static_cast<char>('A' + (key_id % 26));
    std::string value(value_bytes_, fill);
    const std::string prefix =
        "key=" + std::to_string(key_id) +
        ";version=" + (gpu_version ? std::string("gpu;") : std::string("base;"));
    std::copy(prefix.begin(), prefix.end(), value.begin());
    return value;
  }

  bool HasGpuRewrite(int key_id) const {
    return (key_id % kBaseInputFiles) == 0;
  }

  void ReopenWithDummyGpuCompaction(Options* options) {
    target_input_sst_bytes_ = ClampInputSstBytes();
    value_bytes_ = std::max<size_t>(target_input_sst_bytes_ / kValuesPerInputFile,
                                    4096);
    total_keys_ = kBaseInputFiles * kValuesPerInputFile;

    options->env = env_;
    options->create_if_missing = true;
    options->disable_auto_compactions = true;
    options->statistics = CreateDBStatistics();
    options->compression = kNoCompression;
    options->bottommost_compression = kDisableCompressionOption;
    options->write_buffer_size = target_input_sst_bytes_ + (1U << 20);
    options->target_file_size_base = target_input_sst_bytes_;

    compaction_service_ = std::make_shared<GroundTruthReplayCompactionService>(
        dbname_, *options);
    options->compaction_service = compaction_service_;
    DestroyAndReopen(*options);
  }

  void GenerateCompactionInputs() {
    for (int file_id = 0; file_id < kBaseInputFiles; ++file_id) {
      for (int i = 0; i < kValuesPerInputFile; ++i) {
        const int key_id = file_id * kValuesPerInputFile + i;
        ASSERT_OK(Put(Key(key_id), BuildValuePayload(key_id, false)));
      }
      ASSERT_OK(Flush());
      ASSERT_GE(NumTableFilesAtLevel(0), file_id + 1);
    }

    for (int key_id = 0; key_id < total_keys_; ++key_id) {
      if (!HasGpuRewrite(key_id)) {
        continue;
      }
      ASSERT_OK(Put(Key(key_id), BuildValuePayload(key_id, true)));
    }
    ASSERT_OK(Flush());
    ASSERT_GE(NumTableFilesAtLevel(0), kBaseInputFiles + 1);
  }

  void VerifyCompactionOutputs() {
    for (int key_id = 0; key_id < total_keys_; ++key_id) {
      ASSERT_EQ(Get(Key(key_id)),
                BuildValuePayload(key_id, HasGpuRewrite(key_id)));
    }
  }

  std::shared_ptr<GroundTruthReplayCompactionService> compaction_service_;
  size_t target_input_sst_bytes_ = 0;
  size_t value_bytes_ = 0;
  int total_keys_ = 0;
};

TEST_F(GpuCompactionHookBenchmarkTest, ManualCompactionReplaysGroundTruthSsts) {
  const int reps = std::max(1, GetEnvInt("NUM_REPS", 1));
  const std::string csv_path = GetEnvString("GPU_COMPACTION_HOOK_CSV", "");

  for (int rep = 0; rep < reps; ++rep) {
    SCOPED_TRACE("rep=" + std::to_string(rep));

    Options options = CurrentOptions();
    ReopenWithDummyGpuCompaction(&options);
    GenerateCompactionInputs();

    const double compact_start = NowMicros();
    ASSERT_OK(db_->CompactRange(CompactRangeOptions(), nullptr, nullptr));
    const double compact_done = NowMicros();

    const double verify_start = NowMicros();
    VerifyCompactionOutputs();
    const double verify_done = NowMicros();

    ASSERT_EQ(compaction_service_->GetGroundTruthRuns(), 1);
    ASSERT_EQ(compaction_service_->GetReplayRuns(), 1);
    ASSERT_GT(compaction_service_->GetLastInputFileCount(), 0U);
    ASSERT_GT(compaction_service_->GetLastOutputFileCount(), 0U);
    ASSERT_GT(compaction_service_->GetLastInputBytes(), 0U);
    ASSERT_EQ(compaction_service_->GetFinalInstallStatus(),
              CompactionServiceJobStatus::kSuccess);

    std::vector<std::string> children;
    ASSERT_OK(env_->GetChildren(compaction_service_->GetLastGroundTruthDir(),
                                &children));

    size_t sst_files = 0;
    for (const auto& child : children) {
      if (EndsWith(child, ".sst")) {
        ++sst_files;
      }
    }
    ASSERT_EQ(sst_files, compaction_service_->GetLastOutputFileCount());

    const HookBenchmarkRunMetrics metrics =
        compaction_service_->GetLastRunMetrics();
    ASSERT_GT(metrics.input_stage.bytes, 0U);
    ASSERT_GT(metrics.input_stage.total_us, 0.0);
    ASSERT_GT(metrics.replay.bytes, 0U);
    ASSERT_GT(metrics.replay.total_us, 0.0);
    ASSERT_GT(metrics.gpu_pipeline_total_us, 0.0);
    ASSERT_GT(metrics.replay.kernel_us, 0.0);
    ASSERT_GT(OutputPersistUs(metrics), 0.0);
    ASSERT_GT(metrics.wait_total_us, 0.0);
    ASSERT_GE(metrics.input_stage.bytes,
              static_cast<uint64_t>(target_input_sst_bytes_ * (kBaseInputFiles + 1)));
    ASSERT_GE(metrics.replay.bytes,
              static_cast<uint64_t>(target_input_sst_bytes_ * kBaseInputFiles));

    const double compact_range_us = compact_done - compact_start;
    const double gpu_hook_e2e_us = GpuHookE2eUs(compact_range_us, metrics);
    const double verify_us = verify_done - verify_start;
    const double post_write_completion_us =
        PostWriteCompletionUs(compact_range_us, metrics);

    if (!csv_path.empty()) {
      AppendBenchmarkCsv(csv_path, rep, compact_range_us, verify_us, metrics);
    }

    fprintf(stdout,
            "GPU_COMPACTION_HOOK_BENCH rep=%d compact_range_us=%.1f "
            "gpu_hook_e2e_us=%.1f stage_ground_truth_us=%.1f "
            "gpu_pipeline_total_us=%.1f input_file_reads_us=%.1f "
            "dummy_compaction_us=%.1f output_persist_to_disk_us=%.1f "
            "post_write_completion_us=%.1f verify_us=%.1f input_bytes=%" PRIu64
            " output_bytes=%" PRIu64 "\n",
            rep, compact_range_us, gpu_hook_e2e_us,
            metrics.stage_ground_truth_us, metrics.gpu_pipeline_total_us,
            metrics.input_stage.total_us, metrics.replay.kernel_us,
            OutputPersistUs(metrics), post_write_completion_us, verify_us,
            metrics.input_stage.bytes, metrics.replay.bytes);
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  RegisterCustomObjects(argc, argv);
  return RUN_ALL_TESTS();
}
