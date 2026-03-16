// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <memory>
#include <string>

#include "rocksdb/options.h"

namespace ROCKSDB_NAMESPACE {

struct GpuCompactionServiceOptions {
  int cuda_device = 0;
  std::string scratch_root;
  bool fallback_on_unsupported_input = true;
};

struct GpuCompactionServiceCounters {
  uint64_t accepted_gpu_jobs = 0;
  uint64_t fallback_local_jobs = 0;
  uint64_t completed_gpu_jobs = 0;
  int last_gpu_base_input_level = -1;
  int last_gpu_output_level = -1;
};

class GpuCompactionService : public CompactionService {
 public:
  static const char* Type() { return "GpuCompactionService"; }

  virtual Status GetCounters(GpuCompactionServiceCounters* counters) const = 0;

  ~GpuCompactionService() override = default;
};

Status NewGpuCompactionService(
    const Options& db_options,
    const GpuCompactionServiceOptions& service_options,
    std::shared_ptr<CompactionService>* out);

}  // namespace ROCKSDB_NAMESPACE
