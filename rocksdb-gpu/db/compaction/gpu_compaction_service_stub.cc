// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#include "rocksdb/utilities/gpu_compaction_service.h"

namespace ROCKSDB_NAMESPACE {

Status NewGpuCompactionService(
    const Options& /*db_options*/,
    const GpuCompactionServiceOptions& /*service_options*/,
    std::shared_ptr<CompactionService>* out) {
  if (out != nullptr) {
    *out = nullptr;
  }
  return Status::NotSupported("RocksDB was built without USE_GPCOMP=1");
}

}  // namespace ROCKSDB_NAMESPACE
