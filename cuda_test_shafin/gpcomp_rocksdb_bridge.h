#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gpcomp_rocksdb_bridge {

constexpr std::size_t kInternalKeyBytes = 16;
constexpr std::size_t kValueBytes = 32;

struct Options {
  int cuda_device = 0;
};

struct Record {
  uint8_t key[kInternalKeyBytes];
  uint8_t value[kValueBytes];
};

struct InputRun {
  const Record* records = nullptr;
  std::size_t num_records = 0;
};

struct MergeResult {
  Record* records = nullptr;
  std::size_t num_records = 0;
  float kernel_ms = 0.0f;
  float wall_ms = 0.0f;
};

struct Handle;

std::string Open(const Options& options, Handle** out);
std::string RunMerge(Handle* handle, const InputRun* runs, std::size_t num_runs,
                     MergeResult* out);
void FreeResult(MergeResult* result);
void Close(Handle* handle);

}  // namespace gpcomp_rocksdb_bridge
