#include "gpcomp_rocksdb_bridge.h"

#include "gpcomp_common.cuh"
#include "gpcomp_merge.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace gpcomp_rocksdb_bridge {
namespace {

static_assert(sizeof(Record) == sizeof(KVPair),
              "Bridge record layout must match GPComp KVPair layout");

std::string MakeCudaStatus(const char* op, cudaError_t err) {
  if (err == cudaSuccess) {
    return "";
  }
  return std::string(op) + ": " + cudaGetErrorString(err);
}

}  // namespace

struct Handle {
  int cuda_device = 0;
};

std::string Open(const Options& options, Handle** out) {
  if (out == nullptr) {
    return "invalid output handle pointer";
  }
  *out = nullptr;

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return MakeCudaStatus("cudaGetDeviceCount", err);
  }
  if (device_count <= 0) {
    return "no CUDA devices available";
  }
  if (options.cuda_device < 0 || options.cuda_device >= device_count) {
    return "requested CUDA device is out of range";
  }

  err = cudaSetDevice(options.cuda_device);
  if (err != cudaSuccess) {
    return MakeCudaStatus("cudaSetDevice", err);
  }
  err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    return MakeCudaStatus("cudaFree", err);
  }

  Handle* handle = new (std::nothrow) Handle();
  if (handle == nullptr) {
    return "failed to allocate GPU bridge handle";
  }
  handle->cuda_device = options.cuda_device;
  *out = handle;
  return "";
}

std::string RunMerge(Handle* handle, const InputRun* runs, std::size_t num_runs,
                     MergeResult* out) {
  if (handle == nullptr) {
    return "GPU bridge handle is null";
  }
  if (out == nullptr) {
    return "GPU merge result pointer is null";
  }

  FreeResult(out);

  cudaError_t err = cudaSetDevice(handle->cuda_device);
  if (err != cudaSuccess) {
    return MakeCudaStatus("cudaSetDevice", err);
  }

  if (num_runs == 0) {
    return "";
  }
  if (num_runs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return "too many input runs for GPComp v1";
  }

  std::vector<std::vector<KVPair>> host_runs;
  host_runs.reserve(num_runs);

  std::vector<KVPair*> run_ptrs(num_runs, nullptr);
  std::vector<int> run_sizes(num_runs, 0);
  std::size_t total_records = 0;

  for (std::size_t i = 0; i < num_runs; ++i) {
    const InputRun& run = runs[i];
    if (run.num_records >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return "input run exceeds GPComp v1 record limit";
    }

    host_runs.emplace_back(run.num_records);
    if (run.num_records > 0) {
      if (run.records == nullptr) {
        return "input run records pointer is null";
      }
      std::memcpy(host_runs.back().data(), run.records,
                  run.num_records * sizeof(Record));
      run_ptrs[i] = host_runs.back().data();
    }
    run_sizes[i] = static_cast<int>(run.num_records);
    total_records += run.num_records;
  }

  if (total_records == 0) {
    return "";
  }
  if (total_records >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return "merged record count exceeds GPComp v1 limit";
  }

  std::vector<KVPair> merged(total_records);
  const auto wall_start = std::chrono::steady_clock::now();
  err = static_cast<cudaError_t>(
      launch_merge(run_ptrs.data(), run_sizes.data(),
                   static_cast<int>(num_runs), merged.data()));
  const auto wall_end = std::chrono::steady_clock::now();
  if (err != cudaSuccess) {
    return MakeCudaStatus("launch_merge", err);
  }

  Record* output_records = new (std::nothrow) Record[total_records];
  if (output_records == nullptr) {
    return "failed to allocate merged output buffer";
  }
  std::memcpy(output_records, merged.data(), total_records * sizeof(Record));

  out->records = output_records;
  out->num_records = total_records;
  out->kernel_ms = 0.0f;
  out->wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(wall_end - wall_start)
          .count());
  return "";
}

void FreeResult(MergeResult* result) {
  if (result == nullptr) {
    return;
  }
  delete[] result->records;
  result->records = nullptr;
  result->num_records = 0;
  result->kernel_ms = 0.0f;
  result->wall_ms = 0.0f;
}

void Close(Handle* handle) { delete handle; }

}  // namespace gpcomp_rocksdb_bridge
