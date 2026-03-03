/*
 * gpu_file_replay_bench.cu  -  cuFile-backed file replay helper.
 *
 * This helper is used by the RocksDB dummy GPU compaction benchmark:
 * it stages a manifest of SST files through GPU buffers using
 *
 *   (cuFileRead or pread) -> trivial device copy kernel -> (cuFileWrite or pwrite)
 *
 * Usage:
 *   ./gpu_file_replay_bench --manifest files.tsv --gpu_device 0 --alignment 4096
 *
 * Manifest format (tab-separated):
 *   <source_path>\t<destination_path>\t<size_bytes>
 * The destination field may be empty, which means the file is only staged into
 * GPU memory and not written back out.
 *
 * On success the tool prints a single machine-readable line:
 *   GPU_FILE_REPLAY_METRICS total_us=... stage_files=... copy_files=...
 *   stage_total_us=... copy_total_us=...
 */

#include <cuda_runtime.h>
#include <cufile.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                     \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CUFILE_CHECK(status, msg)                                          \
  do {                                                                     \
    if ((status).err != CU_FILE_SUCCESS) {                                 \
      fprintf(stderr, "cuFile error %s:%d - %s (cu_err=%d)\n",             \
              __FILE__, __LINE__, (msg), (int)(status).err);               \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

struct ReplayJob {
  std::string source;
  std::string destination;
  size_t size = 0;
};

struct ReplayMetrics {
  size_t files = 0;
  size_t bytes = 0;
  double read_us = 0.0;
  double host_to_device_us = 0.0;
  double kernel_us = 0.0;
  double device_to_host_us = 0.0;
  double write_us = 0.0;
  double sync_us = 0.0;
  double total_us = 0.0;
};

static double now_us() {
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::micro>(
             clk::now().time_since_epoch())
      .count();
}

static size_t align_down(size_t value, size_t alignment) {
  return value - (value % alignment);
}

static size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__global__ void passthrough_copy_kernel(const unsigned char* src,
                                        unsigned char* dst, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

static std::vector<ReplayJob> load_manifest(const std::string& manifest_path) {
  std::ifstream manifest(manifest_path);
  if (!manifest) {
    fprintf(stderr, "failed to open manifest %s\n", manifest_path.c_str());
    exit(EXIT_FAILURE);
  }

  std::vector<ReplayJob> jobs;
  std::string line;
  while (std::getline(manifest, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream iss(line);
    ReplayJob job;
    std::string size_str;

    if (!std::getline(iss, job.source, '\t') ||
        !std::getline(iss, job.destination, '\t') ||
        !std::getline(iss, size_str)) {
      fprintf(stderr, "invalid manifest line: %s\n", line.c_str());
      exit(EXIT_FAILURE);
    }

    char* end = nullptr;
    unsigned long long parsed = strtoull(size_str.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') {
      fprintf(stderr, "invalid file size in manifest: %s\n", size_str.c_str());
      exit(EXIT_FAILURE);
    }
    job.size = static_cast<size_t>(parsed);
    jobs.push_back(std::move(job));
  }

  if (jobs.empty()) {
    fprintf(stderr, "manifest %s did not contain any jobs\n",
            manifest_path.c_str());
    exit(EXIT_FAILURE);
  }

  return jobs;
}

static void replay_one_file(const ReplayJob& job, bool copy_to_destination,
                            size_t alignment, size_t direct_buffer_bytes,
                            size_t buffer_bytes, void* d_in, void* d_out,
                            unsigned char* h_buf, ReplayMetrics* metrics) {
  if (copy_to_destination && job.destination.empty()) {
    fprintf(stderr, "copy destination missing for %s\n",
            job.source.c_str());
    exit(EXIT_FAILURE);
  }

  const double file_t0 = now_us();
  metrics->files += 1;
  metrics->bytes += job.size;

  int src_fd = open(job.source.c_str(), O_RDONLY);
  if (src_fd < 0) {
    fprintf(stderr, "open(%s): %s\n", job.source.c_str(), strerror(errno));
    exit(EXIT_FAILURE);
  }

  int dst_fd = -1;
  if (copy_to_destination) {
    dst_fd = open(job.destination.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (dst_fd < 0) {
      fprintf(stderr, "open(%s): %s\n", job.destination.c_str(), strerror(errno));
      close(src_fd);
      exit(EXIT_FAILURE);
    }
    if (ftruncate(dst_fd, static_cast<off_t>(align_up(job.size, alignment))) != 0) {
      perror("ftruncate");
      close(src_fd);
      close(dst_fd);
      exit(EXIT_FAILURE);
    }
  }

  int src_fd_direct = -1;
  int dst_fd_direct = -1;
  CUfileHandle_t src_cfh = nullptr;
  CUfileHandle_t dst_cfh = nullptr;
  const bool use_direct = direct_buffer_bytes > 0 && job.size >= alignment;

  if (use_direct) {
    src_fd_direct = open(job.source.c_str(), O_RDONLY | O_DIRECT);
    if (src_fd_direct < 0) {
      perror("open(O_DIRECT source)");
      close(src_fd);
      if (dst_fd >= 0) {
        close(dst_fd);
      }
      exit(EXIT_FAILURE);
    }

    if (copy_to_destination) {
      dst_fd_direct = open(job.destination.c_str(), O_WRONLY | O_DIRECT);
      if (dst_fd_direct < 0) {
        perror("open(O_DIRECT destination)");
        close(src_fd);
        close(src_fd_direct);
        close(dst_fd);
        exit(EXIT_FAILURE);
      }
    }

    CUfileDescr_t src_descr;
    memset(&src_descr, 0, sizeof(src_descr));
    src_descr.handle.fd = src_fd_direct;
    src_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileDescr_t dst_descr;
    memset(&dst_descr, 0, sizeof(dst_descr));
    CUfileError_t status = cuFileHandleRegister(&src_cfh, &src_descr);
    CUFILE_CHECK(status, "cuFileHandleRegister(source)");
    if (copy_to_destination) {
      dst_descr.handle.fd = dst_fd_direct;
      dst_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      status = cuFileHandleRegister(&dst_cfh, &dst_descr);
      CUFILE_CHECK(status, "cuFileHandleRegister(destination)");
    }
  }

  size_t offset = 0;
  while (offset < job.size) {
    const size_t remaining = job.size - offset;
    size_t chunk = (remaining < buffer_bytes) ? remaining : buffer_bytes;
    bool use_direct_chunk = false;
    if (use_direct && chunk >= alignment) {
      chunk = align_down(chunk, alignment);
      use_direct_chunk = (chunk > 0);
    }

    double t0 = now_us();
    if (use_direct_chunk) {
      const ssize_t n =
          cuFileRead(src_cfh, d_in, chunk, static_cast<off_t>(offset), 0);
      if (n != static_cast<ssize_t>(chunk)) {
        fprintf(stderr, "cuFileRead failed for %s (got %zd, expected %zu)\n",
                job.source.c_str(), n, chunk);
        exit(EXIT_FAILURE);
      }
    } else {
      const ssize_t n =
          pread(src_fd, h_buf, chunk, static_cast<off_t>(offset));
      if (n != static_cast<ssize_t>(chunk)) {
        fprintf(stderr, "pread failed for %s (got %zd, expected %zu)\n",
                job.source.c_str(), n, chunk);
        exit(EXIT_FAILURE);
      }
      metrics->read_us += now_us() - t0;
      t0 = now_us();
      CUDA_CHECK(cudaMemcpy(d_in, h_buf, chunk, cudaMemcpyHostToDevice));
      metrics->host_to_device_us += now_us() - t0;
    }
    if (use_direct_chunk) {
      metrics->read_us += now_us() - t0;
    }

    t0 = now_us();
    const int threads = 256;
    const int blocks = static_cast<int>((chunk + threads - 1) / threads);
    passthrough_copy_kernel<<<blocks, threads>>>(
        static_cast<unsigned char*>(d_in),
        static_cast<unsigned char*>(d_out), chunk);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    metrics->kernel_us += now_us() - t0;

    if (copy_to_destination) {
      t0 = now_us();
      if (use_direct_chunk) {
        const ssize_t n =
            cuFileWrite(dst_cfh, d_out, chunk, static_cast<off_t>(offset), 0);
        if (n != static_cast<ssize_t>(chunk)) {
          fprintf(stderr, "cuFileWrite failed for %s (got %zd, expected %zu)\n",
                  job.destination.c_str(), n, chunk);
          exit(EXIT_FAILURE);
        }
        metrics->write_us += now_us() - t0;
      } else {
        CUDA_CHECK(cudaMemcpy(h_buf, d_out, chunk, cudaMemcpyDeviceToHost));
        metrics->device_to_host_us += now_us() - t0;

        t0 = now_us();
        const ssize_t n =
            pwrite(dst_fd, h_buf, chunk, static_cast<off_t>(offset));
        if (n != static_cast<ssize_t>(chunk)) {
          fprintf(stderr, "pwrite failed for %s (got %zd, expected %zu)\n",
                  job.destination.c_str(), n, chunk);
          exit(EXIT_FAILURE);
        }
        metrics->write_us += now_us() - t0;
      }
    }

    offset += chunk;
  }

  metrics->total_us += now_us() - file_t0;

  if (src_cfh != nullptr) {
    cuFileHandleDeregister(src_cfh);
  }
  if (dst_cfh != nullptr) {
    cuFileHandleDeregister(dst_cfh);
  }
  if (src_fd_direct >= 0) {
    close(src_fd_direct);
  }
  if (dst_fd_direct >= 0) {
    close(dst_fd_direct);
  }
  close(src_fd);
  if (dst_fd >= 0) {
    if (ftruncate(dst_fd, static_cast<off_t>(job.size)) != 0) {
      perror("ftruncate");
      close(dst_fd);
      exit(EXIT_FAILURE);
    }
    const double t0 = now_us();
    fsync(dst_fd);
    metrics->sync_us += now_us() - t0;
    close(dst_fd);
  }
}

int main(int argc, char** argv) {
  std::string manifest_path;
  int gpu_device = 0;
  size_t alignment = 4096;
  size_t buffer_bytes = 8ULL << 20;

  static struct option long_opts[] = {
      {"manifest", required_argument, nullptr, 'm'},
      {"gpu_device", required_argument, nullptr, 'g'},
      {"alignment", required_argument, nullptr, 'a'},
      {"buffer_bytes", required_argument, nullptr, 'b'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'm':
        manifest_path = optarg;
        break;
      case 'g':
        gpu_device = atoi(optarg);
        break;
      case 'a':
        alignment = static_cast<size_t>(strtoull(optarg, nullptr, 10));
        break;
      case 'b':
        buffer_bytes = static_cast<size_t>(strtoull(optarg, nullptr, 10));
        break;
      default:
        fprintf(stderr,
                "usage: %s --manifest <path> [--gpu_device N] [--alignment N] "
                "[--buffer_bytes N]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
  }

  if (manifest_path.empty()) {
    fprintf(stderr, "error: --manifest is required\n");
    return EXIT_FAILURE;
  }
  if (alignment == 0) {
    fprintf(stderr, "error: --alignment must be > 0\n");
    return EXIT_FAILURE;
  }
  if (buffer_bytes == 0) {
    fprintf(stderr, "error: --buffer_bytes must be > 0\n");
    return EXIT_FAILURE;
  }

  const std::vector<ReplayJob> jobs = load_manifest(manifest_path);
  const size_t direct_buffer_bytes = align_down(buffer_bytes, alignment);

  CUDA_CHECK(cudaSetDevice(gpu_device));

  CUfileError_t status = cuFileDriverOpen();
  CUFILE_CHECK(status, "cuFileDriverOpen");

  void* d_in = nullptr;
  void* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, buffer_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, buffer_bytes));
  status = cuFileBufRegister(d_in, buffer_bytes, 0);
  CUFILE_CHECK(status, "cuFileBufRegister(d_in)");
  status = cuFileBufRegister(d_out, buffer_bytes, 0);
  CUFILE_CHECK(status, "cuFileBufRegister(d_out)");

  unsigned char* h_buf = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_buf, buffer_bytes));

  ReplayMetrics stage_metrics;
  ReplayMetrics copy_metrics;
  const double pipeline_t0 = now_us();
  for (const auto& job : jobs) {
    const bool copy_to_destination = !job.destination.empty();
    replay_one_file(job, copy_to_destination, alignment, direct_buffer_bytes,
                    buffer_bytes, d_in, d_out, h_buf,
                    copy_to_destination ? &copy_metrics : &stage_metrics);
  }
  const double pipeline_total_us = now_us() - pipeline_t0;

  CUDA_CHECK(cudaFreeHost(h_buf));
  cuFileBufDeregister(d_in);
  cuFileBufDeregister(d_out);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  cuFileDriverClose();

  printf(
      "GPU_FILE_REPLAY_METRICS total_us=%.1f "
      "stage_files=%zu stage_bytes=%zu stage_total_us=%.1f "
      "stage_read_us=%.1f stage_h2d_us=%.1f stage_kernel_us=%.1f "
      "copy_files=%zu copy_bytes=%zu copy_total_us=%.1f "
      "copy_read_us=%.1f copy_h2d_us=%.1f copy_kernel_us=%.1f "
      "copy_d2h_us=%.1f copy_write_us=%.1f copy_sync_us=%.1f\n",
      pipeline_total_us, stage_metrics.files, stage_metrics.bytes,
      stage_metrics.total_us, stage_metrics.read_us,
      stage_metrics.host_to_device_us, stage_metrics.kernel_us,
      copy_metrics.files, copy_metrics.bytes, copy_metrics.total_us,
      copy_metrics.read_us, copy_metrics.host_to_device_us,
      copy_metrics.kernel_us, copy_metrics.device_to_host_us,
      copy_metrics.write_us, copy_metrics.sync_us);
  return EXIT_SUCCESS;
}
