/*
 * gpu_io_bench.cu  –  cuFile-based IO benchmark for GPU vs CPU compaction IO
 *
 * Simulates the IO pattern of a single LSM-tree compaction:
 *   READ  phase: read  NUM_L0_READ  "L0" files  (each of size l0_size)
 *   WRITE phase: write NUM_L1_WRITE "L1" files  (each of size l0_size)
 *
 * Two paths are benchmarked:
 *
 *   CPU path (direct IO):
 *       SSD ──read()──▸ CPU-RAM ──write()──▸ SSD
 *
 *   GPU path (cuFile compat / bounce-buffer):
 *       SSD ──cuFileRead()──▸ GPU-RAM ──cuFileWrite()──▸ SSD
 *       (internally:  SSD → CPU-RAM → GPU → CPU-RAM → SSD)
 *
 * On GeForce GPUs (no GDS kernel driver) cuFile transparently falls back to a
 * bounce-buffer through host memory, which is exactly the extra-hop overhead
 * we want to measure.
 *
 * Build:
 *   make            (see accompanying Makefile)
 *
 * Usage:
 *   ./gpu_io_bench --data_dir /tmp/bench --l0_size 8388608 \
 *                  --num_l0_read 4 --num_l1_write 3 --reps 10 \
 *                  --alignment 4096 --direct_io 1 --gpu_device 0 \
 *                  --csv results.csv
 */

#include <cuda_runtime.h>
#include <cufile.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <getopt.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

/* ─── helpers ─────────────────────────────────────────────────────── */

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                     \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CUFILE_CHECK(status, msg)                                          \
  do {                                                                     \
    if ((status).err != CU_FILE_SUCCESS) {                                 \
      fprintf(stderr, "cuFile error %s:%d – %s (cu_err=%d)\n",            \
              __FILE__, __LINE__, (msg), (int)(status).err);               \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

static double now_us() {
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::micro>(
             clk::now().time_since_epoch())
      .count();
}

/* ─── aligned host buffer (for CPU direct-IO path) ───────────────── */

static void *alloc_aligned(size_t size, size_t alignment) {
  void *p = nullptr;
  if (posix_memalign(&p, alignment, size) != 0) {
    perror("posix_memalign");
    exit(EXIT_FAILURE);
  }
  return p;
}

/* ─── file helpers ────────────────────────────────────────────────── */

static std::string make_path(const std::string &dir, const char *prefix,
                             int idx) {
  char buf[512];
  snprintf(buf, sizeof(buf), "%s/%s_%04d.dat", dir.c_str(), prefix, idx);
  return std::string(buf);
}

/* Fill a file on disk with deterministic data so reads are realistic.
 * Uses O_DIRECT when direct_io is true. */
static void create_data_file(const std::string &path, size_t size,
                             size_t alignment, bool direct_io) {
  int flags = O_CREAT | O_WRONLY | O_TRUNC;
  if (direct_io) flags |= O_DIRECT;
  int fd = open(path.c_str(), flags, 0644);
  if (fd < 0) {
    fprintf(stderr, "open(%s) for create: %s\n", path.c_str(),
            strerror(errno));
    exit(EXIT_FAILURE);
  }

  /* Write in 1 MB chunks. */
  const size_t chunk = 1 << 20;
  void *buf = alloc_aligned(chunk, alignment);
  /* Fill with a repeating byte pattern so data is deterministic. */
  memset(buf, 0xAB, chunk);

  size_t remaining = size;
  while (remaining > 0) {
    size_t to_write = remaining < chunk ? remaining : chunk;
    /* For O_DIRECT the write size must be a multiple of alignment. */
    size_t aligned_write = (to_write + alignment - 1) & ~(alignment - 1);
    if (aligned_write > chunk) aligned_write = chunk;
    ssize_t n = write(fd, buf, aligned_write);
    if (n < 0) {
      perror("write (create_data_file)");
      close(fd);
      free(buf);
      exit(EXIT_FAILURE);
    }
    remaining -= to_write;
  }
  /* Truncate to exact size (O_DIRECT may have written padding). */
  if (ftruncate(fd, (off_t)size) != 0) {
    perror("ftruncate");
  }
  fsync(fd);
  close(fd);
  free(buf);
}

/* ─── CPU IO benchmark ────────────────────────────────────────────── */

struct BenchResult {
  double read_us;
  double write_us;
  double total_us;
};

static BenchResult bench_cpu(const std::string &data_dir, size_t l0_size,
                             int num_read, int num_write, size_t alignment,
                             bool direct_io) {
  int flags_r = O_RDONLY;
  int flags_w = O_CREAT | O_WRONLY | O_TRUNC;
  if (direct_io) {
    flags_r |= O_DIRECT;
    flags_w |= O_DIRECT;
  }

  /* Allocate a host buffer large enough for one file. */
  void *buf = alloc_aligned(l0_size, alignment);

  /* ── READ phase ── */
  double t0 = now_us();
  for (int i = 0; i < num_read; ++i) {
    std::string path = make_path(data_dir, "l0_input", i);
    int fd = open(path.c_str(), flags_r);
    if (fd < 0) {
      fprintf(stderr, "open(%s): %s\n", path.c_str(), strerror(errno));
      exit(EXIT_FAILURE);
    }
    size_t remaining = l0_size;
    size_t offset = 0;
    while (remaining > 0) {
      size_t to_read = remaining;
      ssize_t n = pread(fd, (char *)buf + offset, to_read, (off_t)offset);
      if (n <= 0) {
        if (n < 0) perror("pread (cpu)");
        break;
      }
      remaining -= (size_t)n;
      offset += (size_t)n;
    }
    close(fd);
  }
  double t_read = now_us();

  /* ── WRITE phase ── */
  /* Reuse the buffer filled from the last read (realistic: data was just
   * merge-sorted in RAM). */
  memset(buf, 0xCD, l0_size); /* simulate output data */
  for (int i = 0; i < num_write; ++i) {
    std::string path = make_path(data_dir, "l1_output_cpu", i);
    int fd = open(path.c_str(), flags_w, 0644);
    if (fd < 0) {
      fprintf(stderr, "open(%s): %s\n", path.c_str(), strerror(errno));
      exit(EXIT_FAILURE);
    }
    size_t remaining = l0_size;
    size_t offset = 0;
    while (remaining > 0) {
      size_t to_write = remaining;
      /* O_DIRECT needs aligned size for the last chunk. */
      size_t aligned_write =
          (to_write + alignment - 1) & ~(alignment - 1);
      ssize_t n = pwrite(fd, (char *)buf + offset, aligned_write,
                          (off_t)offset);
      if (n <= 0) {
        if (n < 0) perror("pwrite (cpu)");
        break;
      }
      remaining -= to_write;
      offset += to_write;
    }
    if (ftruncate(fd, (off_t)l0_size) != 0) perror("ftruncate");
    fsync(fd);
    close(fd);
  }
  double t_write = now_us();

  free(buf);

  BenchResult r;
  r.read_us = t_read - t0;
  r.write_us = t_write - t_read;
  r.total_us = t_write - t0;
  return r;
}

/* ─── GPU IO benchmark (cuFile) ───────────────────────────────────── */

static BenchResult bench_gpu(const std::string &data_dir, size_t l0_size,
                             int num_read, int num_write, size_t alignment,
                             int gpu_device) {
  CUDA_CHECK(cudaSetDevice(gpu_device));

  /* Allocate GPU buffer for one file. */
  void *d_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, l0_size));

  /* Register the GPU buffer with cuFile. */
  CUfileError_t status = cuFileBufRegister(d_buf, l0_size, 0);
  CUFILE_CHECK(status, "cuFileBufRegister");

  /* ── READ phase ── */
  double t0 = now_us();
  for (int i = 0; i < num_read; ++i) {
    std::string path = make_path(data_dir, "l0_input", i);
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
      fprintf(stderr, "open(%s): %s\n", path.c_str(), strerror(errno));
      exit(EXIT_FAILURE);
    }

    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t cfh;
    status = cuFileHandleRegister(&cfh, &descr);
    CUFILE_CHECK(status, "cuFileHandleRegister (read)");

    ssize_t n = cuFileRead(cfh, d_buf, l0_size, 0 /* file_offset */,
                           0 /* buf_offset */);
    if (n < 0) {
      fprintf(stderr, "cuFileRead error: %zd\n", n);
      exit(EXIT_FAILURE);
    }

    cuFileHandleDeregister(cfh);
    close(fd);
  }
  double t_read = now_us();

  /* Optionally: launch a trivial kernel to "touch" data on GPU and mimic
   * a merge step.  We skip actual computation since this benchmark is
   * IO-only. */

  /* ── WRITE phase ── */
  /* Fill device buffer with deterministic pattern for writes. */
  CUDA_CHECK(cudaMemset(d_buf, 0xCD, l0_size));

  for (int i = 0; i < num_write; ++i) {
    std::string path = make_path(data_dir, "l1_output_gpu", i);
    int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT,
                  0644);
    if (fd < 0) {
      fprintf(stderr, "open(%s): %s\n", path.c_str(), strerror(errno));
      exit(EXIT_FAILURE);
    }

    /* Pre-allocate the file so cuFileWrite has space. */
    if (fallocate(fd, 0, 0, (off_t)l0_size) != 0) {
      perror("fallocate");
      /* Non-fatal: continue anyway. */
    }

    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t cfh;
    status = cuFileHandleRegister(&cfh, &descr);
    CUFILE_CHECK(status, "cuFileHandleRegister (write)");

    ssize_t n = cuFileWrite(cfh, d_buf, l0_size, 0 /* file_offset */,
                            0 /* buf_offset */);
    if (n < 0) {
      fprintf(stderr, "cuFileWrite error: %zd\n", n);
      exit(EXIT_FAILURE);
    }

    cuFileHandleDeregister(cfh);
    fsync(fd);
    close(fd);
  }
  double t_write = now_us();

  cuFileBufDeregister(d_buf);
  CUDA_CHECK(cudaFree(d_buf));

  BenchResult r;
  r.read_us = t_read - t0;
  r.write_us = t_write - t_read;
  r.total_us = t_write - t0;
  return r;
}

/* ─── drop page cache (best-effort) ──────────────────────────────── */

static void drop_caches() {
  /* Requires root or appropriate sysctl. */
  int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
  if (fd >= 0) {
    const char *val = "3\n";
    ssize_t written = write(fd, val, 2);
    (void)written;
    close(fd);
  }
  sync();
}

/* ─── main ────────────────────────────────────────────────────────── */

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [options]\n"
          "  --data_dir DIR       temp data directory  [/tmp/gpu_io_bench]\n"
          "  --l0_size  BYTES     size of each L0 file [8388608]\n"
          "  --num_l0_read  N     L0 files to read     [4]\n"
          "  --num_l1_write N     L1 files to write    [3]\n"
          "  --alignment  N       O_DIRECT alignment   [4096]\n"
          "  --direct_io  0|1     use O_DIRECT for CPU [1]\n"
          "  --gpu_device N       CUDA device ordinal  [0]\n"
          "  --reps N             repetitions per cfg  [10]\n"
          "  --csv FILE           append results CSV   [stdout]\n"
          "  --drop_caches 0|1    drop page cache      [1]\n",
          prog);
}

int main(int argc, char **argv) {
  /* defaults */
  std::string data_dir = "/tmp/gpu_io_bench";
  size_t l0_size = 8388608;      /* 8 MB */
  int num_l0_read = 4;
  int num_l1_write = 3;
  size_t alignment = 4096;
  bool direct_io = true;
  int gpu_device = 0;
  int reps = 10;
  std::string csv_path;
  bool do_drop_caches = true;

  static struct option long_opts[] = {
      {"data_dir", required_argument, nullptr, 'd'},
      {"l0_size", required_argument, nullptr, 's'},
      {"num_l0_read", required_argument, nullptr, 'r'},
      {"num_l1_write", required_argument, nullptr, 'w'},
      {"alignment", required_argument, nullptr, 'a'},
      {"direct_io", required_argument, nullptr, 'D'},
      {"gpu_device", required_argument, nullptr, 'g'},
      {"reps", required_argument, nullptr, 'n'},
      {"csv", required_argument, nullptr, 'c'},
      {"drop_caches", required_argument, nullptr, 'C'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "hd:s:r:w:a:D:g:n:c:C:", long_opts,
                            nullptr)) != -1) {
    switch (opt) {
      case 'd': data_dir = optarg; break;
      case 's': l0_size = (size_t)strtoull(optarg, nullptr, 10); break;
      case 'r': num_l0_read = atoi(optarg); break;
      case 'w': num_l1_write = atoi(optarg); break;
      case 'a': alignment = (size_t)strtoull(optarg, nullptr, 10); break;
      case 'D': direct_io = (atoi(optarg) != 0); break;
      case 'g': gpu_device = atoi(optarg); break;
      case 'n': reps = atoi(optarg); break;
      case 'c': csv_path = optarg; break;
      case 'C': do_drop_caches = (atoi(optarg) != 0); break;
      default: usage(argv[0]); return EXIT_FAILURE;
    }
  }

  /* Validate l0_size is a multiple of alignment for O_DIRECT. */
  if (l0_size % alignment != 0) {
    fprintf(stderr,
            "warning: l0_size (%zu) is not a multiple of alignment (%zu).\n"
            "         Rounding up.\n",
            l0_size, alignment);
    l0_size = (l0_size + alignment - 1) & ~(alignment - 1);
  }

  /* Print configuration. */
  printf("=== GPU IO Bench ===\n");
  printf("data_dir      : %s\n", data_dir.c_str());
  printf("l0_size       : %zu bytes (%.1f MB)\n", l0_size,
         l0_size / (1024.0 * 1024.0));
  printf("num_l0_read   : %d\n", num_l0_read);
  printf("num_l1_write  : %d\n", num_l1_write);
  printf("alignment     : %zu\n", alignment);
  printf("direct_io     : %s\n", direct_io ? "true" : "false");
  printf("gpu_device    : %d\n", gpu_device);
  printf("reps          : %d\n", reps);
  printf("drop_caches   : %s\n", do_drop_caches ? "true" : "false");
  if (!csv_path.empty())
    printf("csv           : %s\n", csv_path.c_str());
  printf("\n");

  /* Create data directory. */
  {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", data_dir.c_str());
    if (system(cmd) != 0) {
      fprintf(stderr, "Failed to create data_dir: %s\n", data_dir.c_str());
      return EXIT_FAILURE;
    }
  }

  /* Initialize cuFile driver. */
  CUfileError_t cuf_status = cuFileDriverOpen();
  if (cuf_status.err != CU_FILE_SUCCESS) {
    fprintf(stderr,
            "cuFileDriverOpen failed (err=%d). cuFile may not be supported.\n",
            (int)cuf_status.err);
    fprintf(stderr, "Continuing anyway – compat mode may still work.\n");
    /* Don't exit; on some setups compat mode still works. */
  }

  /* Print GPU info. */
  {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_device));
    printf("GPU           : %s (compute %d.%d)\n", prop.name, prop.major,
           prop.minor);
    printf("GPU memory    : %.0f MB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("\n");
  }

  /* Create input L0 files. */
  printf("Creating %d input L0 files (%.1f MB each) ...\n", num_l0_read,
         l0_size / (1024.0 * 1024.0));
  for (int i = 0; i < num_l0_read; ++i) {
    create_data_file(make_path(data_dir, "l0_input", i), l0_size, alignment,
                     direct_io);
  }
  printf("Done.\n\n");

  /* Open CSV output. */
  FILE *csv_fp = nullptr;
  if (!csv_path.empty()) {
    bool needs_header = (access(csv_path.c_str(), F_OK) != 0);
    csv_fp = fopen(csv_path.c_str(), "a");
    if (!csv_fp) {
      fprintf(stderr, "Cannot open CSV: %s\n", csv_path.c_str());
      return EXIT_FAILURE;
    }
    if (needs_header) {
      fprintf(csv_fp,
              "path,l0_size_bytes,l0_size_mb,num_l0_read,num_l1_write,"
              "rep,read_us,write_us,total_us,direct_io\n");
    }
  }

  /* ── Run benchmarks ── */

  printf("%-5s  %-10s  %12s  %12s  %12s\n", "Path", "Rep", "Read(us)",
         "Write(us)", "Total(us)");
  printf("%-5s  %-10s  %12s  %12s  %12s\n", "-----", "----------",
         "------------", "------------", "------------");

  for (int rep = 0; rep < reps; ++rep) {
    /* ── CPU path ── */
    if (do_drop_caches) drop_caches();
    BenchResult cpu = bench_cpu(data_dir, l0_size, num_l0_read, num_l1_write,
                                alignment, direct_io);
    printf("CPU    rep=%02d    %12.1f  %12.1f  %12.1f\n", rep, cpu.read_us,
           cpu.write_us, cpu.total_us);

    if (csv_fp) {
      fprintf(csv_fp, "cpu,%zu,%.1f,%d,%d,%d,%.1f,%.1f,%.1f,%s\n", l0_size,
              l0_size / (1024.0 * 1024.0), num_l0_read, num_l1_write, rep,
              cpu.read_us, cpu.write_us, cpu.total_us,
              direct_io ? "true" : "false");
    }

    /* ── GPU path ── */
    if (do_drop_caches) drop_caches();
    BenchResult gpu = bench_gpu(data_dir, l0_size, num_l0_read, num_l1_write,
                                alignment, gpu_device);
    printf("GPU    rep=%02d    %12.1f  %12.1f  %12.1f\n", rep, gpu.read_us,
           gpu.write_us, gpu.total_us);

    if (csv_fp) {
      fprintf(csv_fp, "gpu,%zu,%.1f,%d,%d,%d,%.1f,%.1f,%.1f,%s\n", l0_size,
              l0_size / (1024.0 * 1024.0), num_l0_read, num_l1_write, rep,
              gpu.read_us, gpu.write_us, gpu.total_us, "true");
    }
  }

  if (csv_fp) {
    fclose(csv_fp);
    printf("\nResults appended to %s\n", csv_path.c_str());
  }

  /* Cleanup temp files. */
  for (int i = 0; i < num_l0_read; ++i) {
    unlink(make_path(data_dir, "l0_input", i).c_str());
  }
  for (int i = 0; i < num_l1_write; ++i) {
    unlink(make_path(data_dir, "l1_output_cpu", i).c_str());
    unlink(make_path(data_dir, "l1_output_gpu", i).c_str());
  }

  cuFileDriverClose();
  printf("\nDone.\n");
  return EXIT_SUCCESS;
}
