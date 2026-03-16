#include "gpcomp_pipeline.cuh"

#include <chrono>
#include <cmath>
#include <cerrno>
#include <dirent.h>
#include <numeric>
#include <string>
#include <sys/stat.h>

struct RunSummary {
    double total_ms = 0.0;
    double read_parse_ms = 0.0;
    double write_ms = 0.0;
    CompactionStageTimes stage;
    float unpack_kernel_ms = 0.0f;
    float merge_kernel_ms = 0.0f;
    float bloom_kernel_ms = 0.0f;
    float pack_kernel_ms = 0.0f;
    size_t output_bytes = 0;
    size_t output_blocks = 0;
    size_t output_files = 0;
};

struct Stats {
    double min = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
};

static Stats compute_stats(const std::vector<double>& values)
{
    Stats s{};
    if (values.empty()) return s;
    s.min = *std::min_element(values.begin(), values.end());
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / (double)values.size();
    double accum = 0.0;
    for (double v : values) {
        double d = v - s.mean;
        accum += d * d;
    }
    s.stddev = std::sqrt(accum / (double)values.size());
    return s;
}

static std::vector<std::string> collect_sst_paths(const std::string& dataset_dir,
                                                  size_t             expected_count = 0)
{
    std::vector<std::string> paths;
    DIR* dir = opendir(dataset_dir.c_str());
    if (!dir) throw std::runtime_error("failed to open dataset dir");
    for (dirent* entry = readdir(dir); entry != nullptr; entry = readdir(dir)) {
        std::string name = entry->d_name;
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".sst") {
            paths.push_back(dataset_dir + "/" + name);
        }
    }
    closedir(dir);
    std::sort(paths.begin(), paths.end());
    if (expected_count != 0 && paths.size() != expected_count) {
        throw std::runtime_error("expected exactly 4 input SSTs in " + dataset_dir);
    }
    return paths;
}

static void ensure_dir(const std::string& dir)
{
    struct stat st{};
    if (stat(dir.c_str(), &st) == 0) return;
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("failed to create output directory");
    }
}

static void clear_sst_files_in_dir(const std::string& dir)
{
    DIR* handle = opendir(dir.c_str());
    if (!handle) return;
    for (dirent* entry = readdir(handle); entry != nullptr; entry = readdir(handle)) {
        std::string name = entry->d_name;
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".sst") {
            std::string path = dir + "/" + name;
            std::remove(path.c_str());
        }
    }
    closedir(handle);
}

static std::vector<std::string> write_output_set(const SSTBuildSet& output,
                                                 const std::string& dir,
                                                 const std::string& prefix)
{
    ensure_dir(dir);
    clear_sst_files_in_dir(dir);
    std::vector<std::string> paths;
    for (size_t i = 0; i < output.files.size(); ++i) {
        char name[256];
        std::snprintf(name, sizeof(name), "%s_%04zu.sst", prefix.c_str(), i);
        std::string path = dir + "/" + name;
        write_binary_file(path, output.files[i].file_bytes);
        paths.push_back(path);
    }
    return paths;
}

static bool compare_output_sets(const std::vector<std::string>& lhs,
                                const std::vector<std::string>& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (read_binary_file(lhs[i]) != read_binary_file(rhs[i])) return false;
    }
    return true;
}

static size_t total_output_bytes(const SSTBuildSet& output)
{
    size_t total = 0;
    for (const auto& file : output.files) total += file.file_bytes.size();
    return total;
}

static size_t total_output_blocks(const SSTBuildSet& output)
{
    size_t total = 0;
    for (const auto& file : output.files) total += file.data_meta.size();
    return total;
}

static std::vector<ParsedSST> load_inputs_with_timing(const std::vector<std::string>& paths, double& ms)
{
    auto t0 = std::chrono::steady_clock::now();
    std::vector<ParsedSST> parsed;
    parsed.reserve(paths.size());
    for (const auto& path : paths) parsed.push_back(read_sst_file(path));
    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return parsed;
}

static RunSummary run_cpu_once(const std::vector<std::string>& input_paths,
                               const std::string&               output_dir,
                               const std::string&               gpu_mode)
{
    RunSummary summary;
    auto total_start = std::chrono::steady_clock::now();

    std::vector<ParsedSST> inputs = load_inputs_with_timing(input_paths, summary.read_parse_ms);
    CPUCompactionResult cpu;
    if (gpu_mode == "q_paper" || gpu_mode == "q_paper_overlap") {
        cpu = cpu_q_compaction_paper_from_parsed(inputs);
    }
    else cpu = cpu_q_compaction_from_parsed(inputs);

    auto write_start = std::chrono::steady_clock::now();
    write_output_set(cpu.output, output_dir, "cpu_compacted");
    auto write_end = std::chrono::steady_clock::now();

    auto total_end = std::chrono::steady_clock::now();
    summary.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    summary.write_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();
    summary.stage = cpu.stage;
    summary.output_bytes = total_output_bytes(cpu.output);
    summary.output_blocks = total_output_blocks(cpu.output);
    summary.output_files = cpu.output.files.size();
    return summary;
}

static RunSummary run_gpu_once(const std::vector<std::string>& input_paths,
                               const std::string&               output_dir,
                               const std::string&               gpu_mode)
{
    RunSummary summary;
    auto total_start = std::chrono::steady_clock::now();

    std::vector<ParsedSST> inputs = load_inputs_with_timing(input_paths, summary.read_parse_ms);
    GPUCompactionResult gpu;
    if (gpu_mode == "q_pipeline") gpu = gpu_q_compaction_pipeline_from_parsed(inputs);
    else if (gpu_mode == "q_paper") gpu = gpu_q_compaction_paper_from_parsed(inputs);
    else if (gpu_mode == "q_paper_overlap") gpu = gpu_q_compaction_paper_overlap_from_parsed(inputs);
    else gpu = gpu_q_compaction_from_parsed(inputs);

    auto write_start = std::chrono::steady_clock::now();
    write_output_set(gpu.output, output_dir, "gpu_compacted");
    auto write_end = std::chrono::steady_clock::now();

    auto total_end = std::chrono::steady_clock::now();
    summary.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    summary.write_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();
    summary.stage = gpu.stage;
    summary.unpack_kernel_ms = gpu.unpack_kernel_ms;
    summary.merge_kernel_ms = gpu.merge_kernel_ms;
    summary.bloom_kernel_ms = gpu.bloom_kernel_ms;
    summary.pack_kernel_ms = gpu.pack_kernel_ms;
    summary.output_bytes = total_output_bytes(gpu.output);
    summary.output_blocks = total_output_blocks(gpu.output);
    summary.output_files = gpu.output.files.size();
    return summary;
}

int main(int argc, char** argv)
{
    std::string dataset_dir = "dataset";
    std::string out_dir = "results";
    std::string gpu_mode = "legacy";
    int runs = 3;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::runtime_error("missing CLI argument");
            return argv[++i];
        };
        if (arg == "--dataset") dataset_dir = next();
        else if (arg == "--out_dir") out_dir = next();
        else if (arg == "--runs") runs = std::stoi(next());
        else if (arg == "--gpu_mode") gpu_mode = next();
        else if (arg == "--help") {
            std::printf("Usage: %s [--dataset DIR] [--out_dir DIR] [--runs N] [--gpu_mode legacy|q_pipeline|q_paper|q_paper_overlap]\n",
                        argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    ensure_dir(out_dir);
    std::vector<std::string> input_paths = collect_sst_paths(dataset_dir, GP_NUM_INPUT_SSTS);

    cudaFree(0);

    std::printf("Q-compaction benchmark\n");
    std::printf("  input SSTs: %zu\n", input_paths.size());
    std::printf("  key bytes: %d  value bytes: %d  restart interval: %d  block size: %d\n",
                GP_KEY_BYTES, GP_VALUE_BYTES, GP_RESTART_INTERVAL, GP_DATA_BLOCK_BYTES);
    std::printf("  runs: %d  (timed)\n", runs);
    std::printf("  gpu mode: %s\n", gpu_mode.c_str());
    std::printf("  garbage collection: disabled (Q-compaction only)\n\n");

    std::vector<RunSummary> cpu_runs, gpu_runs;
    cpu_runs.reserve(runs);
    gpu_runs.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        char cpu_dir[256];
        char gpu_dir[256];
        std::snprintf(cpu_dir, sizeof(cpu_dir), "%s/cpu_run%d", out_dir.c_str(), r);
        std::snprintf(gpu_dir, sizeof(gpu_dir), "%s/gpu_run%d", out_dir.c_str(), r);
        ensure_dir(cpu_dir);
        ensure_dir(gpu_dir);
        cpu_runs.push_back(run_cpu_once(input_paths, cpu_dir, gpu_mode));
        gpu_runs.push_back(run_gpu_once(input_paths, gpu_dir, gpu_mode));
    }

    std::vector<double> cpu_totals, gpu_totals, cpu_reads, gpu_reads;
    for (int r = 0; r < runs; ++r) {
        cpu_totals.push_back(cpu_runs[r].total_ms);
        gpu_totals.push_back(gpu_runs[r].total_ms);
        cpu_reads.push_back(cpu_runs[r].read_parse_ms);
        gpu_reads.push_back(gpu_runs[r].read_parse_ms);
    }

    Stats cpu_total_stats = compute_stats(cpu_totals);
    Stats gpu_total_stats = compute_stats(gpu_totals);
    Stats cpu_read_stats = compute_stats(cpu_reads);
    Stats gpu_read_stats = compute_stats(gpu_reads);

    std::vector<std::string> cpu_last, gpu_last;
    cpu_last = collect_sst_paths(out_dir + "/cpu_run" + std::to_string(runs - 1));
    gpu_last = collect_sst_paths(out_dir + "/gpu_run" + std::to_string(runs - 1));
    bool outputs_match = compare_output_sets(cpu_last, gpu_last);

    size_t cpu_best_idx = (size_t)(std::min_element(cpu_totals.begin(), cpu_totals.end()) - cpu_totals.begin());
    size_t gpu_best_idx = (size_t)(std::min_element(gpu_totals.begin(), gpu_totals.end()) - gpu_totals.begin());
    const RunSummary& cpu_best = cpu_runs[cpu_best_idx];
    const RunSummary& gpu_best = gpu_runs[gpu_best_idx];

    std::printf("CPU total: min=%.2f ms  mean=%.2f +- %.2f ms\n",
                cpu_total_stats.min, cpu_total_stats.mean, cpu_total_stats.stddev);
    std::printf("GPU total: min=%.2f ms  mean=%.2f +- %.2f ms\n",
                gpu_total_stats.min, gpu_total_stats.mean, gpu_total_stats.stddev);
    std::printf("Speedup: %.2fx (min totals)\n", cpu_total_stats.min / gpu_total_stats.min);
    std::printf("Output identical: %s\n\n", outputs_match ? "PASS" : "FAIL");

    std::printf("Best CPU run breakdown (ms):\n");
    std::printf("  read+parse %.2f  unpack %.2f  sort(merge) %.2f  plan %.2f  bloom %.2f  pack+assemble %.2f  write %.2f\n",
                cpu_best.read_parse_ms, cpu_best.stage.unpack_ms, cpu_best.stage.merge_ms,
                cpu_best.stage.planning_ms, cpu_best.stage.bloom_ms, cpu_best.stage.pack_ms, cpu_best.write_ms);
    std::printf("  output bytes %zu  data blocks %zu  output files %zu\n\n",
                cpu_best.output_bytes, cpu_best.output_blocks, cpu_best.output_files);

    std::printf("Best GPU run breakdown (ms):\n");
    std::printf("  read+parse %.2f  unpack %.2f  sort(merge) %.2f  plan %.2f  bloom %.2f  pack+assemble %.2f  write %.2f\n",
                gpu_best.read_parse_ms, gpu_best.stage.unpack_ms, gpu_best.stage.merge_ms,
                gpu_best.stage.planning_ms, gpu_best.stage.bloom_ms, gpu_best.stage.pack_ms, gpu_best.write_ms);
    std::printf("  kernel-only: unpack %.2f  sort(merge) %.2f  bloom %.2f  pack %.2f\n",
                gpu_best.unpack_kernel_ms, gpu_best.merge_kernel_ms, gpu_best.bloom_kernel_ms, gpu_best.pack_kernel_ms);
    std::printf("  output bytes %zu  data blocks %zu  output files %zu\n",
                gpu_best.output_bytes, gpu_best.output_blocks, gpu_best.output_files);

    return outputs_match ? 0 : 1;
}
