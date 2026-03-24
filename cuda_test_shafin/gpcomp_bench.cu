#include "gpcomp_pipeline.cuh"

#include <sys/resource.h>
static double get_cpu_time_ms() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000.0 +
           (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1000.0;
}

#include <chrono>
#include <cmath>
#include <cerrno>
#include <dirent.h>
#include <numeric>
#include <string>
#include <sys/stat.h>

struct RunSummary {
    double total_ms = 0.0;
    double cpu_time_ms = 0.0;
    double read_parse_ms = 0.0;
    double write_ms = 0.0;
    double pipeline_wall_ms = 0.0;
    double pipeline_cpu_time_ms = 0.0;
    CompactionStageTimes stage;
    float unpack_kernel_ms = 0.0f;
    float merge_kernel_ms = 0.0f;
    float bloom_kernel_ms = 0.0f;
    float pack_kernel_ms = 0.0f;
    size_t output_bytes = 0;
    size_t output_blocks = 0;
    size_t output_files = 0;
    size_t input_bytes = 0;
    size_t h2d_lower_bound_bytes = 0;
    size_t d2h_lower_bound_bytes = 0;
    double unpack_h2d_ms = 0.0;
    double unpack_d2h_ms = 0.0;
    size_t unpack_h2d_bytes = 0;
    size_t unpack_d2h_bytes = 0;
    double merge_h2d_ms = 0.0;
    double merge_d2h_ms = 0.0;
    size_t merge_h2d_bytes = 0;
    size_t merge_d2h_bytes = 0;
    double gc_h2d_ms = 0.0;
    double gc_d2h_ms = 0.0;
    size_t gc_h2d_bytes = 0;
    size_t gc_d2h_bytes = 0;
    double planning_h2d_ms = 0.0;
    double planning_d2h_ms = 0.0;
    size_t planning_h2d_bytes = 0;
    size_t planning_d2h_bytes = 0;
    double bloom_h2d_ms = 0.0;
    double bloom_d2h_ms = 0.0;
    size_t bloom_h2d_bytes = 0;
    size_t bloom_d2h_bytes = 0;
    double pack_h2d_ms = 0.0;
    double pack_d2h_ms = 0.0;
    size_t pack_h2d_bytes = 0;
    size_t pack_d2h_bytes = 0;
    double unpack_non_kernel_ms = 0.0;
    double merge_non_kernel_ms = 0.0;
    double bloom_non_kernel_ms = 0.0;
    double pack_non_kernel_ms = 0.0;
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

static std::vector<std::string> write_serialized_output_set(const SerializedSSTHostSet& output,
                                                            const std::string&          dir,
                                                            const std::string&          prefix)
{
    ensure_dir(dir);
    clear_sst_files_in_dir(dir);
    std::vector<std::string> paths;
    for (size_t i = 0; i < output.file_sizes.size(); ++i) {
        char name[256];
        std::snprintf(name, sizeof(name), "%s_%04zu.sst", prefix.c_str(), i);
        std::string path = dir + "/" + name;
        write_binary_file_span(path,
                               output.all_file_bytes.data + output.file_offsets[i],
                               (size_t)output.file_sizes[i]);
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

static std::vector<KVPair> unpack_output_set(const std::vector<std::string>& paths)
{
    std::vector<KVPair> out;
    for (const auto& path : paths) {
        ParsedSST parsed = read_sst_file(path);
        std::vector<KVPair> sst_kv = cpu_unpack_sst(parsed);
        out.insert(out.end(), sst_kv.begin(), sst_kv.end());
    }
    return out;
}

static bool compare_output_sets_logical(const std::vector<std::string>& lhs,
                                        const std::vector<std::string>& rhs)
{
    std::vector<KVPair> lhs_kv = unpack_output_set(lhs);
    std::vector<KVPair> rhs_kv = unpack_output_set(rhs);
    lhs_kv = garbage_collect_sorted_kv(lhs_kv);
    rhs_kv = garbage_collect_sorted_kv(rhs_kv);
    if (lhs_kv.size() != rhs_kv.size()) return false;
    for (size_t i = 0; i < lhs_kv.size(); ++i) {
        if (!kv_equal(lhs_kv[i], rhs_kv[i])) return false;
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

static size_t total_output_bytes(const SerializedSSTHostSet& output)
{
    return output.total_bytes();
}

static size_t total_output_blocks(const SerializedSSTHostSet& output)
{
    return output.total_blocks();
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

static size_t total_input_bytes(const std::vector<ParsedSST>& inputs)
{
    size_t total = 0;
    for (const auto& input : inputs) total += input.file_bytes.size();
    return total;
}

static size_t lower_bound_unpack_h2d_bytes(const std::vector<ParsedSST>& inputs)
{
    size_t total = 0;
    for (const auto& parsed : inputs) {
        std::vector<DataBlockPlanEntry> plans = plans_from_parsed(parsed);
        std::vector<uint32_t> offsets = block_offsets_from_parsed(parsed);
        total += parsed.file_bytes.size();
        total += offsets.size() * sizeof(uint32_t);
        total += plans.size() * sizeof(uint32_t) * 2;
    }
    return total;
}

static double clamp_non_kernel(double stage_ms, float kernel_ms)
{
    double v = stage_ms - (double)kernel_ms;
    return v > 0.0 ? v : 0.0;
}

static double mb_per_sec(size_t bytes, double ms)
{
    if (ms <= 0.0) return 0.0;
    return ((double)bytes / (1024.0 * 1024.0)) / (ms / 1000.0);
}

static bool is_c_with_plan_mode(const std::string& gpu_mode)
{
    return gpu_mode == "c_paper" || gpu_mode == "c_paper_with_plan"
        || gpu_mode == "c_paper_keys_only_with_plan";
}

static bool is_c_without_plan_mode(const std::string& gpu_mode)
{
    return gpu_mode == "c_paper_without_plan"
        || gpu_mode == "c_paper_keys_only_without_plan";
}

static bool is_gc_enabled_mode(const std::string& gpu_mode)
{
    return is_c_with_plan_mode(gpu_mode) || is_c_without_plan_mode(gpu_mode);
}

static bool is_exact_match_mode(const std::string& gpu_mode)
{
    return is_c_with_plan_mode(gpu_mode);
}

static RunSummary run_cpu_once(const std::vector<std::string>& input_paths,
                               const std::string&               output_dir,
                               const std::string&               gpu_mode)
{
    RunSummary summary;
    auto total_start = std::chrono::steady_clock::now();
    double cpu_time_start = get_cpu_time_ms();

    std::vector<ParsedSST> inputs = load_inputs_with_timing(input_paths, summary.read_parse_ms);
    summary.input_bytes = total_input_bytes(inputs);
    CPUCompactionResult cpu;
    double pipeline_cpu_start = get_cpu_time_ms();
    auto   pipeline_wall_start = std::chrono::steady_clock::now();
    (void)gpu_mode;
    cpu = cpu_c_compaction_paper_from_parsed(inputs);
    auto pipeline_wall_end = std::chrono::steady_clock::now();
    summary.pipeline_cpu_time_ms = get_cpu_time_ms() - pipeline_cpu_start;
    summary.pipeline_wall_ms = std::chrono::duration<double, std::milli>(pipeline_wall_end - pipeline_wall_start).count();

    auto write_start = std::chrono::steady_clock::now();
    write_output_set(cpu.output, output_dir, "cpu_compacted");
    auto write_end = std::chrono::steady_clock::now();

    auto total_end = std::chrono::steady_clock::now();
    summary.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    summary.cpu_time_ms = get_cpu_time_ms() - cpu_time_start;
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
    double cpu_time_start = get_cpu_time_ms();

    std::vector<ParsedSST> inputs = load_inputs_with_timing(input_paths, summary.read_parse_ms);
    summary.input_bytes = total_input_bytes(inputs);
    summary.h2d_lower_bound_bytes = lower_bound_unpack_h2d_bytes(inputs);
    GPUCompactionResult gpu;
    double pipeline_cpu_start = get_cpu_time_ms();
    auto   pipeline_wall_start = std::chrono::steady_clock::now();
    if (gpu_mode == "q_plan_on_gpu_slow") {
        gpu = gpu_q_compaction_pipeline_from_parsed(inputs);
    }
    else if (gpu_mode == "q_paper_with_plan") {
        gpu = gpu_q_compaction_paper_from_parsed(inputs, false);
    }
    else if (gpu_mode == "q_paper_without_plan") {
        gpu = gpu_q_compaction_without_plan_from_parsed(inputs, false);
    }
    else if (gpu_mode == "q_paper_with_plan_profile") {
        gpu = gpu_q_compaction_paper_profile_from_parsed(inputs, false);
    }
    else if (gpu_mode == "q_paper_without_plan_profile") {
        gpu = gpu_q_compaction_without_plan_profile_from_parsed(inputs, false);
    }
    else if (gpu_mode == "c_paper" || gpu_mode == "c_paper_with_plan") {
        gpu = gpu_c_compaction_paper_from_parsed(inputs, false);
    }
    else if (gpu_mode == "c_paper_without_plan") {
        gpu = gpu_c_compaction_without_plan_from_parsed(inputs, false);
    }
    else if (gpu_mode == "c_paper_keys_only_with_plan") {
        gpu = gpu_c_compaction_paper_keys_only_from_parsed(inputs, false);
    }
    else if (gpu_mode == "c_paper_keys_only_without_plan") {
        gpu = gpu_c_compaction_without_plan_keys_only_from_parsed(inputs, false);
    }
    else gpu = gpu_q_compaction_from_parsed(inputs);
    auto pipeline_wall_end = std::chrono::steady_clock::now();
    summary.pipeline_cpu_time_ms = get_cpu_time_ms() - pipeline_cpu_start;
    summary.pipeline_wall_ms = std::chrono::duration<double, std::milli>(pipeline_wall_end - pipeline_wall_start).count();

    auto write_start = std::chrono::steady_clock::now();
    if (!gpu.serialized_output.empty()) {
        write_serialized_output_set(gpu.serialized_output, output_dir, "gpu_compacted");
    } else {
        write_output_set(gpu.output, output_dir, "gpu_compacted");
    }
    auto write_end = std::chrono::steady_clock::now();

    auto total_end = std::chrono::steady_clock::now();
    summary.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    summary.cpu_time_ms = get_cpu_time_ms() - cpu_time_start;
    summary.write_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();
    summary.stage = gpu.stage;
    summary.unpack_kernel_ms = gpu.unpack_kernel_ms;
    summary.merge_kernel_ms = gpu.merge_kernel_ms;
    summary.bloom_kernel_ms = gpu.bloom_kernel_ms;
    summary.pack_kernel_ms = gpu.pack_kernel_ms;
    if (!gpu.serialized_output.empty()) {
        summary.output_bytes = total_output_bytes(gpu.serialized_output);
        summary.output_blocks = total_output_blocks(gpu.serialized_output);
        summary.output_files = gpu.serialized_output.file_sizes.size();
    } else {
        summary.output_bytes = total_output_bytes(gpu.output);
        summary.output_blocks = total_output_blocks(gpu.output);
        summary.output_files = gpu.output.files.size();
    }
    summary.unpack_h2d_ms = gpu.unpack_h2d_ms;
    summary.unpack_d2h_ms = gpu.unpack_d2h_ms;
    summary.unpack_h2d_bytes = gpu.unpack_h2d_bytes;
    summary.unpack_d2h_bytes = gpu.unpack_d2h_bytes;
    summary.merge_h2d_ms = gpu.merge_h2d_ms;
    summary.merge_d2h_ms = gpu.merge_d2h_ms;
    summary.merge_h2d_bytes = gpu.merge_h2d_bytes;
    summary.merge_d2h_bytes = gpu.merge_d2h_bytes;
    summary.gc_h2d_ms = gpu.gc_h2d_ms;
    summary.gc_d2h_ms = gpu.gc_d2h_ms;
    summary.gc_h2d_bytes = gpu.gc_h2d_bytes;
    summary.gc_d2h_bytes = gpu.gc_d2h_bytes;
    summary.planning_h2d_ms = gpu.planning_h2d_ms;
    summary.planning_d2h_ms = gpu.planning_d2h_ms;
    summary.planning_h2d_bytes = gpu.planning_h2d_bytes;
    summary.planning_d2h_bytes = gpu.planning_d2h_bytes;
    summary.bloom_h2d_ms = gpu.bloom_h2d_ms;
    summary.bloom_d2h_ms = gpu.bloom_d2h_ms;
    summary.bloom_h2d_bytes = gpu.bloom_h2d_bytes;
    summary.bloom_d2h_bytes = gpu.bloom_d2h_bytes;
    summary.pack_h2d_ms = gpu.pack_h2d_ms;
    summary.pack_d2h_ms = gpu.pack_d2h_ms;
    summary.pack_h2d_bytes = gpu.pack_h2d_bytes;
    summary.pack_d2h_bytes = gpu.pack_d2h_bytes;
    summary.h2d_lower_bound_bytes = summary.unpack_h2d_bytes + summary.merge_h2d_bytes
                                  + summary.gc_h2d_bytes
                                  + summary.planning_h2d_bytes + summary.bloom_h2d_bytes
                                  + summary.pack_h2d_bytes;
    summary.d2h_lower_bound_bytes = summary.unpack_d2h_bytes + summary.merge_d2h_bytes
                                  + summary.gc_d2h_bytes
                                  + summary.planning_d2h_bytes + summary.bloom_d2h_bytes
                                  + summary.pack_d2h_bytes;
    summary.unpack_non_kernel_ms = clamp_non_kernel(summary.stage.unpack_ms, summary.unpack_kernel_ms);
    summary.merge_non_kernel_ms = clamp_non_kernel(summary.stage.merge_ms, summary.merge_kernel_ms);
    summary.bloom_non_kernel_ms = clamp_non_kernel(summary.stage.bloom_ms, summary.bloom_kernel_ms);
    summary.pack_non_kernel_ms = clamp_non_kernel(summary.stage.pack_ms, summary.pack_kernel_ms);
    return summary;
}

int main(int argc, char** argv)
{
    std::string dataset_dir = "dataset";
    std::string out_dir = "results";
    std::string gpu_mode = "baseline";
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
            std::printf("Usage: %s [--dataset DIR] [--out_dir DIR] [--runs N] [--gpu_mode baseline|q_plan_on_gpu_slow|q_paper_with_plan|q_paper_without_plan|c_paper_with_plan|c_paper_without_plan|c_paper_keys_only_with_plan|c_paper_keys_only_without_plan]\n",
                        argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    ensure_dir(out_dir);
    std::vector<std::string> input_paths = collect_sst_paths(dataset_dir, GP_NUM_INPUT_SSTS);

    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaFree(0);

    std::printf("%s benchmark\n", is_gc_enabled_mode(gpu_mode) ? "C-compaction" : "Q-compaction");
    std::printf("  input SSTs: %zu\n", input_paths.size());
    std::printf("  key bytes: %d  value bytes: %d  restart interval: %d  block size: %d\n",
                GP_KEY_BYTES, GP_VALUE_BYTES, GP_RESTART_INTERVAL, GP_DATA_BLOCK_BYTES);
    std::printf("  runs: %d  (timed)\n", runs);
    std::printf("  gpu mode: %s\n", gpu_mode.c_str());
    std::printf("  cpu baseline: enabled (full GC + group-aligned planning)\n");
    std::printf("  gpu garbage collection: %s\n\n",
                is_gc_enabled_mode(gpu_mode)
                    ? "enabled (CPU after GPU sort, before GPU SST construction)"
                    : "disabled (Q-compaction only)");

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
    bool exact_output_match = compare_output_sets(cpu_last, gpu_last);
    bool logical_output_match = compare_output_sets_logical(cpu_last, gpu_last);
    bool outputs_match = is_exact_match_mode(gpu_mode) ? exact_output_match : logical_output_match;

    size_t cpu_best_idx = (size_t)(std::min_element(cpu_totals.begin(), cpu_totals.end()) - cpu_totals.begin());
    size_t gpu_best_idx = (size_t)(std::min_element(gpu_totals.begin(), gpu_totals.end()) - gpu_totals.begin());
    const RunSummary& cpu_best = cpu_runs[cpu_best_idx];
    const RunSummary& gpu_best = gpu_runs[gpu_best_idx];

    std::printf("CPU total (Wall): min=%.2f ms  mean=%.2f +- %.2f ms\n",
                cpu_total_stats.min, cpu_total_stats.mean, cpu_total_stats.stddev);
    std::printf("CPU total (CPU-Time): %.2f ms\n", cpu_best.cpu_time_ms);
    std::printf("CPU pipeline (Wall): %.2f ms  (CPU-Time): %.2f ms  utilization: %.1f%%\n",
                cpu_best.pipeline_wall_ms, cpu_best.pipeline_cpu_time_ms,
                cpu_best.pipeline_wall_ms > 0 ? (cpu_best.pipeline_cpu_time_ms / cpu_best.pipeline_wall_ms) * 100.0 : 0.0);
    std::printf("GPU total (Wall): min=%.2f ms  mean=%.2f +- %.2f ms\n",
                gpu_total_stats.min, gpu_total_stats.mean, gpu_total_stats.stddev);
    std::printf("GPU total (CPU-Time): %.2f ms\n", gpu_best.cpu_time_ms);
    std::printf("GPU pipeline (Wall): %.2f ms  (CPU-Time): %.2f ms  utilization: %.1f%%\n",
                gpu_best.pipeline_wall_ms, gpu_best.pipeline_cpu_time_ms,
                gpu_best.pipeline_wall_ms > 0 ? (gpu_best.pipeline_cpu_time_ms / gpu_best.pipeline_wall_ms) * 100.0 : 0.0);
    std::printf("Speedup: %.2fx (min totals)\n", cpu_total_stats.min / gpu_total_stats.min);
    if (!is_exact_match_mode(gpu_mode)) {
        std::printf("Output logically identical: %s", logical_output_match ? "PASS" : "FAIL");
        if (!exact_output_match && logical_output_match) {
            std::printf("  (different block/file layout than CPU baseline)");
        }
        std::printf("\n\n");
    } else {
        std::printf("Output identical: %s\n\n", outputs_match ? "PASS" : "FAIL");
    }

    std::printf("Best CPU run breakdown (ms):\n");
    std::printf("  read+parse %.2f  unpack %.2f  sort(merge) %.2f  gc %.2f  plan %.2f  bloom %.2f  pack+assemble %.2f  write %.2f\n",
                cpu_best.read_parse_ms, cpu_best.stage.unpack_ms, cpu_best.stage.merge_ms,
                cpu_best.stage.gc_ms, cpu_best.stage.planning_ms, cpu_best.stage.bloom_ms,
                cpu_best.stage.pack_ms, cpu_best.write_ms);
    std::printf("  output bytes %zu  data blocks %zu  output files %zu\n\n",
                cpu_best.output_bytes, cpu_best.output_blocks, cpu_best.output_files);
    std::printf("  I/O profile: input bytes %zu  estimated SSD read BW %.2f MB/s  estimated SSD write BW %.2f MB/s\n\n",
                cpu_best.input_bytes,
                mb_per_sec(cpu_best.input_bytes, cpu_best.read_parse_ms),
                mb_per_sec(cpu_best.output_bytes, cpu_best.write_ms));
    std::printf("  SSD<->CPU (estimated): SSD->CPU %.2f ms (%zu B, %.2f MB/s)  CPU->SSD %.2f ms (%zu B, %.2f MB/s)\n\n",
                cpu_best.read_parse_ms,
                cpu_best.input_bytes,
                mb_per_sec(cpu_best.input_bytes, cpu_best.read_parse_ms),
                cpu_best.write_ms,
                cpu_best.output_bytes,
                mb_per_sec(cpu_best.output_bytes, cpu_best.write_ms));

    std::printf("Best GPU run breakdown (ms):\n");
    std::printf("  read+parse %.2f  unpack %.2f  sort(merge) %.2f  gc %.2f  plan %.2f  bloom %.2f  pack+assemble %.2f  write %.2f\n",
                gpu_best.read_parse_ms, gpu_best.stage.unpack_ms, gpu_best.stage.merge_ms,
                gpu_best.stage.gc_ms, gpu_best.stage.planning_ms, gpu_best.stage.bloom_ms,
                gpu_best.stage.pack_ms, gpu_best.write_ms);
    std::printf("  kernel-only: unpack %.2f  sort(merge) %.2f  bloom %.2f  pack %.2f\n",
                gpu_best.unpack_kernel_ms, gpu_best.merge_kernel_ms, gpu_best.bloom_kernel_ms, gpu_best.pack_kernel_ms);
    std::printf("  non-kernel overhead (ms): unpack %.2f  sort(merge) %.2f  bloom %.2f  pack %.2f\n",
                gpu_best.unpack_non_kernel_ms, gpu_best.merge_non_kernel_ms,
                gpu_best.bloom_non_kernel_ms, gpu_best.pack_non_kernel_ms);
    std::printf("  transfer stage detail (measured):\n");
    std::printf("    unpack  H2D %.2f ms (%zu B)  D2H %.2f ms (%zu B)\n",
                gpu_best.unpack_h2d_ms, gpu_best.unpack_h2d_bytes,
                gpu_best.unpack_d2h_ms, gpu_best.unpack_d2h_bytes);
    std::printf("    merge   H2D %.2f ms (%zu B)  D2H %.2f ms (%zu B)\n",
                gpu_best.merge_h2d_ms, gpu_best.merge_h2d_bytes,
                gpu_best.merge_d2h_ms, gpu_best.merge_d2h_bytes);
    std::printf("    gc      H2D %.2f ms (%zu B)  D2H %.2f ms (%zu B)\n",
                gpu_best.gc_h2d_ms, gpu_best.gc_h2d_bytes,
                gpu_best.gc_d2h_ms, gpu_best.gc_d2h_bytes);
    std::printf("    plan    H2D %.2f ms (%zu B)  D2H %.2f ms (%zu B)\n",
                gpu_best.planning_h2d_ms, gpu_best.planning_h2d_bytes,
                gpu_best.planning_d2h_ms, gpu_best.planning_d2h_bytes);
    std::printf("    bloom   H2D %.2f ms (%zu B)  D2H %.2f ms (%zu B)\n",
                gpu_best.bloom_h2d_ms, gpu_best.bloom_h2d_bytes,
                gpu_best.bloom_d2h_ms, gpu_best.bloom_d2h_bytes);
    std::printf("    pack    H2D %.2f ms (%zu B)  D2H %.2f ms (%zu B)\n",
                gpu_best.pack_h2d_ms, gpu_best.pack_h2d_bytes,
                gpu_best.pack_d2h_ms, gpu_best.pack_d2h_bytes);
    double gpu_h2d_ms = gpu_best.unpack_h2d_ms + gpu_best.merge_h2d_ms + gpu_best.gc_h2d_ms
                      + gpu_best.planning_h2d_ms
                      + gpu_best.bloom_h2d_ms + gpu_best.pack_h2d_ms;
    double gpu_d2h_ms = gpu_best.unpack_d2h_ms + gpu_best.merge_d2h_ms + gpu_best.gc_d2h_ms
                      + gpu_best.planning_d2h_ms
                      + gpu_best.bloom_d2h_ms + gpu_best.pack_d2h_ms;
    std::printf("  measured transfer totals: H2D %.2f ms (%zu B, %.2f MB/s)  D2H %.2f ms (%zu B, %.2f MB/s)\n",
                gpu_h2d_ms, gpu_best.h2d_lower_bound_bytes,
                mb_per_sec(gpu_best.h2d_lower_bound_bytes, gpu_h2d_ms),
                gpu_d2h_ms, gpu_best.d2h_lower_bound_bytes,
                mb_per_sec(gpu_best.d2h_lower_bound_bytes, gpu_d2h_ms));
    double gpu_non_kernel_total = gpu_best.unpack_non_kernel_ms + gpu_best.merge_non_kernel_ms
                                + gpu_best.gc_h2d_ms + gpu_best.gc_d2h_ms
                                + gpu_best.bloom_non_kernel_ms + gpu_best.pack_non_kernel_ms;
    std::printf("  transfer lower-bound bytes: H2D %zu  D2H %zu\n",
                gpu_best.h2d_lower_bound_bytes, gpu_best.d2h_lower_bound_bytes);
    std::printf("  estimated transfer+sync BW (lower-bound): %.2f MB/s\n",
                mb_per_sec(gpu_best.h2d_lower_bound_bytes + gpu_best.d2h_lower_bound_bytes,
                           gpu_non_kernel_total));
    std::printf("  CPU<->GPU (lower-bound): CPU->GPU H2D %zu B  GPU->CPU D2H %zu B\n",
                gpu_best.h2d_lower_bound_bytes, gpu_best.d2h_lower_bound_bytes);
    std::printf("  CPU<->GPU envelope: non-kernel total %.2f ms  estimated BW %.2f MB/s\n",
                gpu_non_kernel_total,
                mb_per_sec(gpu_best.h2d_lower_bound_bytes + gpu_best.d2h_lower_bound_bytes,
                           gpu_non_kernel_total));
    std::printf("  output bytes %zu  data blocks %zu  output files %zu\n",
                gpu_best.output_bytes, gpu_best.output_blocks, gpu_best.output_files);
    std::printf("  I/O profile: input bytes %zu  estimated SSD read BW %.2f MB/s  estimated SSD write BW %.2f MB/s\n",
                gpu_best.input_bytes,
                mb_per_sec(gpu_best.input_bytes, gpu_best.read_parse_ms),
                mb_per_sec(gpu_best.output_bytes, gpu_best.write_ms));
    std::printf("  SSD<->CPU (estimated): SSD->CPU %.2f ms (%zu B, %.2f MB/s)  CPU->SSD %.2f ms (%zu B, %.2f MB/s)\n",
                gpu_best.read_parse_ms,
                gpu_best.input_bytes,
                mb_per_sec(gpu_best.input_bytes, gpu_best.read_parse_ms),
                gpu_best.write_ms,
                gpu_best.output_bytes,
                mb_per_sec(gpu_best.output_bytes, gpu_best.write_ms));

    return outputs_match ? 0 : 1;
}
