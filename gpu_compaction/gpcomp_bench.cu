#include "gpcomp_pipeline.cuh"
#include <nvToolsExt.h>
#include <sys/resource.h>
static double get_cpu_time_ms() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000.0 +
           (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1000.0;
}

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cerrno>
#include <dirent.h>
#include <mutex>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <thread>

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
    float gc_kernel_ms = 0.0f;
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
    double gc_non_kernel_ms = 0.0;
    double bloom_non_kernel_ms = 0.0;
    double pack_non_kernel_ms = 0.0;
    double input_active_ms = 0.0;
    double input_idle_ms = 0.0;
    double output_active_ms = 0.0;
    double output_idle_ms = 0.0;
    double output_d2h_ms = 0.0;
    double output_write_ms = 0.0;
    double stream_overlap_pct = 0.0;
};

struct Stats {
    double min = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
};

struct NvtxRange {
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    explicit NvtxRange(std::string name)
        : name_(std::move(name))
    {
        nvtxRangePushA(name_.c_str());
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }

    std::string name_;
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

static long long current_epoch_ms()
{
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

static void print_iteration_marker(int               run_index,
                                   int               runs,
                                   bool              profile_only,
                                   bool              gpu_only,
                                   const char*       phase,
                                   long long         epoch_ms,
                                   const RunSummary* gpu_summary = nullptr,
                                   const RunSummary* cpu_summary = nullptr)
{
    const bool is_warmup = !profile_only && runs > 1 && run_index == 0;
    const bool is_timed = profile_only || runs <= 1 || run_index > 0;

    std::printf("BENCH_ITERATION_MARKER phase=%s index=%d warmup=%d timed=%d runs=%d profile_only=%d gpu_only=%d epoch_ms=%lld",
                phase,
                run_index,
                is_warmup ? 1 : 0,
                is_timed ? 1 : 0,
                runs,
                profile_only ? 1 : 0,
                gpu_only ? 1 : 0,
                epoch_ms);
    if (gpu_summary != nullptr) {
        std::printf(" gpu_total_ms=%.3f gpu_pipeline_wall_ms=%.3f gpu_pipeline_cpu_ms=%.3f gpu_read_parse_ms=%.3f gpu_write_ms=%.3f gpu_output_bytes=%zu",
                    gpu_summary->total_ms,
                    gpu_summary->pipeline_wall_ms,
                    gpu_summary->pipeline_cpu_time_ms,
                    gpu_summary->read_parse_ms,
                    gpu_summary->write_ms,
                    gpu_summary->output_bytes);
    }
    if (cpu_summary != nullptr) {
        std::printf(" cpu_total_ms=%.3f cpu_pipeline_wall_ms=%.3f cpu_pipeline_cpu_ms=%.3f cpu_read_parse_ms=%.3f cpu_write_ms=%.3f cpu_output_bytes=%zu",
                    cpu_summary->total_ms,
                    cpu_summary->pipeline_wall_ms,
                    cpu_summary->pipeline_cpu_time_ms,
                    cpu_summary->read_parse_ms,
                    cpu_summary->write_ms,
                    cpu_summary->output_bytes);
    }
    std::printf("\n");
    std::fflush(stdout);
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

static std::vector<ParsedSST> load_inputs(const std::vector<std::string>& paths)
{
    std::vector<ParsedSST> parsed;
    parsed.reserve(paths.size());
    for (const auto& path : paths) parsed.push_back(read_sst_file(path));
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

struct StreamedInputFileInfo {
    std::string path;
    ParsedSST   parsed;
    size_t      logical_bytes = 0;
    size_t      physical_bytes = 0;
};

struct InputReadSlot {
    uint8_t* data = nullptr;
    size_t   capacity = 0;
    bool     registered = false;
    size_t   physical_size = 0;
    size_t   logical_size = 0;
    size_t   file_index = 0;
    bool     ready = false;
    bool     in_use = false;

    void reset()
    {
        if (data) {
            if (registered) cudaHostUnregister(data);
            std::free(data);
        }
        data = nullptr;
        capacity = 0;
        registered = false;
        physical_size = 0;
        logical_size = 0;
        file_index = 0;
        ready = false;
        in_use = false;
    }
};

static std::vector<StreamedInputFileInfo>
load_input_metadata_with_timing(const std::vector<std::string>& paths, double& ms)
{
    auto t0 = std::chrono::steady_clock::now();
    std::vector<StreamedInputFileInfo> parsed;
    parsed.reserve(paths.size());
    for (const auto& path : paths) {
        ParsedSSTMetadataOnly meta = read_sst_file_metadata_only(path);
        StreamedInputFileInfo info;
        info.path = path;
        info.parsed = std::move(meta.parsed);
        info.logical_bytes = meta.logical_size;
        info.physical_bytes = meta.physical_size;
        parsed.push_back(std::move(info));
    }
    auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return parsed;
}

static size_t total_input_bytes(const std::vector<StreamedInputFileInfo>& inputs)
{
    size_t total = 0;
    for (const auto& input : inputs) total += input.logical_bytes;
    return total;
}

static size_t lower_bound_unpack_h2d_bytes(const std::vector<StreamedInputFileInfo>& inputs)
{
    size_t total = 0;
    for (const auto& input : inputs) {
        std::vector<DataBlockPlanEntry> plans = plans_from_parsed(input.parsed);
        std::vector<uint32_t> offsets = block_offsets_from_parsed(input.parsed);
        total += input.logical_bytes;
        total += offsets.size() * sizeof(uint32_t);
        total += plans.size() * sizeof(uint32_t) * 2;
    }
    return total;
}

static double clamp_non_kernel(double stage_ms, float kernel_ms);

static void stream_input_unpack_refs(const std::vector<StreamedInputFileInfo>& inputs,
                                     RunSummary&                               summary,
                                     std::vector<GPURefUnpackStreamState>&     unpack_states)
{
    static constexpr int kInputRingDepth = 3;
    InputReadSlot slots[kInputRingDepth];
    std::mutex input_mu;
    std::condition_variable input_cv;
    double input_active_ms = 0.0;

    std::thread reader([&]() {
        for (size_t file_idx = 0; file_idx < inputs.size(); ++file_idx) {
            InputReadSlot& slot = slots[file_idx % (size_t)kInputRingDepth];
            {
                std::unique_lock<std::mutex> lock(input_mu);
                input_cv.wait(lock, [&]() { return !slot.ready && !slot.in_use; });
            }

            ensure_registered_direct_read_buffer(slot.data, slot.capacity, slot.registered,
                                                 std::max(inputs[file_idx].physical_bytes, (size_t)1));
            auto read_start = std::chrono::steady_clock::now();
            read_binary_file_direct_into_registered(inputs[file_idx].path, slot.data, inputs[file_idx].physical_bytes);
            auto read_end = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lock(input_mu);
                input_active_ms += std::chrono::duration<double, std::milli>(read_end - read_start).count();
                slot.physical_size = inputs[file_idx].physical_bytes;
                slot.logical_size = inputs[file_idx].logical_bytes;
                slot.file_index = file_idx;
                slot.ready = true;
            }
            input_cv.notify_all();
        }
    });

    auto unpack_stage_start = std::chrono::steady_clock::now();
    for (size_t file_idx = 0; file_idx < inputs.size(); ++file_idx) {
        if (file_idx >= (size_t)kInputRingDepth) {
            GPURefUnpackStreamState& prior_state = unpack_states[file_idx - (size_t)kInputRingDepth];
            if (prior_state.h2d_stop) cudaEventSynchronize(prior_state.h2d_stop);
            {
                std::lock_guard<std::mutex> lock(input_mu);
                slots[file_idx % (size_t)kInputRingDepth].in_use = false;
            }
            input_cv.notify_all();
        }

        InputReadSlot& slot = slots[file_idx % (size_t)kInputRingDepth];
        {
            std::unique_lock<std::mutex> lock(input_mu);
            input_cv.wait(lock, [&]() { return slot.ready && slot.file_index == file_idx; });
            slot.ready = false;
            slot.in_use = true;
        }

        const StreamedInputFileInfo& input = inputs[file_idx];
        GPURefUnpackStreamState& state = unpack_states[file_idx];
        std::vector<DataBlockPlanEntry> plans = plans_from_parsed(input.parsed);
        state.block_offsets = block_offsets_from_parsed(input.parsed);
        state.first_kv.resize(plans.size());
        state.num_kv.resize(plans.size());
        state.total_kv = input.parsed.footer.total_kv;
        state.num_blocks = (int)plans.size();
        state.source_sst = (uint32_t)file_idx;
        for (size_t p = 0; p < plans.size(); ++p) {
            state.first_kv[p] = plans[p].first_kv;
            state.num_kv[p] = plans[p].num_kv;
        }
        if (state.num_blocks == 0) continue;

        uint32_t max_restarts = 0;
        for (const auto& plan : plans) {
            max_restarts = std::max(max_restarts,
                                    (plan.num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                                  / (uint32_t)GP_RESTART_INTERVAL);
        }
        int block_dim = ((int)max_restarts + 31) / 32 * 32;
        if (block_dim < 32) block_dim = 32;

        cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking);
        cudaEventCreate(&state.h2d_start);
        cudaEventCreate(&state.h2d_stop);
        cudaEventCreate(&state.kernel_start);
        cudaEventCreate(&state.kernel_stop);
        cudaMalloc(&state.d_buf, (size_t)std::max(input.physical_bytes, (size_t)1));
        cudaMalloc(&state.d_offsets, std::max(state.block_offsets.size(), (size_t)1) * sizeof(uint32_t));
        cudaMalloc(&state.d_first_kv, std::max(state.first_kv.size(), (size_t)1) * sizeof(uint32_t));
        cudaMalloc(&state.d_num_kv, std::max(state.num_kv.size(), (size_t)1) * sizeof(uint32_t));
        cudaMalloc(&state.d_out, (size_t)std::max((int)state.total_kv, 1) * sizeof(KVRef));

        cudaEventRecord(state.h2d_start, state.stream);
        cudaMemcpyAsync(state.d_buf, slot.data, input.physical_bytes, cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_offsets, state.block_offsets.data(),
                        state.block_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_first_kv, state.first_kv.data(),
                        state.first_kv.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_num_kv, state.num_kv.data(),
                        state.num_kv.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, state.stream);
        cudaEventRecord(state.h2d_stop, state.stream);

        state.h2d_bytes = input.physical_bytes
                        + state.block_offsets.size() * sizeof(uint32_t)
                        + state.first_kv.size() * sizeof(uint32_t)
                        + state.num_kv.size() * sizeof(uint32_t);

        cudaEventRecord(state.kernel_start, state.stream);
        unpack_ref_kernel<<<state.num_blocks, block_dim, 0, state.stream>>>(
            state.d_buf, state.d_offsets, state.d_first_kv, state.d_num_kv, state.num_blocks,
            (uint32_t)file_idx, state.d_out);
        cudaEventRecord(state.kernel_stop, state.stream);
    }

    for (size_t tail = inputs.size() > (size_t)kInputRingDepth ? inputs.size() - (size_t)kInputRingDepth : 0;
         tail < inputs.size();
         ++tail) {
        if (tail < unpack_states.size() && unpack_states[tail].h2d_stop) {
            cudaEventSynchronize(unpack_states[tail].h2d_stop);
        }
    }

    reader.join();
    for (auto& state : unpack_states) {
        if (!state.stream) continue;
        cudaEventSynchronize(state.kernel_stop);
        cudaEventElapsedTime(&state.h2d_ms, state.h2d_start, state.h2d_stop);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, state.kernel_start, state.kernel_stop);
        summary.unpack_kernel_ms += kernel_ms;
        summary.unpack_h2d_ms += state.h2d_ms;
        summary.unpack_h2d_bytes += state.h2d_bytes;
    }
    auto unpack_stage_end = std::chrono::steady_clock::now();
    summary.stage.unpack_ms = std::chrono::duration<double, std::milli>(
        unpack_stage_end - unpack_stage_start).count();
    summary.input_active_ms = input_active_ms;
    summary.input_idle_ms = std::max(0.0, summary.stage.unpack_ms - summary.input_active_ms);

    for (InputReadSlot& slot : slots) slot.reset();
}

static RunSummary run_gpu_streaming_q_ref_once(const std::vector<std::string>& input_paths,
                                               const std::string&               output_dir)
{
    NvtxRange total_range("gpu_streaming_io_total:q_ref");
    RunSummary summary;
    auto total_start = std::chrono::steady_clock::now();
    double cpu_time_start = get_cpu_time_ms();

    std::vector<StreamedInputFileInfo> inputs;
    {
        NvtxRange load_range("gpu_streaming_io_load_metadata");
        inputs = load_input_metadata_with_timing(input_paths, summary.read_parse_ms);
    }
    summary.input_bytes = total_input_bytes(inputs);
    summary.h2d_lower_bound_bytes = lower_bound_unpack_h2d_bytes(inputs);

    double pipeline_cpu_start = get_cpu_time_ms();
    auto pipeline_wall_start = std::chrono::steady_clock::now();
    {
        NvtxRange pipeline_range("gpu_streaming_io_pipeline:q_ref");
        std::vector<GPURefUnpackStreamState> unpack_states(inputs.size());
        stream_input_unpack_refs(inputs, summary, unpack_states);

        std::vector<KVRef*> d_unpacked;
        std::vector<int> unpack_sizes;
        d_unpacked.reserve(unpack_states.size());
        unpack_sizes.reserve(unpack_states.size());
        for (const auto& state : unpack_states) {
            d_unpacked.push_back(state.d_out);
            unpack_sizes.push_back((int)state.total_kv);
        }

        auto t0 = std::chrono::steady_clock::now();
        DeviceMergeRefTimedResult merged = launch_merge_refs_timed_from_device(d_unpacked, unpack_sizes, false);
        auto t1 = std::chrono::steady_clock::now();
        summary.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        summary.merge_kernel_ms = merged.kernel_ms;
        summary.merge_h2d_ms = merged.h2d_ms;
        summary.merge_d2h_ms = merged.d2h_ms;
        summary.merge_h2d_bytes = merged.h2d_bytes;
        summary.merge_d2h_bytes = merged.d2h_bytes;

        t0 = std::chrono::steady_clock::now();
        RestartGroupSizeTimedResult group_sizes =
            launch_restart_group_sizes_timed_from_device(merged.d_output, merged.total);
        std::vector<DataBlockPlanEntry> plans =
            plan_data_blocks_group_aligned_from_group_sizes(group_sizes.group_sizes, (uint32_t)merged.total);
        t1 = std::chrono::steady_clock::now();
        summary.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        DevicePlanArrays device_plans = upload_plans_to_device(plans);
        summary.planning_h2d_ms = device_plans.h2d_ms;
        summary.planning_h2d_bytes = device_plans.h2d_bytes;
        summary.planning_d2h_ms = std::max(0.0, (double)group_sizes.wall_ms - (double)group_sizes.kernel_ms);
        summary.planning_d2h_bytes = group_sizes.group_sizes.size() * sizeof(uint32_t);

        t0 = std::chrono::steady_clock::now();
        DeviceBloomBatchResult bloom = launch_bloom_filter_batched_to_device_from_plans(
            merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv, plans);
        t1 = std::chrono::steady_clock::now();
        summary.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        summary.bloom_kernel_ms = bloom.kernel_ms;
        summary.bloom_h2d_ms = bloom.h2d_ms;
        summary.bloom_d2h_ms = bloom.d2h_ms;
        summary.bloom_h2d_bytes = bloom.h2d_bytes;
        summary.bloom_d2h_bytes = bloom.d2h_bytes;

        PackResult planned_layout;
        planned_layout.plans = plans;
        planned_layout.block_sizes.resize(plans.size());
        std::vector<uint32_t> predicted_filter_lengths(plans.size());
        for (size_t i = 0; i < plans.size(); ++i) {
            planned_layout.block_sizes[i] = plans[i].serialized_size;
            uint32_t byte_vector_len = plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
            predicted_filter_lengths[i] = (byte_vector_len + 7u) / 8u;
        }
        std::vector<std::pair<size_t, size_t>> pack_spans =
            partition_output_blocks(planned_layout, predicted_filter_lengths, GP_TARGET_FILE_BYTES);

        const uint8_t** d_source_files = upload_source_file_ptrs_to_device(
            unpack_states, &summary.pack_h2d_ms, &summary.pack_h2d_bytes);

        t0 = std::chrono::steady_clock::now();
        DevicePackTimedResult pack = launch_pack_to_device_from_device_plans(
            merged.d_output, d_source_files, device_plans.d_first_kv, device_plans.d_num_kv, plans);
        Key128* d_largest_keys =
            gather_largest_keys_to_device(merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv,
                                          device_plans.num_blocks);

        clear_sst_files_in_dir(output_dir);
        StreamedSSTWriteResult streamed_output = assemble_and_write_sst_files_from_spans_streaming(
            plans, pack.block_sizes, pack.d_blocks, d_largest_keys, bloom.d_filter_bytes,
            bloom.bitvec_offsets, bloom.bitvec_lengths, pack_spans, output_dir, "gpu_compacted");
        t1 = std::chrono::steady_clock::now();

        double pack_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        summary.stage.pack_ms = std::max(0.0, pack_total_ms - streamed_output.write_ms);
        summary.pack_kernel_ms = pack.kernel_ms + streamed_output.kernel_ms;
        summary.pack_h2d_ms += pack.h2d_ms + streamed_output.h2d_ms;
        summary.pack_d2h_ms = pack.d2h_ms + streamed_output.d2h_ms;
        summary.pack_h2d_bytes += pack.h2d_bytes + streamed_output.h2d_bytes;
        summary.pack_d2h_bytes = pack.d2h_bytes + streamed_output.d2h_bytes;
        summary.output_bytes = streamed_output.output_bytes;
        summary.output_blocks = streamed_output.output_blocks;
        summary.output_files = streamed_output.output_files;
        summary.output_active_ms = streamed_output.output_active_ms;
        summary.output_idle_ms = streamed_output.output_idle_ms;
        summary.output_d2h_ms = streamed_output.output_d2h_ms;
        summary.output_write_ms = streamed_output.write_ms;
        summary.stream_overlap_pct = streamed_output.stream_overlap_pct;
        summary.write_ms = streamed_output.write_ms;

        if (d_source_files) cudaFree((void*)d_source_files);
        if (d_largest_keys) cudaFree(d_largest_keys);
        destroy_device_pack_timed_result(pack);
        destroy_device_bloom_batch_result(bloom);
        destroy_device_plan_arrays(device_plans);
        cudaFree(merged.d_output);
        for (auto& state : unpack_states) destroy_unpack_stream_state(state);
    }

    auto pipeline_wall_end = std::chrono::steady_clock::now();
    summary.pipeline_cpu_time_ms = get_cpu_time_ms() - pipeline_cpu_start;
    summary.pipeline_wall_ms = std::chrono::duration<double, std::milli>(
        pipeline_wall_end - pipeline_wall_start).count();
    summary.cpu_time_ms = get_cpu_time_ms() - cpu_time_start;
    auto total_end = std::chrono::steady_clock::now();
    summary.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

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
    summary.gc_non_kernel_ms = clamp_non_kernel(summary.stage.gc_ms, summary.gc_kernel_ms);
    summary.bloom_non_kernel_ms = clamp_non_kernel(summary.stage.bloom_ms, summary.bloom_kernel_ms);
    summary.pack_non_kernel_ms = clamp_non_kernel(summary.stage.pack_ms, summary.pack_kernel_ms);
    return summary;
}

static RunSummary run_gpu_streaming_c_ref_once(const std::vector<std::string>& input_paths,
                                               const std::string&               output_dir)
{
    NvtxRange total_range("gpu_streaming_io_total:c_ref");
    RunSummary summary;
    auto total_start = std::chrono::steady_clock::now();
    double cpu_time_start = get_cpu_time_ms();

    std::vector<StreamedInputFileInfo> inputs;
    {
        NvtxRange load_range("gpu_streaming_io_load_metadata");
        inputs = load_input_metadata_with_timing(input_paths, summary.read_parse_ms);
    }
    summary.input_bytes = total_input_bytes(inputs);
    summary.h2d_lower_bound_bytes = lower_bound_unpack_h2d_bytes(inputs);

    double pipeline_cpu_start = get_cpu_time_ms();
    auto pipeline_wall_start = std::chrono::steady_clock::now();
    {
        NvtxRange pipeline_range("gpu_streaming_io_pipeline:c_ref");
        std::vector<GPURefUnpackStreamState> unpack_states(inputs.size());
        stream_input_unpack_refs(inputs, summary, unpack_states);

        std::vector<KVRef*> d_unpacked;
        std::vector<int> unpack_sizes;
        d_unpacked.reserve(unpack_states.size());
        unpack_sizes.reserve(unpack_states.size());
        for (const auto& state : unpack_states) {
            d_unpacked.push_back(state.d_out);
            unpack_sizes.push_back((int)state.total_kv);
        }

        auto t0 = std::chrono::steady_clock::now();
        DeviceMergeRefTimedResult merged = launch_merge_refs_timed_from_device(d_unpacked, unpack_sizes, false);
        auto t1 = std::chrono::steady_clock::now();
        summary.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        summary.merge_kernel_ms = merged.kernel_ms;
        summary.merge_h2d_ms = merged.h2d_ms;
        summary.merge_d2h_ms = merged.d2h_ms;
        summary.merge_h2d_bytes = merged.h2d_bytes;
        summary.merge_d2h_bytes = merged.d2h_bytes;

        t0 = std::chrono::steady_clock::now();
        PinnedRefArray merged_refs = copy_ref_array_to_pinned_from_device(
            merged.d_output, merged.total, &summary.gc_d2h_ms, &summary.gc_d2h_bytes);
        std::vector<KVRef> survivors = garbage_collect_sorted_refs(merged_refs.data, (size_t)merged_refs.count);
        merged_refs.free();
        KVRef* d_gc_output = upload_ref_array_to_device(
            survivors, &summary.gc_h2d_ms, &summary.gc_h2d_bytes);
        t1 = std::chrono::steady_clock::now();
        summary.stage.gc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cudaFree(merged.d_output);

        t0 = std::chrono::steady_clock::now();
        RestartGroupSizeTimedResult group_sizes =
            launch_restart_group_sizes_timed_from_device(d_gc_output, (int)survivors.size());
        std::vector<DataBlockPlanEntry> plans =
            plan_data_blocks_group_aligned_from_group_sizes(group_sizes.group_sizes, (uint32_t)survivors.size());
        t1 = std::chrono::steady_clock::now();
        summary.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        DevicePlanArrays device_plans = upload_plans_to_device(plans);
        summary.planning_h2d_ms = device_plans.h2d_ms;
        summary.planning_h2d_bytes = device_plans.h2d_bytes;
        summary.planning_d2h_ms = std::max(0.0, (double)group_sizes.wall_ms - (double)group_sizes.kernel_ms);
        summary.planning_d2h_bytes = group_sizes.group_sizes.size() * sizeof(uint32_t);

        t0 = std::chrono::steady_clock::now();
        DeviceBloomBatchResult bloom = launch_bloom_filter_batched_to_device_from_plans(
            d_gc_output, device_plans.d_first_kv, device_plans.d_num_kv, plans);
        t1 = std::chrono::steady_clock::now();
        summary.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        summary.bloom_kernel_ms = bloom.kernel_ms;
        summary.bloom_h2d_ms = bloom.h2d_ms;
        summary.bloom_d2h_ms = bloom.d2h_ms;
        summary.bloom_h2d_bytes = bloom.h2d_bytes;
        summary.bloom_d2h_bytes = bloom.d2h_bytes;

        PackResult planned_layout;
        planned_layout.plans = plans;
        planned_layout.block_sizes.resize(plans.size());
        std::vector<uint32_t> predicted_filter_lengths(plans.size());
        for (size_t i = 0; i < plans.size(); ++i) {
            planned_layout.block_sizes[i] = plans[i].serialized_size;
            uint32_t byte_vector_len = plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
            predicted_filter_lengths[i] = (byte_vector_len + 7u) / 8u;
        }
        std::vector<std::pair<size_t, size_t>> pack_spans =
            partition_output_blocks(planned_layout, predicted_filter_lengths, GP_TARGET_FILE_BYTES);

        const uint8_t** d_source_files = upload_source_file_ptrs_to_device(
            unpack_states, &summary.pack_h2d_ms, &summary.pack_h2d_bytes);

        t0 = std::chrono::steady_clock::now();
        DevicePackTimedResult pack = launch_pack_to_device_from_device_plans(
            d_gc_output, d_source_files, device_plans.d_first_kv, device_plans.d_num_kv, plans);
        Key128* d_largest_keys =
            gather_largest_keys_to_device(d_gc_output, device_plans.d_first_kv, device_plans.d_num_kv,
                                          device_plans.num_blocks);

        clear_sst_files_in_dir(output_dir);
        StreamedSSTWriteResult streamed_output = assemble_and_write_sst_files_from_spans_streaming(
            plans, pack.block_sizes, pack.d_blocks, d_largest_keys, bloom.d_filter_bytes,
            bloom.bitvec_offsets, bloom.bitvec_lengths, pack_spans, output_dir, "gpu_compacted");
        t1 = std::chrono::steady_clock::now();

        double pack_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        summary.stage.pack_ms = std::max(0.0, pack_total_ms - streamed_output.write_ms);
        summary.pack_kernel_ms = pack.kernel_ms + streamed_output.kernel_ms;
        summary.pack_h2d_ms += pack.h2d_ms + streamed_output.h2d_ms;
        summary.pack_d2h_ms = pack.d2h_ms + streamed_output.d2h_ms;
        summary.pack_h2d_bytes += pack.h2d_bytes + streamed_output.h2d_bytes;
        summary.pack_d2h_bytes = pack.d2h_bytes + streamed_output.d2h_bytes;
        summary.output_bytes = streamed_output.output_bytes;
        summary.output_blocks = streamed_output.output_blocks;
        summary.output_files = streamed_output.output_files;
        summary.output_active_ms = streamed_output.output_active_ms;
        summary.output_idle_ms = streamed_output.output_idle_ms;
        summary.output_d2h_ms = streamed_output.output_d2h_ms;
        summary.output_write_ms = streamed_output.write_ms;
        summary.stream_overlap_pct = streamed_output.stream_overlap_pct;
        summary.write_ms = streamed_output.write_ms;

        if (d_source_files) cudaFree((void*)d_source_files);
        if (d_largest_keys) cudaFree(d_largest_keys);
        destroy_device_pack_timed_result(pack);
        destroy_device_bloom_batch_result(bloom);
        destroy_device_plan_arrays(device_plans);
        if (d_gc_output) cudaFree(d_gc_output);
        for (auto& state : unpack_states) destroy_unpack_stream_state(state);
    }

    auto pipeline_wall_end = std::chrono::steady_clock::now();
    summary.pipeline_cpu_time_ms = get_cpu_time_ms() - pipeline_cpu_start;
    summary.pipeline_wall_ms = std::chrono::duration<double, std::milli>(
        pipeline_wall_end - pipeline_wall_start).count();
    summary.cpu_time_ms = get_cpu_time_ms() - cpu_time_start;
    auto total_end = std::chrono::steady_clock::now();
    summary.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

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
    summary.gc_non_kernel_ms = clamp_non_kernel(summary.stage.gc_ms, summary.gc_kernel_ms);
    summary.bloom_non_kernel_ms = clamp_non_kernel(summary.stage.bloom_ms, summary.bloom_kernel_ms);
    summary.pack_non_kernel_ms = clamp_non_kernel(summary.stage.pack_ms, summary.pack_kernel_ms);
    return summary;
}

static RunSummary run_gpu_streaming_once(const std::vector<std::string>& input_paths,
                                         const std::string&               output_dir,
                                         const std::string&               gpu_mode)
{
    if (gpu_mode == "q_paper_with_plan_streaming_io") {
        return run_gpu_streaming_q_ref_once(input_paths, output_dir);
    }
    if (gpu_mode == "c_paper_with_plan_streaming_io") {
        return run_gpu_streaming_c_ref_once(input_paths, output_dir);
    }
    throw std::runtime_error("unsupported streaming gpu mode: " + gpu_mode);
}

static void run_gpu_streaming_q_ref_profile_once(const std::vector<std::string>& input_paths,
                                                 const std::string&               output_dir)
{
    NvtxRange total_range("gpu_streaming_io_profile_total:q_ref");

    std::vector<StreamedInputFileInfo> inputs;
    {
        NvtxRange load_range("gpu_streaming_io_profile_load_metadata:q_ref");
        double unused_read_parse_ms = 0.0;
        inputs = load_input_metadata_with_timing(input_paths, unused_read_parse_ms);
    }

    {
        NvtxRange pipeline_range("gpu_streaming_io_profile_pipeline:q_ref");
        RunSummary unused_summary;
        std::vector<GPURefUnpackStreamState> unpack_states(inputs.size());
        stream_input_unpack_refs(inputs, unused_summary, unpack_states);

        std::vector<KVRef*> d_unpacked;
        std::vector<int> unpack_sizes;
        d_unpacked.reserve(unpack_states.size());
        unpack_sizes.reserve(unpack_states.size());
        for (const auto& state : unpack_states) {
            d_unpacked.push_back(state.d_out);
            unpack_sizes.push_back((int)state.total_kv);
        }

        DeviceMergeRefUntimedResult merged =
            launch_merge_refs_untimed_from_device(d_unpacked, unpack_sizes);
        std::vector<uint32_t> group_sizes =
            launch_restart_group_sizes_untimed_from_device(merged.d_output, merged.total);
        std::vector<DataBlockPlanEntry> plans =
            plan_data_blocks_group_aligned_from_group_sizes(group_sizes, (uint32_t)merged.total);
        DevicePlanArraysUntimed device_plans = upload_plans_to_device_untimed(plans);
        DeviceBloomBatchUntimedResult bloom = launch_bloom_filter_batched_untimed_from_plans(
            merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv, plans);

        PackResult planned_layout;
        planned_layout.plans = plans;
        planned_layout.block_sizes.resize(plans.size());
        for (size_t i = 0; i < plans.size(); ++i) {
            planned_layout.block_sizes[i] = plans[i].serialized_size;
        }
        std::vector<uint32_t> predicted_filter_lengths = predict_filter_lengths_from_plans(plans);
        std::vector<std::pair<size_t, size_t>> pack_spans =
            partition_output_blocks(planned_layout, predicted_filter_lengths, GP_TARGET_FILE_BYTES);

        const uint8_t** d_source_files = upload_source_file_ptrs_to_device(unpack_states);
        DevicePackUntimedResult pack = launch_pack_untimed_from_device_plans(
            merged.d_output, d_source_files, device_plans.d_first_kv, device_plans.d_num_kv, plans);
        Key128* d_largest_keys = gather_largest_keys_to_device(
            merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv, device_plans.num_blocks);
        clear_sst_files_in_dir(output_dir);
        (void)assemble_and_write_sst_files_from_spans_streaming(
            plans, pack.block_sizes, pack.d_blocks, d_largest_keys, bloom.d_filter_bytes,
            bloom.bitvec_offsets, bloom.bitvec_lengths, pack_spans, output_dir, "gpu_compacted");

        if (d_source_files) cudaFree((void*)d_source_files);
        if (d_largest_keys) cudaFree(d_largest_keys);
        destroy_device_pack_untimed_result(pack);
        destroy_device_bloom_batch_untimed_result(bloom);
        destroy_device_plan_arrays_untimed(device_plans);
        if (merged.d_output) cudaFree(merged.d_output);
        for (auto& state : unpack_states) destroy_unpack_stream_state(state);
    }
}

static void run_gpu_streaming_c_ref_profile_once(const std::vector<std::string>& input_paths,
                                                 const std::string&               output_dir)
{
    NvtxRange total_range("gpu_streaming_io_profile_total:c_ref");

    std::vector<StreamedInputFileInfo> inputs;
    {
        NvtxRange load_range("gpu_streaming_io_profile_load_metadata:c_ref");
        double unused_read_parse_ms = 0.0;
        inputs = load_input_metadata_with_timing(input_paths, unused_read_parse_ms);
    }

    {
        NvtxRange pipeline_range("gpu_streaming_io_profile_pipeline:c_ref");
        RunSummary unused_summary;
        std::vector<GPURefUnpackStreamState> unpack_states(inputs.size());
        stream_input_unpack_refs(inputs, unused_summary, unpack_states);

        std::vector<KVRef*> d_unpacked;
        std::vector<int> unpack_sizes;
        d_unpacked.reserve(unpack_states.size());
        unpack_sizes.reserve(unpack_states.size());
        for (const auto& state : unpack_states) {
            d_unpacked.push_back(state.d_out);
            unpack_sizes.push_back((int)state.total_kv);
        }

        DeviceMergeRefUntimedResult merged =
            launch_merge_refs_untimed_from_device(d_unpacked, unpack_sizes);
        PinnedRefArray pinned = copy_ref_array_to_pinned_from_device(merged.d_output, merged.total);
        std::vector<KVRef> survivors = garbage_collect_sorted_refs(
            pinned.data, (size_t)pinned.count);
        pinned.free();

        KVRef* d_gc_output = upload_ref_array_to_device(survivors);
        if (merged.d_output) cudaFree(merged.d_output);

        std::vector<uint32_t> group_sizes = launch_restart_group_sizes_untimed_from_device(
            d_gc_output, (int)survivors.size());
        std::vector<DataBlockPlanEntry> plans = plan_data_blocks_group_aligned_from_group_sizes(
            group_sizes, (uint32_t)survivors.size());
        DevicePlanArraysUntimed device_plans = upload_plans_to_device_untimed(plans);
        DeviceBloomBatchUntimedResult bloom = launch_bloom_filter_batched_untimed_from_plans(
            d_gc_output, device_plans.d_first_kv, device_plans.d_num_kv, plans);

        PackResult planned_layout;
        planned_layout.plans = plans;
        planned_layout.block_sizes.resize(plans.size());
        for (size_t i = 0; i < plans.size(); ++i) {
            planned_layout.block_sizes[i] = plans[i].serialized_size;
        }
        std::vector<uint32_t> predicted_filter_lengths = predict_filter_lengths_from_plans(plans);
        std::vector<std::pair<size_t, size_t>> pack_spans =
            partition_output_blocks(planned_layout, predicted_filter_lengths, GP_TARGET_FILE_BYTES);

        const uint8_t** d_source_files = upload_source_file_ptrs_to_device(unpack_states);
        DevicePackUntimedResult pack = launch_pack_untimed_from_device_plans(
            d_gc_output, d_source_files, device_plans.d_first_kv, device_plans.d_num_kv, plans);
        Key128* d_largest_keys = gather_largest_keys_to_device(
            d_gc_output, device_plans.d_first_kv, device_plans.d_num_kv, device_plans.num_blocks);
        clear_sst_files_in_dir(output_dir);
        (void)assemble_and_write_sst_files_from_spans_streaming(
            plans, pack.block_sizes, pack.d_blocks, d_largest_keys, bloom.d_filter_bytes,
            bloom.bitvec_offsets, bloom.bitvec_lengths, pack_spans, output_dir, "gpu_compacted");

        if (d_source_files) cudaFree((void*)d_source_files);
        if (d_largest_keys) cudaFree(d_largest_keys);
        destroy_device_pack_untimed_result(pack);
        destroy_device_bloom_batch_untimed_result(bloom);
        destroy_device_plan_arrays_untimed(device_plans);
        if (d_gc_output) cudaFree(d_gc_output);
        for (auto& state : unpack_states) destroy_unpack_stream_state(state);
    }
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
        || gpu_mode == "c_paper_with_plan_streaming_io";
}

static bool is_c_without_plan_mode(const std::string& gpu_mode)
{
    return gpu_mode == "c_paper_without_plan";
}

static bool is_gc_enabled_mode(const std::string& gpu_mode)
{
    return is_c_with_plan_mode(gpu_mode) || is_c_without_plan_mode(gpu_mode);
}

static bool is_exact_match_mode(const std::string& gpu_mode)
{
    return gpu_mode == "c_paper"
        || gpu_mode == "c_paper_with_plan";
}

static bool is_streaming_io_mode(const std::string& gpu_mode)
{
    return gpu_mode == "q_paper_with_plan_streaming_io"
        || gpu_mode == "c_paper_with_plan_streaming_io";
}

static bool gpu_mode_supports_profile_only(const std::string& gpu_mode)
{
    return gpu_mode == "q_paper_with_plan"
        || gpu_mode == "q_paper_with_plan_streaming_io"
        || gpu_mode == "q_paper_without_plan"
        || gpu_mode == "c_paper"
        || gpu_mode == "c_paper_with_plan"
        || gpu_mode == "c_paper_with_plan_streaming_io"
        || gpu_mode == "c_paper_without_plan";
}

static RunSummary run_cpu_once(const std::vector<std::string>& input_paths,
                               const std::string&               output_dir,
                               const std::string&               gpu_mode)
{
    NvtxRange total_range("cpu_baseline_total");
    RunSummary summary;
    auto total_start = std::chrono::steady_clock::now();
    double cpu_time_start = get_cpu_time_ms();

    std::vector<ParsedSST> inputs;
    {
        NvtxRange load_range("cpu_baseline_load_inputs");
        inputs = load_inputs_with_timing(input_paths, summary.read_parse_ms);
    }
    summary.input_bytes = total_input_bytes(inputs);
    CPUCompactionResult cpu;
    double pipeline_cpu_start = get_cpu_time_ms();
    auto   pipeline_wall_start = std::chrono::steady_clock::now();
    (void)gpu_mode;
    {
        NvtxRange pipeline_range("cpu_baseline_pipeline");
        cpu = cpu_c_compaction_paper_from_parsed(inputs);
    }
    auto pipeline_wall_end = std::chrono::steady_clock::now();
    summary.pipeline_cpu_time_ms = get_cpu_time_ms() - pipeline_cpu_start;
    summary.pipeline_wall_ms = std::chrono::duration<double, std::milli>(pipeline_wall_end - pipeline_wall_start).count();

    auto write_start = std::chrono::steady_clock::now();
    {
        NvtxRange write_range("cpu_baseline_write_output");
        write_output_set(cpu.output, output_dir, "cpu_compacted");
    }
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
    if (is_streaming_io_mode(gpu_mode)) {
        return run_gpu_streaming_once(input_paths, output_dir, gpu_mode);
    }

    NvtxRange total_range(std::string("gpu_total:") + gpu_mode);
    RunSummary summary;
    auto total_start = std::chrono::steady_clock::now();
    double cpu_time_start = get_cpu_time_ms();

    std::vector<ParsedSST> inputs;
    {
        NvtxRange load_range(std::string("gpu_load_inputs:") + gpu_mode);
        inputs = load_inputs_with_timing(input_paths, summary.read_parse_ms);
    }
    summary.input_bytes = total_input_bytes(inputs);
    summary.h2d_lower_bound_bytes = lower_bound_unpack_h2d_bytes(inputs);
    GPUCompactionResult gpu;
    double pipeline_cpu_start = get_cpu_time_ms();
    auto   pipeline_wall_start = std::chrono::steady_clock::now();
    {
        NvtxRange pipeline_range(std::string("gpu_pipeline:") + gpu_mode);
        if (gpu_mode == "q_paper_with_plan") {
            gpu = gpu_q_compaction_paper_from_parsed(inputs, false);
        }
        else if (gpu_mode == "q_paper_without_plan") {
            gpu = gpu_q_compaction_without_plan_from_parsed(inputs, false);
        }
        else if (gpu_mode == "q_paper_with_plan_profile") {
            gpu = gpu_q_compaction_paper_profile_from_parsed(inputs, false);
        }
        else if (gpu_mode == "c_paper" || gpu_mode == "c_paper_with_plan") {
            gpu = gpu_c_compaction_paper_from_parsed(inputs, false);
        }
        else if (gpu_mode == "c_paper_without_plan") {
            gpu = gpu_c_compaction_without_plan_from_parsed(inputs, false);
        }
        else gpu = gpu_q_compaction_from_parsed(inputs);
    }
    auto pipeline_wall_end = std::chrono::steady_clock::now();
    summary.pipeline_cpu_time_ms = get_cpu_time_ms() - pipeline_cpu_start;
    summary.pipeline_wall_ms = std::chrono::duration<double, std::milli>(pipeline_wall_end - pipeline_wall_start).count();

    auto write_start = std::chrono::steady_clock::now();
    {
        NvtxRange write_range(std::string("gpu_write_output:") + gpu_mode);
        if (!gpu.serialized_output.empty()) {
            write_serialized_output_set(gpu.serialized_output, output_dir, "gpu_compacted");
        } else {
            write_output_set(gpu.output, output_dir, "gpu_compacted");
        }
    }
    auto write_end = std::chrono::steady_clock::now();

    auto total_end = std::chrono::steady_clock::now();
    summary.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    summary.cpu_time_ms = get_cpu_time_ms() - cpu_time_start;
    summary.write_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();
    summary.stage = gpu.stage;
    summary.unpack_kernel_ms = gpu.unpack_kernel_ms;
    summary.merge_kernel_ms = gpu.merge_kernel_ms;
    summary.gc_kernel_ms = gpu.gc_kernel_ms;
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
    summary.gc_non_kernel_ms = clamp_non_kernel(summary.stage.gc_ms, summary.gc_kernel_ms);
    summary.bloom_non_kernel_ms = clamp_non_kernel(summary.stage.bloom_ms, summary.bloom_kernel_ms);
    summary.pack_non_kernel_ms = clamp_non_kernel(summary.stage.pack_ms, summary.pack_kernel_ms);
    return summary;
}

static void run_gpu_profile_once(const std::vector<std::string>& input_paths,
                                 const std::string&               output_dir,
                                 const std::string&               gpu_mode)
{
    NvtxRange total_range(std::string("gpu_profile_total:") + gpu_mode);

    if (gpu_mode == "q_paper_with_plan_streaming_io") {
        NvtxRange pipeline_range(std::string("gpu_profile_pipeline:") + gpu_mode);
        run_gpu_streaming_q_ref_profile_once(input_paths, output_dir);
        return;
    }
    if (gpu_mode == "c_paper_with_plan_streaming_io") {
        NvtxRange pipeline_range(std::string("gpu_profile_pipeline:") + gpu_mode);
        run_gpu_streaming_c_ref_profile_once(input_paths, output_dir);
        return;
    }

    GPUCompactionResult gpu;
    {
        NvtxRange pipeline_range(std::string("gpu_profile_pipeline:") + gpu_mode);
        std::vector<ParsedSST> inputs;
        {
            NvtxRange load_range(std::string("gpu_profile_load_inputs:") + gpu_mode);
            inputs = load_inputs(input_paths);
        }
        if (gpu_mode == "q_paper_with_plan") {
            gpu = gpu_q_compaction_paper_profile_from_parsed(inputs, false);
        }
        else if (gpu_mode == "q_paper_without_plan") {
            gpu = gpu_q_compaction_without_plan_from_parsed(inputs, false);
        }
        else if (gpu_mode == "c_paper" || gpu_mode == "c_paper_with_plan") {
            gpu = gpu_c_compaction_paper_profile_from_parsed(inputs, false);
        }
        else if (gpu_mode == "c_paper_without_plan") {
            gpu = gpu_c_compaction_without_plan_from_parsed(inputs, false);
        }
        else {
            throw std::runtime_error("profile-only mode does not support gpu_mode: " + gpu_mode);
        }
    }

    {
        NvtxRange write_range(std::string("gpu_profile_write_output:") + gpu_mode);
        if (!gpu.serialized_output.empty()) {
            write_serialized_output_set(gpu.serialized_output, output_dir, "gpu_compacted");
        } else {
            write_output_set(gpu.output, output_dir, "gpu_compacted");
        }
    }
}

int main(int argc, char** argv)
{
    std::string dataset_dir = "dataset";
    std::string out_dir = "results";
    std::string gpu_mode = "baseline";
    int runs = 3;
    bool gpu_only = false;
    bool profile_only = false;
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
        else if (arg == "--gpu_only") gpu_only = true;
        else if (arg == "--profile_only") profile_only = true;
        else if (arg == "--help") {
            std::printf("Usage: %s [--dataset DIR] [--out_dir DIR] [--runs N] [--gpu_mode baseline|q_paper_with_plan|q_paper_with_plan_streaming_io|q_paper_without_plan|c_paper_with_plan|c_paper_with_plan_streaming_io|c_paper_without_plan] [--gpu_only] [--profile_only]\n",
                        argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (profile_only) {
        gpu_only = true;
        if (!gpu_mode_supports_profile_only(gpu_mode)) {
            throw std::runtime_error("profile-only mode only supports q/c paper modes");
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
    if (profile_only)
        std::printf("  runs: %d  (profile-only, all runs profiled)\n", runs);
    else if (runs > 1)
        std::printf("  runs: %d  (1 warmup + %d timed)\n", runs, runs - 1);
    else
        std::printf("  runs: %d  (timed, no warmup)\n", runs);
    std::printf("  gpu mode: %s\n", gpu_mode.c_str());
    std::printf("  profile-only mode: %s\n", profile_only ? "enabled" : "disabled");
    std::printf("  storage IO: %s\n", gpcomp_direct_io_enabled() ? "direct (O_DIRECT + fdatasync)" : "buffered");
    if (profile_only) {
        std::printf("  cpu baseline: disabled (--profile_only)\n");
    } else if (gpu_only) {
        std::printf("  cpu baseline: disabled (--gpu_only)\n");
    } else {
        std::printf("  cpu baseline: enabled (full GC + group-aligned planning)\n");
    }
    const char* gc_desc = is_gc_enabled_mode(gpu_mode) ? "cpu" : "none";
    std::printf("  garbage collection type: %s\n\n", gc_desc);

    std::vector<RunSummary> cpu_runs, gpu_runs;
    if (!gpu_only) {
        cpu_runs.reserve(runs);
    }
    gpu_runs.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        NvtxRange iteration_range(std::string("benchmark_iteration:") + std::to_string(r));
        print_iteration_marker(r, runs, profile_only, gpu_only, "start", current_epoch_ms());
        char gpu_dir[256];
        std::snprintf(gpu_dir, sizeof(gpu_dir), "%s/gpu_run%d", out_dir.c_str(), r);
        ensure_dir(gpu_dir);
        if (profile_only) {
            run_gpu_profile_once(input_paths, gpu_dir, gpu_mode);
            print_iteration_marker(r, runs, profile_only, gpu_only, "end", current_epoch_ms());
            continue;
        }
        RunSummary cpu_run;
        const RunSummary* cpu_summary = nullptr;
        if (!gpu_only) {
            char cpu_dir[256];
            std::snprintf(cpu_dir, sizeof(cpu_dir), "%s/cpu_run%d", out_dir.c_str(), r);
            ensure_dir(cpu_dir);
            cpu_run = run_cpu_once(input_paths, cpu_dir, gpu_mode);
            cpu_runs.push_back(cpu_run);
            cpu_summary = &cpu_run;
        }
        RunSummary gpu_run = run_gpu_once(input_paths, gpu_dir, gpu_mode);
        gpu_runs.push_back(gpu_run);
        print_iteration_marker(r, runs, profile_only, gpu_only, "end", current_epoch_ms(), &gpu_run, cpu_summary);
    }

    if (profile_only) {
        std::printf("Profile-only GPU compaction complete.\n");
        return 0;
    }

    int timed_start = (runs > 1) ? 1 : 0;
    std::vector<double> cpu_totals, gpu_totals, cpu_reads, gpu_reads;
    for (int r = timed_start; r < runs; ++r) {
        gpu_totals.push_back(gpu_runs[r].total_ms);
        gpu_reads.push_back(gpu_runs[r].read_parse_ms);
        if (!gpu_only) {
            cpu_totals.push_back(cpu_runs[r].total_ms);
            cpu_reads.push_back(cpu_runs[r].read_parse_ms);
        }
    }

    Stats gpu_total_stats = compute_stats(gpu_totals);
    Stats gpu_read_stats = compute_stats(gpu_reads);
    size_t gpu_best_idx = timed_start + (size_t)(std::min_element(gpu_totals.begin(), gpu_totals.end()) - gpu_totals.begin());
    const RunSummary& gpu_best = gpu_runs[gpu_best_idx];
    std::printf("GPU total (Wall): min=%.2f ms  mean=%.2f +- %.2f ms\n",
                gpu_total_stats.min, gpu_total_stats.mean, gpu_total_stats.stddev);
    std::printf("GPU total (CPU-Time): %.2f ms\n", gpu_best.cpu_time_ms);
    std::printf("GPU pipeline (Wall): %.2f ms  (CPU-Time): %.2f ms  utilization: %.1f%%\n",
                gpu_best.pipeline_wall_ms, gpu_best.pipeline_cpu_time_ms,
                gpu_best.pipeline_wall_ms > 0 ? (gpu_best.pipeline_cpu_time_ms / gpu_best.pipeline_wall_ms) * 100.0 : 0.0);

    bool outputs_match = true;
    if (!gpu_only) {
        Stats cpu_total_stats = compute_stats(cpu_totals);
        Stats cpu_read_stats = compute_stats(cpu_reads);
        std::vector<std::string> cpu_last, gpu_last;
        cpu_last = collect_sst_paths(out_dir + "/cpu_run" + std::to_string(runs - 1));
        gpu_last = collect_sst_paths(out_dir + "/gpu_run" + std::to_string(runs - 1));
        bool exact_output_match = compare_output_sets(cpu_last, gpu_last);
        bool logical_output_match = compare_output_sets_logical(cpu_last, gpu_last);
        outputs_match = is_exact_match_mode(gpu_mode) ? exact_output_match : logical_output_match;

        size_t cpu_best_idx = timed_start + (size_t)(std::min_element(cpu_totals.begin(), cpu_totals.end()) - cpu_totals.begin());
        const RunSummary& cpu_best = cpu_runs[cpu_best_idx];

        std::printf("CPU total (Wall): min=%.2f ms  mean=%.2f +- %.2f ms\n",
                    cpu_total_stats.min, cpu_total_stats.mean, cpu_total_stats.stddev);
        std::printf("CPU total (CPU-Time): %.2f ms\n", cpu_best.cpu_time_ms);
        std::printf("CPU pipeline (Wall): %.2f ms  (CPU-Time): %.2f ms  utilization: %.1f%%\n",
                    cpu_best.pipeline_wall_ms, cpu_best.pipeline_cpu_time_ms,
                    cpu_best.pipeline_wall_ms > 0 ? (cpu_best.pipeline_cpu_time_ms / cpu_best.pipeline_wall_ms) * 100.0 : 0.0);
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
    } else {
        std::printf("CPU baseline: skipped (gpu_only mode)\n\n");
    }

    std::printf("Best GPU run breakdown (ms):\n");
    std::printf("  read+parse %.2f  unpack %.2f  sort(merge) %.2f  gc %.2f  plan %.2f  bloom %.2f  pack+assemble %.2f  write %.2f\n",
                gpu_best.read_parse_ms, gpu_best.stage.unpack_ms, gpu_best.stage.merge_ms,
                gpu_best.stage.gc_ms, gpu_best.stage.planning_ms, gpu_best.stage.bloom_ms,
                gpu_best.stage.pack_ms, gpu_best.write_ms);
    std::printf("  kernel-only: unpack %.2f  sort(merge) %.2f  gc %.2f  bloom %.2f  pack %.2f\n",
                gpu_best.unpack_kernel_ms, gpu_best.merge_kernel_ms, gpu_best.gc_kernel_ms,
                gpu_best.bloom_kernel_ms, gpu_best.pack_kernel_ms);
    std::printf("  non-kernel overhead (ms): unpack %.2f  sort(merge) %.2f  gc %.2f  bloom %.2f  pack %.2f\n",
                gpu_best.unpack_non_kernel_ms, gpu_best.merge_non_kernel_ms,
                gpu_best.gc_non_kernel_ms, gpu_best.bloom_non_kernel_ms, gpu_best.pack_non_kernel_ms);
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
    double gpu_input_io_ms = gpu_best.input_active_ms > 0.0 ? gpu_best.input_active_ms : gpu_best.read_parse_ms;
    double gpu_output_io_ms = gpu_best.output_write_ms > 0.0 ? gpu_best.output_write_ms : gpu_best.write_ms;
    std::printf("  output bytes %zu  data blocks %zu  output files %zu\n",
                gpu_best.output_bytes, gpu_best.output_blocks, gpu_best.output_files);
    std::printf("  I/O profile: input bytes %zu  estimated SSD read BW %.2f MB/s  estimated SSD write BW %.2f MB/s\n",
                gpu_best.input_bytes,
                mb_per_sec(gpu_best.input_bytes, gpu_input_io_ms),
                mb_per_sec(gpu_best.output_bytes, gpu_output_io_ms));
    std::printf("  SSD<->CPU (estimated): SSD->CPU %.2f ms (%zu B, %.2f MB/s)  CPU->SSD %.2f ms (%zu B, %.2f MB/s)\n",
                gpu_input_io_ms,
                gpu_best.input_bytes,
                mb_per_sec(gpu_best.input_bytes, gpu_input_io_ms),
                gpu_output_io_ms,
                gpu_best.output_bytes,
                mb_per_sec(gpu_best.output_bytes, gpu_output_io_ms));
    if (gpu_best.input_active_ms > 0.0 || gpu_best.output_active_ms > 0.0) {
        std::printf("  streaming I/O: input active %.2f ms  input idle %.2f ms  output active %.2f ms  output idle %.2f ms\n",
                    gpu_best.input_active_ms,
                    gpu_best.input_idle_ms,
                    gpu_best.output_active_ms,
                    gpu_best.output_idle_ms);
        std::printf("  streamed output detail: D2H %.2f ms  write %.2f ms  overlap %.1f%%\n",
                    gpu_best.output_d2h_ms,
                    gpu_best.output_write_ms,
                    gpu_best.stream_overlap_pct);
    }

    return outputs_match ? 0 : 1;
}
