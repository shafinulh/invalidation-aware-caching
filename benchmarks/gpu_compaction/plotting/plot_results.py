import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASE_DIR = os.environ.get(
    "GPCOMP_RESULTS_DIR",
    os.environ.get("GPU_COMPACTION_OUT_ROOT", "sweep_results"),
)
DEFAULT_GRAPH_DIR = os.environ.get("GPCOMP_GRAPHS_DIR", "graphs")

GPU_MODE_ORDER = [
    "q_paper_with_plan",
    "q_paper_with_plan_streaming_io",
    "q_paper_without_plan",
    "c_paper_with_plan",
    "c_paper_with_plan_streaming_io",
    "c_paper_without_plan",
]
FULL_MODE_ORDER = ["cpu_baseline", *GPU_MODE_ORDER]

MODE_LABELS = {
    "cpu_baseline": "CPU Baseline",
    "q_paper_with_plan": "Q (With Plan)",
    "q_paper_with_plan_streaming_io": "Q (With Plan, Stream IO)",
    "q_paper_without_plan": "Q (Without Plan)",
    "c_paper_with_plan": "C (With Plan)",
    "c_paper_with_plan_streaming_io": "C (With Plan, Stream IO)",
    "c_paper_without_plan": "C (Without Plan)",
}

MODE_COLORS = {
    "cpu_baseline": "#f4e7c5",
    "q_paper_with_plan": "#c6dbef",
    "q_paper_with_plan_streaming_io": "#6db1bf",
    "q_paper_without_plan": "#9ecae1",
    "c_paper_with_plan": "#c7e9c0",
    "c_paper_with_plan_streaming_io": "#90be6d",
    "c_paper_without_plan": "#74c476",
}


def extract_label(sweep_path):
    name = os.path.basename(os.path.normpath(sweep_path))
    match = re.match(r"sweep_(.+)$", name)
    return match.group(1) if match else name


def parse_log(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    val_match = re.search(r"value bytes:\s*(\d+)", content)
    key_match = re.search(r"key bytes:\s*(\d+)", content)
    mode_match = re.search(r"gpu mode:\s*([^\n]+)", content)
    cpu_wall = re.search(r"CPU total \(Wall\):\s*min=([\d.]+)\s*ms", content)
    gpu_wall = re.search(r"GPU total \(Wall\):\s*min=([\d.]+)\s*ms", content)
    output_bytes_match = re.search(r"output bytes\s*(\d+)", content)

    val_bytes = int(val_match.group(1)) if val_match else 0
    key_bytes = int(key_match.group(1)) if key_match else 16
    mode = mode_match.group(1).strip() if mode_match else ""
    cpu_wall_min = float(cpu_wall.group(1)) if cpu_wall else 0.0
    gpu_wall_min = float(gpu_wall.group(1)) if gpu_wall else 0.0
    output_bytes = float(output_bytes_match.group(1)) if output_bytes_match else 0.0
    profile_only = "profile-only mode: enabled" in content

    ops = output_bytes / (key_bytes + val_bytes) if (key_bytes + val_bytes) > 0 else 0.0
    cpu_throughput = ops / (cpu_wall_min / 1000.0) if cpu_wall_min > 0 else 0.0
    gpu_throughput = ops / (gpu_wall_min / 1000.0) if gpu_wall_min > 0 else 0.0
    gpu_only = (
        "cpu baseline: disabled (--gpu_only)" in content
        or "CPU baseline: skipped (gpu_only mode)" in content
        or "cpu baseline: disabled (--profile_only)" in content
    )

    return {
        "value_bytes": val_bytes,
        "mode": mode,
        "cpu_throughput": cpu_throughput,
        "gpu_throughput": gpu_throughput,
        "gpu_only": gpu_only,
        "profile_only": profile_only,
    }


def parse_host_metrics_summary(log_path):
    base_name = os.path.basename(log_path)
    match = re.match(r"result_val(\d+)B_(.+)\.log$", base_name)
    if not match:
        return None

    val_bytes, mode = match.groups()
    summary_path = os.path.join(
        os.path.dirname(log_path),
        "host_metrics",
        f"val{val_bytes}B_{mode}",
        "summary.json",
    )
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    return {
        "sample_count": int(summary.get("sample_count", 0)),
        "process_cpu_pct": float(summary.get("avg_process_cpu_pct", 0.0)),
        "device_util_pct": float(summary.get("avg_device_util_pct", 0.0)),
        "read_bw_mib_per_sec": float(summary.get("avg_device_rkib_per_sec", 0.0)) / 1024.0,
        "write_bw_mib_per_sec": float(summary.get("avg_device_wkib_per_sec", 0.0)) / 1024.0,
    }


def mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def plot_grouped_bars(ax, x, width, series_map, sorted_vals, mode_order, ylabel, title, ylim=None):
    if len(mode_order) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-(len(mode_order) - 1) / 2.0, (len(mode_order) - 1) / 2.0, len(mode_order)) * width

    for idx, mode in enumerate(mode_order):
        series = [series_map[mode].get(v, 0.0) for v in sorted_vals]
        ax.bar(
            x + offsets[idx],
            series,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            edgecolor="black",
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Value Size (Bytes)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_vals)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)


def build_sorted_values(parsed_logs):
    return sorted({entry["value_bytes"] for entry in parsed_logs if entry["value_bytes"] > 0})


def main(sweep_dir, graph_dir, label=None, gpu_only_override=False, profile_only_override=False):
    print(f"Reading logs from: {sweep_dir}")

    if label is None:
        label = extract_label(sweep_dir)

    log_files = sorted(glob.glob(os.path.join(sweep_dir, "result_val*B_*.log")))
    parsed_logs = [parse_log(path) for path in log_files]
    sorted_vals = build_sorted_values(parsed_logs)
    if not sorted_vals:
        print("No valid plotted data.")
        return

    gpu_only_run = gpu_only_override or any(entry["gpu_only"] for entry in parsed_logs)
    profile_only_run = profile_only_override or any(entry["profile_only"] for entry in parsed_logs)
    mode_order = GPU_MODE_ORDER if gpu_only_run else FULL_MODE_ORDER
    gpu_metrics = {mode: defaultdict(dict) for mode in GPU_MODE_ORDER}
    cpu_samples = defaultdict(lambda: defaultdict(list))
    host_metrics_found = False

    for file_path, parsed in zip(log_files, parsed_logs):
        val = parsed["value_bytes"]
        mode = parsed["mode"]
        if val <= 0 or mode not in gpu_metrics:
            continue

        gpu_metrics[mode][val]["throughput"] = parsed["gpu_throughput"]
        if not gpu_only_run:
            cpu_samples[val]["throughput"].append(parsed["cpu_throughput"])

        host_metrics = parse_host_metrics_summary(file_path)
        if host_metrics and host_metrics["sample_count"] > 0:
            host_metrics_found = True
            gpu_metrics[mode][val]["process_cpu_pct"] = host_metrics["process_cpu_pct"]
            gpu_metrics[mode][val]["device_util_pct"] = host_metrics["device_util_pct"]
            gpu_metrics[mode][val]["read_bw_mib_per_sec"] = host_metrics["read_bw_mib_per_sec"]
            gpu_metrics[mode][val]["write_bw_mib_per_sec"] = host_metrics["write_bw_mib_per_sec"]

    x = np.arange(len(sorted_vals))
    width = min(0.18, 0.78 / max(len(mode_order), 1))
    os.makedirs(graph_dir, exist_ok=True)

    tp_out_path = os.path.join(graph_dir, f"throughput_{label}_sweep.png")
    if profile_only_run:
        remove_if_exists(tp_out_path)
        print("Skipping throughput plot: profile-only logs omit timed throughput summaries.")
    else:
        throughput_series = {}
        if not gpu_only_run:
            throughput_series["cpu_baseline"] = {
                value: mean_or_zero(cpu_samples[value]["throughput"]) for value in sorted_vals
            }
        for mode in GPU_MODE_ORDER:
            throughput_series[mode] = {
                value: gpu_metrics[mode].get(value, {}).get("throughput", 0.0) for value in sorted_vals
            }

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_grouped_bars(
            ax,
            x,
            width,
            throughput_series,
            sorted_vals,
            mode_order,
            "Throughput (Ops/s)",
            f"Throughput vs Value Size ({label})",
        )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            fancybox=True,
            shadow=True,
            ncol=min(3, len(mode_order)),
        )

        plt.tight_layout()
        plt.savefig(tp_out_path)
        print(f"Throughput graph saved to: {tp_out_path}")
        plt.close()

    util_out_path = os.path.join(graph_dir, f"utilization_{label}_sweep.png")
    io_out_path = os.path.join(graph_dir, f"io_utilization_{label}_sweep.png")

    if gpu_only_run and host_metrics_found:
        util_series = {
            mode: {
                value: gpu_metrics[mode].get(value, {}).get("process_cpu_pct", 0.0) for value in sorted_vals
            }
            for mode in GPU_MODE_ORDER
        }
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        plot_grouped_bars(
            ax2,
            x,
            width,
            util_series,
            sorted_vals,
            GPU_MODE_ORDER,
            "CPU Utilization (%)",
            f"Process CPU Utilization During GPU-Only Compaction ({label})",
        )
        ax2.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            fancybox=True,
            shadow=True,
            ncol=min(3, len(GPU_MODE_ORDER)),
        )
        plt.tight_layout()
        plt.savefig(util_out_path)
        print(f"Utilization graph saved to: {util_out_path}")
        plt.close()

        read_bw_series = {
            mode: {
                value: gpu_metrics[mode].get(value, {}).get("read_bw_mib_per_sec", 0.0) for value in sorted_vals
            }
            for mode in GPU_MODE_ORDER
        }
        write_bw_series = {
            mode: {
                value: gpu_metrics[mode].get(value, {}).get("write_bw_mib_per_sec", 0.0) for value in sorted_vals
            }
            for mode in GPU_MODE_ORDER
        }
        device_util_series = {
            mode: {
                value: gpu_metrics[mode].get(value, {}).get("device_util_pct", 0.0) for value in sorted_vals
            }
            for mode in GPU_MODE_ORDER
        }

        fig3, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
        plot_grouped_bars(
            axes[0],
            x,
            width,
            read_bw_series,
            sorted_vals,
            GPU_MODE_ORDER,
            "MiB/s",
            f"Measured SSD Read Bandwidth ({label})",
        )
        plot_grouped_bars(
            axes[1],
            x,
            width,
            write_bw_series,
            sorted_vals,
            GPU_MODE_ORDER,
            "MiB/s",
            f"Measured SSD Write Bandwidth ({label})",
        )
        plot_grouped_bars(
            axes[2],
            x,
            width,
            device_util_series,
            sorted_vals,
            GPU_MODE_ORDER,
            "Utilization (%)",
            f"Measured SSD Utilization ({label})",
        )
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig3.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            fancybox=True,
            shadow=True,
            ncol=min(3, len(GPU_MODE_ORDER)),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(io_out_path)
        print(f"IO graph saved to: {io_out_path}")
        plt.close()
    else:
        remove_if_exists(util_out_path)
        remove_if_exists(io_out_path)
        print("Skipping CPU/IO plots: host metrics are only emitted for GPU-only sweeps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPComp benchmark results.")
    parser.add_argument(
        "--sweep_dir",
        default=None,
        help="Path to a specific results directory (e.g., sweep_results/8mb-sst_24sst)",
    )
    parser.add_argument(
        "--results_dir",
        default=DEFAULT_BASE_DIR,
        help="Directory containing result folders (used if --sweep_dir not given)",
    )
    parser.add_argument(
        "--graphs_dir",
        default=DEFAULT_GRAPH_DIR,
        help="Directory for output graphs",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Custom label for output filenames (e.g., 8mb-sst_24sst)",
    )
    parser.add_argument(
        "--gpu_only",
        action="store_true",
        help="Treat the sweep as GPU-only and render host-metrics CPU/IO plots when available",
    )
    parser.add_argument(
        "--profile_only",
        action="store_true",
        help="Treat the sweep as profile-only and skip throughput plotting",
    )
    args = parser.parse_args()

    if args.sweep_dir:
        main(
            args.sweep_dir,
            args.graphs_dir,
            label=args.label,
            gpu_only_override=args.gpu_only,
            profile_only_override=args.profile_only,
        )
    else:
        sweep_dirs = sorted(
            path for path in glob.glob(os.path.join(args.results_dir, "*"))
            if os.path.isdir(path)
        )[-1:]
        if not sweep_dirs:
            print("No sweep directories found.")
        else:
            main(
                sweep_dirs[-1],
                args.graphs_dir,
                label=args.label,
                gpu_only_override=args.gpu_only,
                profile_only_override=args.profile_only,
            )
