#!/usr/bin/env python3
"""Analyze CPU vs GPU compaction comparison results.

Reads the experiment directory produced by run_comparison.sh and generates:
  - comparison/summary.md   — tables of CPU vs GPU throughput, utilization
  - comparison/throughput_utilization.png — bar+line chart (throughput vs CPU/IO)

Usage:
    python3 analyze_comparison.py <experiment_dir>

Example:
    python3 analyze_comparison.py atest/initial_gpu_cpu_compaction_microbenchmark/quick-test
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


GPU_MODE_ORDER = [
    "q_paper_with_plan",
    "q_paper_without_plan",
    "c_paper_with_plan",
    "c_paper_without_plan",
]

GPU_MODE_LABELS = {
    "q_paper_with_plan": "q_plan",
    "q_paper_without_plan": "q_no_plan",
    "c_paper_with_plan": "c_plan",
    "c_paper_without_plan": "c_no_plan",
}

GPU_MODE_COLORS = {
    "q_paper_with_plan": "#33b39f",
    "q_paper_without_plan": "#16897b",
    "c_paper_with_plan": "#f4a261",
    "c_paper_without_plan": "#e76f51",
}

CPU_BASELINE_COLOR = "#d9d9d9"
CPU_SUBCOMP_COLORS = {
    1: CPU_BASELINE_COLOR,
    2: "#dbe9f6",
    4: "#b6d3eb",
    8: "#8dbfe0",
    16: "#5f9ed1",
    32: "#2f6ba7",
    64: "#1d4f7a",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_cpu_summary_csv(csv_path: Path) -> list[dict]:
    """Read the CPU sweep summary_metrics.csv into a list of row dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_gpu_breakdown_log(log_path: Path) -> dict:
    """Extract CPU baseline and GPU timing from a gpcomp_bench result log."""
    text = log_path.read_text()
    result: dict = {"path": str(log_path)}

    # Parse mode from filename: result_val32B_q_paper_with_plan.log
    m = re.match(r"result_val(\d+)B_(.+)\.log", log_path.name)
    if m:
        result["value_size"] = int(m.group(1))
        result["mode"] = m.group(2)

    # GPU total
    m = re.search(r"GPU total \(Wall\): min=([\d.]+) ms\s+mean=([\d.]+) \+- ([\d.]+) ms", text)
    if m:
        result["gpu_total_min_ms"] = float(m.group(1))
        result["gpu_total_mean_ms"] = float(m.group(2))
        result["gpu_total_std_ms"] = float(m.group(3))

    # CPU total (synthetic baseline from gpcomp_bench)
    m = re.search(r"CPU total \(Wall\): min=([\d.]+) ms\s+mean=([\d.]+) \+- ([\d.]+) ms", text)
    if m:
        result["cpu_synthetic_min_ms"] = float(m.group(1))
        result["cpu_synthetic_mean_ms"] = float(m.group(2))
        result["cpu_synthetic_std_ms"] = float(m.group(3))

    # Speedup
    m = re.search(r"Speedup: ([\d.]+)x", text)
    if m:
        result["speedup"] = float(m.group(1))

    # Output bytes (from GPU best run breakdown)
    m = re.search(r"Best GPU run breakdown.*?output bytes (\d+)", text, re.DOTALL)
    if m:
        result["output_bytes"] = int(m.group(1))

    # Input bytes
    m = re.search(r"input bytes (\d+)", text)
    if m:
        result["input_bytes"] = int(m.group(1))

    return result


def parse_host_metrics_summary(summary_path: Path) -> dict:
    """Read a host_metrics summary.json."""
    return json.loads(summary_path.read_text())


def compute_throughput_mrecords(total_ms: float, input_bytes: int,
                                key_size: int = 16, value_size: int = 32) -> float:
    """Compute throughput in million records/s."""
    if total_ms <= 0:
        return 0.0
    record_size = key_size + value_size
    records = input_bytes / record_size
    return records / (total_ms / 1000.0) / 1e6


def input_case_label(input_data_mb: int | float, requested_input_sst_count: int | float) -> str:
    if int(requested_input_sst_count) > 0:
        return f"{int(requested_input_sst_count)} SSTs ({int(input_data_mb)}MB)"
    return f"{int(input_data_mb)}MB"


def input_tag(input_data_mb: int | float, requested_input_sst_count: int | float) -> str:
    if int(requested_input_sst_count) > 0:
        return f"{int(requested_input_sst_count)}sst"
    return f"{int(input_data_mb)}MB"


def format_config_label(config_type: str, subcompactions: int | None = None,
                        mode: str | None = None) -> str:
    """Return the config label used in comparison figures."""
    if config_type == "cpu":
        return f"sub {subcompactions}"
    if mode is None:
        return "gpu"
    return GPU_MODE_LABELS.get(mode, mode)


def config_color(config_type: str, subcompactions: int | None = None,
                 mode: str | None = None) -> str:
    """Return the comparison-series color for a config."""
    if config_type == "cpu":
        return CPU_SUBCOMP_COLORS.get(subcompactions or 1, "#2f6ba7")
    return GPU_MODE_COLORS.get(mode or "", "#666666")


def config_sort_key(config_type: str, subcompactions: int | None = None,
                    mode: str | None = None) -> tuple[int, int]:
    """Keep CPU subcompactions first, then GPU modes in comparison order."""
    if config_type == "cpu":
        return (0, subcompactions or 0)
    try:
        mode_idx = GPU_MODE_ORDER.index(mode or "")
    except ValueError:
        mode_idx = len(GPU_MODE_ORDER)
    return (1, mode_idx)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def collect_cpu_data(experiment_dir: Path) -> list[dict]:
    """Collect CPU sweep data points with host metrics."""
    cpu_dir = experiment_dir / "cpu"
    csv_path = cpu_dir / "analysis" / "summary_metrics.csv"
    if not csv_path.exists():
        print(f"warning: CPU summary CSV not found: {csv_path}", file=sys.stderr)
        return []

    rows = parse_cpu_summary_csv(csv_path)
    data = []
    for row in rows:
        data.append({
            "label": f"CPU sub={row['requested_subcompactions']}",
            "type": "cpu",
            "value_size": int(float(row["value_size"])),
            "subcompactions": int(float(row["requested_subcompactions"])),
            "throughput_mrecords": float(row["logical_input_throughput_mrecords_per_sec"]),
            "throughput_mrecords_std": float(row.get("logical_input_throughput_mrecords_per_sec_std", 0)),
            "avg_cpu_pct": float(row.get("avg_process_cpu_pct", 0)),
            "avg_io_pct": float(row.get("avg_device_util_pct", 0)),
        })
    return data


def collect_gpu_breakdown_data(experiment_dir: Path) -> list[dict]:
    """Collect GPU breakdown data (CPU synthetic baseline + GPU throughput)."""
    data = []
    # Support both current gpu_breakdown* and legacy sweep_gpu_breakdown* names.
    breakdown_dirs = sorted(experiment_dir.glob("gpu_breakdown*"))
    if not breakdown_dirs:
        breakdown_dirs = sorted(experiment_dir.glob("sweep_gpu_breakdown*"))
    if not breakdown_dirs:
        print("warning: no GPU breakdown directory found", file=sys.stderr)
        return []

    breakdown_dir = breakdown_dirs[0]
    for log_path in sorted(breakdown_dir.glob("result_val*B_*.log")):
        parsed = parse_gpu_breakdown_log(log_path)
        if "gpu_total_mean_ms" not in parsed:
            continue
        input_bytes = parsed.get("input_bytes", 0)
        value_size = parsed.get("value_size", 32)
        data.append(parsed)
    return data


def collect_gpu_host_metrics(experiment_dir: Path) -> list[dict]:
    """Collect host metrics from the GPU profile-only run."""
    data = []
    host_dirs = sorted(experiment_dir.glob("gpu_host_metrics*"))
    if not host_dirs:
        host_dirs = sorted(experiment_dir.glob("sweep_gpu_host_metrics*"))
    if not host_dirs:
        return []

    host_dir = host_dirs[0]
    metrics_root = host_dir / "host_metrics"
    if not metrics_root.exists():
        return []

    for summary_path in sorted(metrics_root.glob("*/summary.json")):
        m = re.match(r"val(\d+)B_(.+)", summary_path.parent.name)
        if not m:
            continue
        value_size = int(m.group(1))
        mode = m.group(2)
        summary = parse_host_metrics_summary(summary_path)
        data.append({
            "value_size": value_size,
            "mode": mode,
            "avg_cpu_pct": summary.get("avg_process_cpu_pct", 0),
            "avg_io_pct": summary.get("avg_device_util_pct", 0),
        })
    return data


# ---------------------------------------------------------------------------
# Output: Markdown summary
# ---------------------------------------------------------------------------

def write_summary_md(experiment_dir: Path, cpu_data: list[dict],
                     gpu_breakdown: list[dict], gpu_host_metrics: list[dict],
                     output_dir: Path):
    """Write comparison/summary.md with tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "summary.md"

    lines = ["# CPU vs GPU Compaction Comparison\n"]

    # -- Table 1: CPU single-threaded (RocksDB) vs CPU synthetic (gpcomp_bench) --
    lines.append("## CPU Single-Threaded: RocksDB vs Synthetic Baseline\n")
    lines.append("Comparison of RocksDB compactall (subcomp=1) wall time vs the")
    lines.append("synthetic CPU compaction baseline from gpcomp_bench.\n")

    cpu_sub1 = [d for d in cpu_data if d["subcompactions"] == 1]
    if cpu_sub1 and gpu_breakdown:
        value_sizes = sorted(set(d["value_size"] for d in cpu_sub1))
        lines.append("| Value Size | RocksDB sub=1 (ms) | Throughput (Mrec/s) |"
                      " Synthetic CPU (ms) | Throughput (Mrec/s) | Ratio |")
        lines.append("|------------|-------------------|--------------------"
                      "|-------------------|---------------------|-------|")

        for vs in value_sizes:
            cpu_row = next((d for d in cpu_sub1 if d["value_size"] == vs), None)
            # Use first GPU mode's CPU baseline for this value size
            gpu_rows = [g for g in gpu_breakdown if g.get("value_size") == vs]
            if cpu_row and gpu_rows:
                gpu_row = gpu_rows[0]
                cpu_ms = cpu_row["throughput_mrecords"]
                syn_ms = gpu_row.get("cpu_synthetic_mean_ms", 0)
                syn_std = gpu_row.get("cpu_synthetic_std_ms", 0)
                input_bytes = gpu_row.get("input_bytes", 0)
                syn_tput = compute_throughput_mrecords(syn_ms, input_bytes, 16, vs)

                # RocksDB time from throughput (reverse): records/s -> time
                rocksdb_tput = cpu_row["throughput_mrecords"]
                rocksdb_tput_std = cpu_row["throughput_mrecords_std"]

                # Get RocksDB wall time from CSV data
                rocksdb_rows = [r for r in parse_cpu_summary_csv(
                    experiment_dir / "cpu" / "analysis" / "summary_metrics.csv")
                    if int(float(r["value_size"])) == vs
                    and int(float(r["requested_subcompactions"])) == 1]
                if rocksdb_rows:
                    rocksdb_wall_ms = float(rocksdb_rows[0]["total_compaction_wall_s"]) * 1000
                    rocksdb_wall_std = float(rocksdb_rows[0].get("total_compaction_wall_s_std", 0)) * 1000
                else:
                    rocksdb_wall_ms = 0
                    rocksdb_wall_std = 0

                ratio = syn_ms / rocksdb_wall_ms if rocksdb_wall_ms > 0 else 0
                lines.append(
                    f"| {vs}B | {rocksdb_wall_ms:.1f} +/- {rocksdb_wall_std:.1f} |"
                    f" {rocksdb_tput:.2f} +/- {rocksdb_tput_std:.2f} |"
                    f" {syn_ms:.1f} +/- {syn_std:.1f} |"
                    f" {syn_tput:.2f} |"
                    f" {ratio:.2f}x |"
                )
    lines.append("")

    # -- Table 2: CPU multithreading sweep --
    lines.append("## CPU Multithreading Sweep\n")
    if cpu_data:
        value_sizes = sorted(set(d["value_size"] for d in cpu_data))
        for vs in value_sizes:
            lines.append(f"### Value Size: {vs}B\n")
            lines.append("| Subcompactions | Throughput (Mrec/s) | Speedup vs sub=1 |"
                          " Avg CPU % | Avg IO % |")
            lines.append("|----------------|--------------------|-----------------"
                          "|-----------|----------|")
            vs_data = sorted([d for d in cpu_data if d["value_size"] == vs],
                             key=lambda d: d["subcompactions"])
            baseline = next((d["throughput_mrecords"] for d in vs_data
                             if d["subcompactions"] == 1), 1.0)
            for d in vs_data:
                speedup = d["throughput_mrecords"] / baseline if baseline > 0 else 0
                lines.append(
                    f"| {d['subcompactions']} |"
                    f" {d['throughput_mrecords']:.2f} +/- {d['throughput_mrecords_std']:.2f} |"
                    f" {speedup:.2f}x |"
                    f" {d['avg_cpu_pct']:.1f}% |"
                    f" {d['avg_io_pct']:.1f}% |"
                )
            lines.append("")

    # -- Table 3: GPU compaction times --
    lines.append("## GPU Compaction Results\n")
    if gpu_breakdown:
        lines.append("| Value Size | Mode | GPU Time (ms) | CPU Synthetic (ms) |"
                      " Speedup | Avg CPU % | Avg IO % |")
        lines.append("|------------|------|---------------|-------------------"
                      "|---------|-----------|----------|")

        for g in sorted(gpu_breakdown, key=lambda x: (x.get("value_size", 0), x.get("mode", ""))):
            vs = g.get("value_size", 0)
            mode = g.get("mode", "")
            gpu_ms = g.get("gpu_total_mean_ms", 0)
            gpu_std = g.get("gpu_total_std_ms", 0)
            syn_ms = g.get("cpu_synthetic_mean_ms", 0)
            syn_std = g.get("cpu_synthetic_std_ms", 0)
            speedup = g.get("speedup", 0)

            # Find matching host metrics
            hm = next((h for h in gpu_host_metrics
                        if h["value_size"] == vs and h["mode"] == mode), None)
            cpu_pct = hm["avg_cpu_pct"] if hm else 0
            io_pct = hm["avg_io_pct"] if hm else 0

            lines.append(
                f"| {vs}B | {mode} |"
                f" {gpu_ms:.1f} +/- {gpu_std:.1f} |"
                f" {syn_ms:.1f} +/- {syn_std:.1f} |"
                f" {speedup:.2f}x |"
                f" {cpu_pct:.1f}% |"
                f" {io_pct:.1f}% |"
            )
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")


# ---------------------------------------------------------------------------
# Output: Throughput + utilization figure
# ---------------------------------------------------------------------------

def plot_throughput_utilization(experiment_dir: Path, cpu_data: list[dict],
                                gpu_breakdown: list[dict],
                                gpu_host_metrics: list[dict],
                                output_dir: Path):
    """Create a bar+line chart: throughput bars with CPU/IO utilization lines."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by value size
    all_value_sizes = set()
    for d in cpu_data:
        all_value_sizes.add(d["value_size"])
    for d in gpu_breakdown:
        all_value_sizes.add(d.get("value_size", 0))

    for vs in sorted(all_value_sizes):
        points = []

        # CPU data points
        vs_cpu = sorted([d for d in cpu_data if d["value_size"] == vs],
                        key=lambda d: d["subcompactions"])
        for d in vs_cpu:
            points.append({
                "config_type": "cpu",
                "subcompactions": d["subcompactions"],
                "mode": None,
                "label": format_config_label("cpu", subcompactions=d["subcompactions"]),
                "color": config_color("cpu", subcompactions=d["subcompactions"]),
                "throughput": d["throughput_mrecords"],
                "throughput_err": d["throughput_mrecords_std"],
                "cpu_pct": d["avg_cpu_pct"],
                "io_pct": d["avg_io_pct"],
            })

        # GPU data points
        vs_gpu = sorted(
            [g for g in gpu_breakdown if g.get("value_size") == vs],
            key=lambda x: config_sort_key("gpu", mode=x.get("mode", "")),
        )
        for g in vs_gpu:
            mode = g.get("mode", "")
            gpu_ms = g.get("gpu_total_mean_ms", 0)
            gpu_std = g.get("gpu_total_std_ms", 0)
            input_bytes = g.get("input_bytes", 0)
            tput = compute_throughput_mrecords(gpu_ms, input_bytes, 16, vs)
            # Error propagation: dT/T = dt/t
            tput_err = tput * (gpu_std / gpu_ms) if gpu_ms > 0 else 0

            hm = next((h for h in gpu_host_metrics
                        if h["value_size"] == vs and h["mode"] == mode), None)
            points.append({
                "config_type": "gpu",
                "subcompactions": None,
                "mode": mode,
                "label": format_config_label("gpu", mode=mode),
                "color": config_color("gpu", mode=mode),
                "throughput": tput,
                "throughput_err": tput_err,
                "cpu_pct": hm["avg_cpu_pct"] if hm else 0,
                "io_pct": hm["avg_io_pct"] if hm else 0,
            })

        points.sort(
            key=lambda p: config_sort_key(
                p["config_type"],
                subcompactions=p["subcompactions"],
                mode=p["mode"],
            )
        )

        if not points:
            continue

        labels = [p["label"] for p in points]
        throughputs = [p["throughput"] for p in points]
        throughput_errs = [p["throughput_err"] for p in points]
        cpu_pcts = [p["cpu_pct"] for p in points]
        io_pcts = [p["io_pct"] for p in points]
        bar_colors = [p["color"] for p in points]

        x = np.arange(len(labels))
        fig, ax1 = plt.subplots(figsize=(max(8.8, len(labels) * 1.25), 5.5))

        # Bars: throughput
        ax1.bar(
            x,
            throughputs,
            yerr=throughput_errs,
            capsize=3,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
        ax1.set_ylabel("Throughput (Million rec/s)", fontsize=11)
        ax1.set_xlabel("Configuration", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylim(bottom=0)
        ax1.grid(axis="y", alpha=0.3, zorder=0)

        # CPU gets its own axis because process CPU can exceed 100% on multi-core runs.
        ax2 = ax1.twinx()
        cpu_line = ax2.plot(
            x, cpu_pcts, "r--x", linewidth=2, markersize=7, label="CPU", zorder=4
        )[0]
        ax2.set_ylabel("CPU Utilization (%)", fontsize=11, color="red")
        ax2.tick_params(axis="y", colors="red")
        cpu_max = max(cpu_pcts, default=0)
        ax2.set_ylim(0, max(100.0, cpu_max * 1.15))

        # IO gets a dedicated fixed 0-100% axis so saturation is visually stable across plots.
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 52))
        io_line = ax3.plot(
            x, io_pcts, "g--+", linewidth=2, markersize=8, label="IO", zorder=4
        )[0]
        ax3.set_ylabel("I/O Utilization (%)", fontsize=11, color="green")
        ax3.tick_params(axis="y", colors="green")
        ax3.set_ylim(0, 100)
        ax3.set_yticks(np.arange(0, 101, 20))
        ax3.axhline(100, color="green", linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
        ax3.text(
            1.01,
            100,
            "100%",
            color="green",
            fontsize=9,
            va="bottom",
            ha="left",
            transform=ax3.get_yaxis_transform(),
        )

        # Match the comparison figure's config legend, then keep a small line legend for utilization.
        legend_handles = [
            Patch(
                facecolor=p["color"],
                edgecolor="black",
                linewidth=0.7,
                label=p["label"],
            )
            for p in points
        ]
        fig.legend(
            handles=legend_handles,
            labels=[p["label"] for p in points],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=min(len(points), 5),
            frameon=False,
            fontsize=11,
            columnspacing=1.6,
            handletextpad=0.7,
        )

        ax2.legend(
            [cpu_line, io_line],
            ["CPU", "IO"],
            loc="upper left",
            fontsize=9,
            frameon=True,
        )

        ax1.set_title(f"Compaction Throughput vs Utilization (value={vs}B)", fontsize=12, pad=10)
        fig.tight_layout(rect=(0, 0, 0.93, 0.84))
        out_path = output_dir / f"throughput_utilization_value_{vs}B.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_cpu_baseline_breakdown(experiment_dir: Path, output_dir: Path):
    """Recreate the CPU baseline time-breakdown figure inside comparison/."""
    cpu_summary_path = experiment_dir / "cpu" / "analysis" / "summary_metrics.csv"
    if not cpu_summary_path.exists():
        print(f"warning: CPU summary CSV not found for breakdown plot: {cpu_summary_path}",
              file=sys.stderr)
        return

    rows = parse_cpu_summary_csv(cpu_summary_path)
    baseline_rows = []
    for row in rows:
        if int(float(row["requested_subcompactions"])) != 1:
            continue
        baseline_rows.append({
            "sst_size_mb": int(float(row["sst_size_mb"])),
            "requested_input_sst_count": int(float(row.get("requested_input_sst_count", 0) or 0)),
            "input_data_mb": int(float(row["input_data_mb"])),
            "value_size": int(float(row["value_size"])),
            "compute_pct": float(row.get("compute_pct", 0)),
            "read_pct": float(row.get("read_pct", 0)),
            "write_pct": float(row.get("write_pct", 0)),
        })

    if not baseline_rows:
        print("warning: no CPU subcomp=1 rows found for breakdown plot", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "compute_pct": "#fb8072",
        "read_pct": "#1bb7b3",
        "write_pct": "#4666d5",
    }

    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for row in baseline_rows:
        key = (
            row["sst_size_mb"],
            row["requested_input_sst_count"],
            row["input_data_mb"],
        )
        grouped.setdefault(key, []).append(row)

    for (sst_size_mb, requested_input_sst_count, input_data_mb), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["value_size"])
        x = np.arange(len(ordered))
        compute = np.array([row["compute_pct"] for row in ordered])
        read = np.array([row["read_pct"] for row in ordered])
        write = np.array([row["write_pct"] for row in ordered])

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x, compute, color=colors["compute_pct"], label="Computation")
        ax.bar(x, read, bottom=compute, color=colors["read_pct"], label="Read")
        ax.bar(x, write, bottom=compute + read, color=colors["write_pct"], label="Write")

        input_label = input_case_label(input_data_mb, requested_input_sst_count)
        file_tag = input_tag(input_data_mb, requested_input_sst_count)
        ax.set_title(
            f"Normalized Compaction Time Breakdown\nSST={sst_size_mb}MB, Input={input_label}",
            pad=26,
        )
        ax.set_xlabel("Value Size (Bytes)")
        ax.set_ylabel("Normalized Execution Time (%)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(row["value_size"]) for row in ordered])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        fig.legend(
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        out_path = output_dir / f"breakdown_sst_{sst_size_mb}MB_input_{file_tag}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze CPU vs GPU compaction comparison")
    parser.add_argument("experiment_dir", type=Path, help="Experiment directory from run_comparison.sh")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        print(f"error: {experiment_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = experiment_dir / "comparison"

    cpu_data = collect_cpu_data(experiment_dir)
    gpu_breakdown = collect_gpu_breakdown_data(experiment_dir)
    gpu_host_metrics = collect_gpu_host_metrics(experiment_dir)

    print(f"Collected {len(cpu_data)} CPU data points, "
          f"{len(gpu_breakdown)} GPU breakdown logs, "
          f"{len(gpu_host_metrics)} GPU host metric summaries")

    write_summary_md(experiment_dir, cpu_data, gpu_breakdown, gpu_host_metrics, output_dir)
    plot_cpu_baseline_breakdown(experiment_dir, output_dir)
    plot_throughput_utilization(experiment_dir, cpu_data, gpu_breakdown,
                                gpu_host_metrics, output_dir)


if __name__ == "__main__":
    main()
