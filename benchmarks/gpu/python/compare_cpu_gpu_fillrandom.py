#!/usr/bin/env python3
"""
compare_cpu_gpu_fillrandom.py  —  Compare CPU fillrandom compaction performance
against the GPU compaction hook benchmark across value sizes.

CPU side  : benchmarks/cpu/results/fillrandom/value_*/subcomp_1/<RUN_ID>/
GPU side  : benchmarks/gpu/results/fillrandom_gpu/value_*/rocksdb_hook/<RUN_ID>/

Usage (from repository root):
  python3 benchmarks/gpu/python/compare_cpu_gpu_fillrandom.py \\
      --cpu-results benchmarks/cpu/results \\
      --gpu-results benchmarks/gpu/results/fillrandom_gpu \\
      [--cpu-run-id 0309_1637] [--gpu-run-id 0309_1829] \\
      [--plot] [--out-dir /tmp/compare_plots]

If --cpu-run-id / --gpu-run-id are omitted the most recent run directory is used.
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

VALUE_SIZES = [32, 64, 128, 256]

# ── helpers ──────────────────────────────────────────────────────────────────


def _latest_run_dir(base: str) -> str | None:
    """Return the lexicographically latest sub-directory inside *base*."""
    entries = sorted(
        [e for e in glob.glob(os.path.join(base, "*")) if os.path.isdir(e)]
    )
    return entries[-1] if entries else None


def _extract_stat(log_text: str, stat_name: str, field: str = "P50") -> float | None:
    """
    Parse a RocksDB histogram line, e.g.:
      rocksdb.compaction.times.micros P50 : 676333.3 P95 : ...
    Returns the requested percentile/field value or None.
    """
    pattern = re.compile(
        rf"{re.escape(stat_name)}\s+{field}\s*:\s*([0-9.]+)", re.IGNORECASE
    )
    m = pattern.search(log_text)
    return float(m.group(1)) if m else None


def _extract_count_sum(log_text: str, stat_name: str):
    """Return (count, sum) for a histogram stat line."""
    count_pat = re.compile(
        rf"{re.escape(stat_name)}.*?COUNT\s*:\s*([0-9]+).*?SUM\s*:\s*([0-9]+)",
        re.IGNORECASE,
    )
    m = count_pat.search(log_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_fillrandom_summary(log_text: str):
    """
    Parse the db_bench summary line:
      fillrandom   :       7.713 micros/op 129645 ops/sec  ... ;    5.9 MB/s
    Returns (micros_per_op, ops_per_sec, mb_per_sec) or (None, None, None).
    """
    m = re.search(
        r"fillrandom\s*:\s*([0-9.]+)\s+micros/op\s+([0-9.]+)\s+ops/sec[^;]+;\s*([0-9.]+)\s+MB/s",
        log_text,
    )
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


# ── loaders ──────────────────────────────────────────────────────────────────


def load_cpu_run(results_base: str, value_size: int, run_id: str | None):
    """Return a dict of CPU stats for *value_size* / subcomp_1."""
    base = os.path.join(results_base, "fillrandom", f"value_{value_size}", "subcomp_1")
    if run_id:
        run_dir = os.path.join(base, run_id)
    else:
        run_dir = _latest_run_dir(base)

    if not run_dir or not os.path.isdir(run_dir):
        return None

    log_file = os.path.join(run_dir, "db_bench.log")
    if not os.path.isfile(log_file):
        return None

    with open(log_file, "r", errors="replace") as f:
        text = f.read()

    micros_op, ops_sec, mb_sec = _extract_fillrandom_summary(text)
    comp_p50 = _extract_stat(text, "rocksdb.compaction.times.micros", "P50")
    comp_p95 = _extract_stat(text, "rocksdb.compaction.times.micros", "P95")
    comp_count, comp_sum = _extract_count_sum(text, "rocksdb.compaction.times.micros")

    read_count, read_sum_us = _extract_count_sum(text, "rocksdb.file.read.compaction.micros")
    prefetch_p50 = _extract_stat(text, "rocksdb.compaction.prefetch.bytes", "P50")

    # Estimate compaction read throughput: median prefetch request / P50 read latency
    read_p50_us = _extract_stat(text, "rocksdb.file.read.compaction.micros", "P50")
    read_throughput_mb = None
    if prefetch_p50 and read_p50_us and read_p50_us > 0:
        read_throughput_mb = (prefetch_p50 / (1024 * 1024)) / (read_p50_us / 1e6)

    return {
        "value_size": value_size,
        "run_dir": run_dir,
        "micros_per_op": micros_op,
        "ops_per_sec": ops_sec,
        "write_mb_per_sec": mb_sec,
        "comp_p50_us": comp_p50,
        "comp_p95_us": comp_p95,
        "comp_count": comp_count,
        "comp_sum_us": comp_sum,
        "read_p50_us": read_p50_us,
        "prefetch_p50_bytes": prefetch_p50,
        "read_throughput_mb": read_throughput_mb,
    }


def load_gpu_run(gpu_results_base: str, value_size: int, run_id: str | None):
    """Return a summary dict for the GPU run at *value_size*."""
    base = os.path.join(gpu_results_base, f"value_{value_size}", "rocksdb_hook")
    if run_id:
        run_dir = os.path.join(base, run_id)
    else:
        run_dir = _latest_run_dir(base)

    if not run_dir or not os.path.isdir(run_dir):
        return None

    csv_path = os.path.join(run_dir, "gpu_compaction_hook.csv")
    if not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path)
    # Drop warm-up rep=0 if more than 1 rep present
    if len(df) > 1:
        df = df[df["rep"] > 0]

    def col(preferred, fallback=None):
        if preferred in df.columns:
            return preferred
        if fallback and fallback in df.columns:
            return fallback
        return None

    in_col = col("input_file_reads_us", "input_draw_us")
    out_col = col("output_persist_to_disk_us", "output_persist_us")

    pipeline_med = df["gpu_pipeline_total_us"].median() if "gpu_pipeline_total_us" in df.columns else None
    pipeline_mean = df["gpu_pipeline_total_us"].mean() if "gpu_pipeline_total_us" in df.columns else None
    pipeline_std = df["gpu_pipeline_total_us"].std() if "gpu_pipeline_total_us" in df.columns else None

    in_med = df[in_col].median() if in_col else None
    out_med = df[out_col].median() if out_col else None

    input_bytes = df["input_bytes"].median() if "input_bytes" in df.columns else None
    output_bytes = df["output_bytes"].median() if "output_bytes" in df.columns else None

    read_throughput_mb = None
    if input_bytes and in_med and in_med > 0:
        read_throughput_mb = (input_bytes / (1024 * 1024)) / (in_med / 1e6)

    write_throughput_mb = None
    if output_bytes and out_med and out_med > 0:
        write_throughput_mb = (output_bytes / (1024 * 1024)) / (out_med / 1e6)

    merge_col = col("dummy_compaction_us")
    if merge_col is None:
        merge_col = col("input_kernel_us")

    return {
        "value_size": value_size,
        "run_dir": run_dir,
        "pipeline_median_us": pipeline_med,
        "pipeline_mean_us": pipeline_mean,
        "pipeline_std_us": pipeline_std,
        "input_read_median_us": in_med,
        "output_persist_median_us": out_med,
        "merge_median_us": df[merge_col].median() if merge_col in (df.columns if merge_col else []) else None,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "read_throughput_mb": read_throughput_mb,
        "write_throughput_mb": write_throughput_mb,
        "df": df,
    }


# ── print summary ─────────────────────────────────────────────────────────────


def print_comparison_table(cpu_data: dict, gpu_data: dict):
    header = (
        f"{'Value':>8}  "
        f"{'CPU ops/s':>12}  {'CPU write MB/s':>14}  "
        f"{'CPU comp P50 ms':>16}  {'CPU read BW MB/s':>17}  "
        f"{'GPU pipeline ms':>16}  {'GPU read BW MB/s':>17}  {'GPU write BW MB/s':>18}  "
        f"{'Speedup (pipeline)':>20}"
    )
    print()
    print("=== CPU vs GPU Fillrandom — Compaction Performance Comparison ===")
    print()
    print(header)
    print("-" * len(header))

    for vs in VALUE_SIZES:
        cpu = cpu_data.get(vs)
        gpu = gpu_data.get(vs)

        def fmt_f(val, decimals=1, suffix=""):
            return f"{val:.{decimals}f}{suffix}" if val is not None else "n/a"

        cpu_ops = fmt_f(cpu["ops_per_sec"], 0) if cpu else "n/a"
        cpu_write = fmt_f(cpu["write_mb_per_sec"]) if cpu else "n/a"
        cpu_comp_p50 = fmt_f(cpu["comp_p50_us"] / 1000 if cpu and cpu["comp_p50_us"] else None) if cpu else "n/a"
        cpu_read_bw = fmt_f(cpu["read_throughput_mb"]) if cpu else "n/a"

        gpu_pipe = fmt_f(gpu["pipeline_median_us"] / 1000 if gpu and gpu["pipeline_median_us"] else None) if gpu else "n/a"
        gpu_read_bw = fmt_f(gpu["read_throughput_mb"]) if gpu else "n/a"
        gpu_write_bw = fmt_f(gpu["write_throughput_mb"]) if gpu else "n/a"

        speedup = "n/a"
        if cpu and gpu and cpu.get("comp_p50_us") and gpu.get("pipeline_median_us"):
            s = cpu["comp_p50_us"] / gpu["pipeline_median_us"]
            speedup = f"{s:.2f}x"

        print(
            f"{vs:>8}B "
            f"{cpu_ops:>12}  {cpu_write:>14}  "
            f"{cpu_comp_p50:>16}  {cpu_read_bw:>17}  "
            f"{gpu_pipe:>16}  {gpu_read_bw:>17}  {gpu_write_bw:>18}  "
            f"{speedup:>20}"
        )

    print()
    print("Notes:")
    print("  CPU ops/s      : total fillrandom write throughput (ops/sec)")
    print("  CPU write MB/s : end-to-end fillrandom write bandwidth (MB/s)")
    print("  CPU comp P50   : median wall time of one CPU compaction job (ms)")
    print("  CPU read BW    : estimated compaction file-read bandwidth (prefetch_p50 / read_p50_latency)")
    print("  GPU pipeline   : median GPU compaction pipeline time for one hook call (ms)")
    print("  GPU read/write BW: input read and output write bandwidth inside the GPU hook (MB/s)")
    print("  Speedup        : CPU comp P50 / GPU pipeline median")
    print()


# ── plotting ──────────────────────────────────────────────────────────────────


def plot_comparison(cpu_data: dict, gpu_data: dict, out_dir: str):
    if not HAS_MPL:
        print("warning: matplotlib not available; skipping plots.", file=sys.stderr)
        return

    os.makedirs(out_dir, exist_ok=True)
    sizes = [vs for vs in VALUE_SIZES if vs in cpu_data or vs in gpu_data]

    # ── Figure 1: compaction latency comparison ──────────────────────────────
    cpu_comp_ms = [
        (cpu_data[vs]["comp_p50_us"] / 1000 if cpu_data.get(vs) and cpu_data[vs].get("comp_p50_us") else None)
        for vs in sizes
    ]
    gpu_pipe_ms = [
        (gpu_data[vs]["pipeline_median_us"] / 1000 if gpu_data.get(vs) and gpu_data[vs].get("pipeline_median_us") else None)
        for vs in sizes
    ]
    gpu_pipe_std = [
        (gpu_data[vs]["pipeline_std_us"] / 1000 if gpu_data.get(vs) and gpu_data[vs].get("pipeline_std_us") else 0)
        for vs in sizes
    ]

    x = range(len(sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        [v if v is not None else 0 for v in cpu_comp_ms],
        width,
        label="CPU Compaction P50 (subcomp_1)",
        color="#4C72B0",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        [v if v is not None else 0 for v in gpu_pipe_ms],
        width,
        yerr=gpu_pipe_std,
        capsize=4,
        label="GPU Pipeline Median",
        color="#DD8452",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{vs}B" for vs in sizes])
    ax.set_xlabel("Value Size")
    ax.set_ylabel("Compaction Time (ms)")
    ax.set_title("CPU vs GPU: Compaction Latency by Value Size")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "cpu_vs_gpu_compaction_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")

    # ── Figure 2: compaction read throughput comparison ──────────────────────
    cpu_bw = [
        (cpu_data[vs]["read_throughput_mb"] if cpu_data.get(vs) else None)
        for vs in sizes
    ]
    gpu_read_bw = [
        (gpu_data[vs]["read_throughput_mb"] if gpu_data.get(vs) else None)
        for vs in sizes
    ]
    gpu_write_bw = [
        (gpu_data[vs]["write_throughput_mb"] if gpu_data.get(vs) else None)
        for vs in sizes
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.25
    ax.bar(
        [i - w for i in x],
        [v if v is not None else 0 for v in cpu_bw],
        w,
        label="CPU Read BW (estimated)",
        color="#4C72B0",
    )
    ax.bar(
        [i for i in x],
        [v if v is not None else 0 for v in gpu_read_bw],
        w,
        label="GPU Read BW",
        color="#DD8452",
    )
    ax.bar(
        [i + w for i in x],
        [v if v is not None else 0 for v in gpu_write_bw],
        w,
        label="GPU Write BW",
        color="#55A868",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{vs}B" for vs in sizes])
    ax.set_xlabel("Value Size")
    ax.set_ylabel("I/O Bandwidth (MB/s)")
    ax.set_title("CPU vs GPU: Compaction I/O Bandwidth by Value Size")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "cpu_vs_gpu_compaction_bandwidth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")

    # ── Figure 3: GPU pipeline breakdown (stacked) ───────────────────────────
    gpu_sizes_avail = [vs for vs in sizes if vs in gpu_data]
    if gpu_sizes_avail:
        read_ms = []
        merge_ms = []
        write_ms = []
        other_ms = []
        for vs in gpu_sizes_avail:
            g = gpu_data[vs]
            df = g["df"]
            if len(df) > 1:
                df = df[df["rep"] > 0]

            def med(c):
                return df[c].median() / 1000 if c in df.columns else 0.0

            in_col = "input_file_reads_us" if "input_file_reads_us" in df.columns else "input_draw_us"
            out_col = "output_persist_to_disk_us" if "output_persist_to_disk_us" in df.columns else "output_persist_us"
            merge_col = "dummy_compaction_us" if "dummy_compaction_us" in df.columns else None

            r = med(in_col) if in_col in df.columns else 0
            w = med(out_col) if out_col in df.columns else 0
            mg = med(merge_col) if merge_col and merge_col in df.columns else 0
            total = g["pipeline_median_us"] / 1000 if g["pipeline_median_us"] else 0
            oth = max(0, total - r - w - mg)

            read_ms.append(r)
            merge_ms.append(mg)
            write_ms.append(w)
            other_ms.append(oth)

        x2 = range(len(gpu_sizes_avail))
        fig, ax = plt.subplots(figsize=(7, 5))
        bottoms = [0] * len(gpu_sizes_avail)
        for vals, label, color in [
            (read_ms, "SST Read", "#4C72B0"),
            (merge_ms, "GPU Merge Kernel", "#DD8452"),
            (write_ms, "SST Write", "#55A868"),
            (other_ms, "Other / Overhead", "#C44E52"),
        ]:
            ax.bar(x2, vals, bottom=bottoms, label=label, color=color)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(list(x2))
        ax.set_xticklabels([f"{vs}B" for vs in gpu_sizes_avail])
        ax.set_xlabel("Value Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title("GPU Compaction Pipeline Breakdown")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, "gpu_pipeline_breakdown.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {path}")

    # ── Figure 4: speedup bar chart ──────────────────────────────────────────
    speedups = []
    sp_sizes = []
    for vs in sizes:
        cpu = cpu_data.get(vs)
        gpu = gpu_data.get(vs)
        if cpu and gpu and cpu.get("comp_p50_us") and gpu.get("pipeline_median_us"):
            speedups.append(cpu["comp_p50_us"] / gpu["pipeline_median_us"])
            sp_sizes.append(vs)

    if speedups:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#4C72B0" if s >= 1 else "#C44E52" for s in speedups]
        ax.bar([f"{vs}B" for vs in sp_sizes], speedups, color=colors)
        ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8, label="1× (break-even)")
        ax.set_xlabel("Value Size")
        ax.set_ylabel("Speedup (CPU comp P50 / GPU pipeline)")
        ax.set_title("GPU Speedup over CPU (compaction phase)")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, "gpu_speedup.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    # Resolve defaults relative to repository root.
    # Script lives in benchmarks/gpu/python/, so go up three levels.
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent

    default_cpu = str(repo_root / "benchmarks" / "cpu" / "results")
    default_gpu = str(repo_root / "benchmarks" / "gpu" / "results" / "fillrandom_gpu")
    default_out = str(repo_root / "benchmarks" / "gpu" / "results" / "plots")

    parser = argparse.ArgumentParser(
        description="Compare CPU vs GPU fillrandom compaction performance."
    )
    parser.add_argument("--cpu-results", default=default_cpu, help="Path to cpu results base (default: %(default)s)")
    parser.add_argument("--gpu-results", default=default_gpu, help="Path to GPU fillrandom_gpu results base (default: %(default)s)")
    parser.add_argument("--cpu-run-id", default=None, help="CPU run ID (e.g. 0309_1637); omit to use latest")
    parser.add_argument("--gpu-run-id", default=None, help="GPU run ID (e.g. 0309_1829); omit to use latest")
    parser.add_argument("--plot", action="store_true", help="Save comparison plots")
    parser.add_argument("--out-dir", default=default_out, help="Output directory for plots (default: %(default)s)")
    args = parser.parse_args()

    cpu_data = {}
    gpu_data = {}

    for vs in VALUE_SIZES:
        cpu = load_cpu_run(args.cpu_results, vs, args.cpu_run_id)
        if cpu:
            cpu_data[vs] = cpu
        else:
            print(f"warning: no CPU result for value_size={vs}", file=sys.stderr)

        gpu = load_gpu_run(args.gpu_results, vs, args.gpu_run_id)
        if gpu:
            gpu_data[vs] = gpu
        else:
            print(f"warning: no GPU result for value_size={vs}", file=sys.stderr)

    if not cpu_data and not gpu_data:
        print("error: no data found — check --cpu-results and --gpu-results paths.", file=sys.stderr)
        sys.exit(1)

    print_comparison_table(cpu_data, gpu_data)

    if args.plot:
        plot_comparison(cpu_data, gpu_data, args.out_dir)
        print(f"\nAll plots saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
