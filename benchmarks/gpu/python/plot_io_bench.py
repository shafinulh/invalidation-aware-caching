#!/usr/bin/env python3
"""
plot_io_bench.py — Parse and visualise GPU vs CPU IO benchmark results.

Reads CSV files produced by gpu_io_bench and generates:
  1. A summary statistics table (mean ± std) per path/size.
  2. A grouped bar chart comparing CPU vs GPU read/write/total times.
  3. An overhead ratio table (GPU / CPU).

Usage:
  python3 plot_io_bench.py <RUN_DIR> [--plot] [--output-dir DIR]

  RUN_DIR should contain io_bench_*mb.csv files produced by run_io_bench.sh.

Examples:
  python3 plot_io_bench.py /tmp/bench_results/gpu/io_bench/io-sweep --plot
  python3 plot_io_bench.py /tmp/bench_results/gpu/io_bench/io-8mb
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

# ── Optional plotting ──
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Optional: nicer terminal tables ──
try:
    from tabulate import tabulate

    HAS_TAB = True
except ImportError:
    HAS_TAB = False


# =====================================================================
# 1.  Load data
# =====================================================================

def load_run_dir(run_dir: str) -> pd.DataFrame:
    """Load all io_bench_*mb.csv files from a run directory."""
    csvs = sorted(glob.glob(os.path.join(run_dir, "io_bench_*mb.csv")))
    if not csvs:
        print(f"error: no io_bench_*mb.csv files in {run_dir}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# =====================================================================
# 2.  Summary statistics
# =====================================================================

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for read/write/total per (path, l0_size_mb)."""
    grouped = df.groupby(["path", "l0_size_mb"]).agg(
        read_mean=("read_us", "mean"),
        read_std=("read_us", "std"),
        write_mean=("write_us", "mean"),
        write_std=("write_us", "std"),
        total_mean=("total_us", "mean"),
        total_std=("total_us", "std"),
        n=("rep", "count"),
    ).reset_index()
    return grouped


def compute_overhead(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute GPU/CPU overhead ratios."""
    cpu = summary[summary["path"] == "cpu"].set_index("l0_size_mb")
    gpu = summary[summary["path"] == "gpu"].set_index("l0_size_mb")

    if cpu.empty or gpu.empty:
        return pd.DataFrame()

    overhead = pd.DataFrame({
        "l0_size_mb": gpu.index,
        "read_overhead": (gpu["read_mean"] / cpu["read_mean"]).values,
        "write_overhead": (gpu["write_mean"] / cpu["write_mean"]).values,
        "total_overhead": (gpu["total_mean"] / cpu["total_mean"]).values,
    })
    return overhead


# =====================================================================
# 3.  Printing
# =====================================================================

def print_summary(summary: pd.DataFrame) -> None:
    """Pretty-print summary statistics."""
    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Path": r["path"].upper(),
            "L0 Size (MB)": f"{r['l0_size_mb']:.0f}",
            "Read (μs)": f"{r['read_mean']:.1f} ± {r['read_std']:.1f}",
            "Write (μs)": f"{r['write_mean']:.1f} ± {r['write_std']:.1f}",
            "Total (μs)": f"{r['total_mean']:.1f} ± {r['total_std']:.1f}",
            "N": int(r["n"]),
        })

    if HAS_TAB:
        print(tabulate(rows, headers="keys", tablefmt="github"))
    else:
        print(pd.DataFrame(rows).to_string(index=False))


def print_overhead(overhead: pd.DataFrame) -> None:
    """Pretty-print overhead ratios."""
    if overhead.empty:
        return
    rows = []
    for _, r in overhead.iterrows():
        rows.append({
            "L0 Size (MB)": f"{r['l0_size_mb']:.0f}",
            "Read Overhead": f"{r['read_overhead']:.2f}×",
            "Write Overhead": f"{r['write_overhead']:.2f}×",
            "Total Overhead": f"{r['total_overhead']:.2f}×",
        })

    print("\n── GPU / CPU Overhead Ratios ──")
    if HAS_TAB:
        print(tabulate(rows, headers="keys", tablefmt="github"))
    else:
        print(pd.DataFrame(rows).to_string(index=False))


# =====================================================================
# 4.  Plotting
# =====================================================================

def plot_comparison(df: pd.DataFrame, summary: pd.DataFrame,
                    output_dir: str) -> None:
    """Generate grouped bar charts for CPU vs GPU IO times."""
    if not HAS_MPL:
        print("warning: matplotlib not available; skipping plots.", file=sys.stderr)
        return

    sizes = sorted(df["l0_size_mb"].unique())

    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5),
                             squeeze=False)
    axes = axes.flatten()

    for ax, size_mb in zip(axes, sizes):
        sub = summary[summary["l0_size_mb"] == size_mb]
        cpu_row = sub[sub["path"] == "cpu"].iloc[0] if not sub[sub["path"] == "cpu"].empty else None
        gpu_row = sub[sub["path"] == "gpu"].iloc[0] if not sub[sub["path"] == "gpu"].empty else None

        labels = ["Read", "Write", "Total"]
        cpu_vals = [cpu_row["read_mean"], cpu_row["write_mean"], cpu_row["total_mean"]] if cpu_row is not None else [0, 0, 0]
        gpu_vals = [gpu_row["read_mean"], gpu_row["write_mean"], gpu_row["total_mean"]] if gpu_row is not None else [0, 0, 0]
        cpu_errs = [cpu_row["read_std"], cpu_row["write_std"], cpu_row["total_std"]] if cpu_row is not None else [0, 0, 0]
        gpu_errs = [gpu_row["read_std"], gpu_row["write_std"], gpu_row["total_std"]] if gpu_row is not None else [0, 0, 0]

        import numpy as np
        x = np.arange(len(labels))
        w = 0.35

        bars_cpu = ax.bar(x - w / 2, cpu_vals, w, yerr=cpu_errs, label="CPU (direct IO)",
                          color="#4C72B0", capsize=4)
        bars_gpu = ax.bar(x + w / 2, gpu_vals, w, yerr=gpu_errs, label="GPU (cuFile bounce)",
                          color="#DD8452", capsize=4)

        ax.set_xlabel("IO Phase")
        ax.set_ylabel("Time (μs)")
        ax.set_title(f"Compaction IO — {int(size_mb)} MB L0 files\n"
                      f"(read {int(df['num_l0_read'].iloc[0])}×, write {int(df['num_l1_write'].iloc[0])}×)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # Annotate bars with values.
        for bar_group in [bars_cpu, bars_gpu]:
            for bar in bar_group:
                height = bar.get_height()
                ax.annotate(f"{height:,.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "io_bench_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")

    # ── Also plot a box plot of raw data ──
    fig2, axes2 = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5),
                                squeeze=False)
    axes2 = axes2.flatten()

    for ax, size_mb in zip(axes2, sizes):
        sub = df[df["l0_size_mb"] == size_mb]
        # Melt to long form for total_us by path.
        cpu_data = sub[sub["path"] == "cpu"]["total_us"]
        gpu_data = sub[sub["path"] == "gpu"]["total_us"]

        bp = ax.boxplot([cpu_data, gpu_data], tick_labels=["CPU", "GPU"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_ylabel("Total IO Time (μs)")
        ax.set_title(f"Total Compaction IO — {int(size_mb)} MB L0 files")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "io_bench_boxplot.png")
    plt.savefig(boxplot_path, dpi=150)
    plt.close(fig2)
    print(f"Box plot saved: {boxplot_path}")


# =====================================================================
# 5.  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse and visualise GPU vs CPU IO benchmark results."
    )
    parser.add_argument("run_dir", help="Path to run directory with CSV files.")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plots (requires matplotlib).")
    parser.add_argument("--output-dir",
                        help="Directory for plots/exports (default: run_dir).")
    parser.add_argument("--csv", help="Export summary to CSV file.")
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir

    if not os.path.isdir(run_dir):
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load data.
    df = load_run_dir(run_dir)
    print(f"Loaded {len(df)} rows from {run_dir}\n")

    # Summary.
    summary = compute_summary(df)
    print("── Summary Statistics ──")
    print_summary(summary)

    # Overhead.
    overhead = compute_overhead(summary)
    print_overhead(overhead)

    # Export.
    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"\nSummary exported to {args.csv}")

    # Also save summary CSV alongside the run data.
    summary_csv = os.path.join(output_dir, "io_bench_summary.csv")
    summary.to_csv(summary_csv, index=False)

    overhead_csv = os.path.join(output_dir, "io_bench_overhead.csv")
    if not overhead.empty:
        overhead.to_csv(overhead_csv, index=False)

    # Plot.
    if args.plot:
        plot_comparison(df, summary, output_dir)


if __name__ == "__main__":
    main()
