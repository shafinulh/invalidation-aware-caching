#!/usr/bin/env python3
"""
plot_gpu_compaction_hook_bench.py - Summarise and visualise the RocksDB dummy
GPU compaction hook benchmark.

Usage:
  python3 plot_gpu_compaction_hook_bench.py <RUN_DIR> [--plot]
"""

import argparse
import os
import sys

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


PHASE_LABELS = {
    "gpu_hook_e2e_us": "GPU Hook E2E",
    "gpu_pipeline_total_us": "GPU Pipeline Total",
    "input_file_reads_us": "Input File Reads",
    "input_draw_us": "Input File Reads",
    "dummy_compaction_us": "Dummy Compaction",
    "output_persist_to_disk_us": "Output Persist To Disk",
    "output_persist_us": "Output Persist To Disk",
    "post_write_completion_us": "Post-Write Completion",
    "stage_ground_truth_us": "Ground Truth Stage",
    "compact_range_us": "CompactRange",
    "verify_us": "Verify",
    "wait_total_us": "Wait Total",
    "gpu_total_us": "GPU Replay",
}


def phase_col(df: pd.DataFrame, preferred: str, fallback: str | None = None) -> str | None:
    if preferred in df.columns:
        return preferred
    if fallback and fallback in df.columns:
        return fallback
    return None


def load_run_dir(run_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(run_dir, "gpu_compaction_hook.csv")
    if not os.path.isfile(csv_path):
        print(f"error: missing {csv_path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(csv_path)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    input_reads_col = phase_col(df, "input_file_reads_us", "input_draw_us")
    output_persist_col = phase_col(
        df, "output_persist_to_disk_us", "output_persist_us"
    )

    if input_reads_col and output_persist_col and "post_write_completion_us" in df.columns:
        metrics = [
            "gpu_hook_e2e_us",
            "gpu_pipeline_total_us",
            input_reads_col,
            "dummy_compaction_us",
            output_persist_col,
            "post_write_completion_us",
            "stage_ground_truth_us",
            "verify_us",
        ]
    elif "input_stage_us" in df.columns and "output_replay_us" in df.columns:
        metrics = [
            "compact_range_us",
            "stage_ground_truth_us",
            "input_stage_us",
            "output_replay_us",
            "verify_us",
            "wait_total_us",
        ]
    else:
        metrics = [
            "compact_range_us",
            "stage_ground_truth_us",
            "gpu_total_us",
            "verify_us",
            "wait_total_us",
        ]

    summary = pd.DataFrame(
        {
            "metric": metrics,
            "mean_us": [df[m].mean() for m in metrics],
            "std_us": [df[m].std() for m in metrics],
        }
    )
    summary["throughput_mb_s"] = None
    if "input_bytes" in df and input_reads_col:
        throughput = (
            df["input_bytes"] / (1024 * 1024) / (df[input_reads_col] / 1_000_000)
        )
        summary.loc[summary["metric"] == input_reads_col, "throughput_mb_s"] = (
            throughput.mean()
        )
    if "output_bytes" in df and output_persist_col:
        throughput = (
            df["output_bytes"] / (1024 * 1024) / (df[output_persist_col] / 1_000_000)
        )
        summary.loc[summary["metric"] == output_persist_col, "throughput_mb_s"] = (
            throughput.mean()
        )
    elif "output_bytes" in df and "output_replay_us" in df:
        throughput = (
            df["output_bytes"] / (1024 * 1024) / (df["output_replay_us"] / 1_000_000)
        )
        summary.loc[summary["metric"] == "output_replay_us", "throughput_mb_s"] = (
            throughput.mean()
        )
    elif "bytes_replayed" in df and "gpu_total_us" in df:
        throughput = (
            df["bytes_replayed"] / (1024 * 1024) / (df["gpu_total_us"] / 1_000_000)
        )
        summary.loc[summary["metric"] == "gpu_total_us", "throughput_mb_s"] = (
            throughput.mean()
        )
    if "input_bytes" in df and "input_stage_us" in df:
        throughput = (
            df["input_bytes"] / (1024 * 1024) / (df["input_stage_us"] / 1_000_000)
        )
        summary.loc[summary["metric"] == "input_stage_us", "throughput_mb_s"] = (
            throughput.mean()
        )
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        metric = row["metric"]
        label = PHASE_LABELS.get(metric, metric.replace("_us", ""))
        text = f"{row['mean_us']:.1f} ± {0.0 if pd.isna(row['std_us']) else row['std_us']:.1f} us"
        if pd.notna(row["throughput_mb_s"]):
            text += f" ({row['throughput_mb_s']:.1f} MB/s)"
        rows.append((label, text))

    print("Summary")
    for label, text in rows:
        print(f"  {label:>18}: {text}")


def plot_summary(df: pd.DataFrame, output_dir: str) -> None:
    if not HAS_MPL:
        print("warning: matplotlib not available; skipping plots.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    input_reads_col = phase_col(df, "input_file_reads_us", "input_draw_us")
    output_persist_col = phase_col(
        df, "output_persist_to_disk_us", "output_persist_us"
    )

    if input_reads_col and output_persist_col and "post_write_completion_us" in df.columns:
        metrics = [
            "gpu_hook_e2e_us",
            input_reads_col,
            "dummy_compaction_us",
            output_persist_col,
            "post_write_completion_us",
        ]
        labels = [PHASE_LABELS[m] for m in metrics]
        colors = ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"]
        title = "RocksDB Hook E2E Breakdown"
    elif "input_stage_us" in df.columns and "output_replay_us" in df.columns:
        metrics = [
            "compact_range_us",
            "stage_ground_truth_us",
            "input_stage_us",
            "output_replay_us",
            "verify_us",
        ]
        labels = ["Compact", "Ground Truth", "Input Stage", "Output Replay", "Verify"]
        colors = ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"]
        title = "RocksDB Dummy GPU Compaction Hook Benchmark"
    else:
        metrics = [
            "compact_range_us",
            "stage_ground_truth_us",
            "gpu_total_us",
            "verify_us",
        ]
        labels = ["Compact", "Ground Truth", "GPU Replay", "Verify"]
        colors = ["#355070", "#6D597A", "#B56576", "#E56B6F"]
        title = "RocksDB Dummy GPU Compaction Hook Benchmark"

    means = [df[m].mean() for m in metrics]
    errs = [0.0 if pd.isna(df[m].std()) else df[m].std() for m in metrics]

    ax.bar(labels, means, yerr=errs, color=colors, capsize=4)
    ax.set_ylabel("Time (us)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "gpu_compaction_hook_summary.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if input_reads_col and output_persist_col and "post_write_completion_us" in df.columns:
        ax.plot(df["rep"], df["gpu_hook_e2e_us"], marker="o", label="GPU Hook E2E")
        ax.plot(
            df["rep"],
            df["gpu_pipeline_total_us"],
            marker="o",
            label="GPU Pipeline Total",
        )
        ax.plot(df["rep"], df[input_reads_col], marker="o", label="Input File Reads")
        ax.plot(df["rep"], df["dummy_compaction_us"], marker="o", label="Dummy Compaction")
        ax.plot(
            df["rep"],
            df[output_persist_col],
            marker="o",
            label="Output Persist To Disk",
        )
        ax.plot(
            df["rep"],
            df["post_write_completion_us"],
            marker="o",
            label="Post-Write Completion",
        )
        ax.plot(
            df["rep"],
            df["stage_ground_truth_us"],
            marker="o",
            linestyle="--",
            label="Ground Truth (Test-Only)",
        )
    else:
        ax.plot(df["rep"], df["compact_range_us"], marker="o", label="CompactRange")
    if "input_stage_us" in df.columns and not input_reads_col:
        ax.plot(df["rep"], df["input_stage_us"], marker="o", label="Input Stage")
    if "output_replay_us" in df.columns and not output_persist_col:
        ax.plot(df["rep"], df["output_replay_us"], marker="o", label="Output Replay")
    elif "gpu_total_us" in df.columns and not input_reads_col:
        ax.plot(df["rep"], df["gpu_total_us"], marker="o", label="GPU Replay")
    ax.plot(df["rep"], df["verify_us"], marker="o", label="Verify")
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Time (us)")
    if input_reads_col and output_persist_col and "post_write_completion_us" in df.columns:
        ax.set_title("Per-Rep Phase Timing")
    else:
        ax.set_title("Per-Rep Timing")
    ax.legend()
    fig.tight_layout()

    line_path = os.path.join(output_dir, "gpu_compaction_hook_per_rep.png")
    fig.savefig(line_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {line_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise and visualise the RocksDB GPU hook benchmark."
    )
    parser.add_argument("run_dir", help="Path to the benchmark run directory.")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    df = load_run_dir(run_dir)
    print(f"Loaded {len(df)} rows from {run_dir}\n")

    summary = compute_summary(df)
    print_summary(summary)

    summary_csv = os.path.join(run_dir, "gpu_compaction_hook_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV: {summary_csv}")

    if args.plot:
        plot_summary(df, run_dir)


if __name__ == "__main__":
    main()
