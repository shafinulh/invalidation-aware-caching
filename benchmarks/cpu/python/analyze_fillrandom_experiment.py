#!/usr/bin/env python3
"""Analyze one fillrandom experiment directory and generate figures."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "cpu_compaction" / "plotting"))
from parse_compaction_profile import events_to_dataframe, parse_compaction_events


STALL_RE = re.compile(r"rocksdb\.db\.write\.stall .* COUNT : (\d+) SUM : (\d+)")


def run_label(subcomp: int, bgcomp: int) -> str:
    return f"sub={subcomp}, bg={bgcomp}"


def plot_label(subcomp: int, bgcomp: int) -> str:
    return f"({subcomp},{bgcomp})"


def compaction_pattern(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")

    def classify(row: pd.Series) -> str:
        levels = sorted(
            {
                int(match.group(1))
                for match in re.finditer(r"@L(\d+)", str(row.get("input_summary", "")))
            }
        )
        if not levels:
            return f"unknown->L{int(row.get('output_level', -1))}"
        input_levels = "+".join(f"L{level}" for level in levels)
        return f"{input_levels}->L{int(row.get('output_level', -1))}"

    return df.apply(classify, axis=1)


def parse_run(run_dir: Path) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    subcomp = int(run_dir.parent.name.split("_", 1)[1])
    bgcomp = int(run_dir.name.split("_", 1)[1])
    label = run_label(subcomp, bgcomp)
    short_label = plot_label(subcomp, bgcomp)

    report_df = pd.read_csv(run_dir / "report.csv")
    host_summary = {}
    summary_path = run_dir / "host_metrics" / "summary.json"
    if summary_path.exists():
        host_summary = json.loads(summary_path.read_text())

    role_df = pd.read_csv(run_dir / "host_metrics" / "thread_role_cpu.csv")
    if role_df.empty:
        role_pivot = pd.DataFrame(columns=["secs_elapsed"])
    else:
        role_pivot = (
            role_df.pivot_table(
                index="secs_elapsed",
                columns="thread_role",
                values="cpu_pct",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
    role_columns = [c for c in role_pivot.columns if c != "secs_elapsed"]
    if role_columns:
        bg_columns = [c for c in role_columns if c != "foreground"]
        role_pivot["background_cpu_pct"] = role_pivot[bg_columns].sum(axis=1)
        role_pivot["foreground_cpu_pct"] = role_pivot.get("foreground", 0)
    else:
        role_pivot["background_cpu_pct"] = 0
        role_pivot["foreground_cpu_pct"] = 0

    device_df = pd.read_csv(run_dir / "host_metrics" / "device_io.csv")
    process_df = pd.read_csv(run_dir / "host_metrics" / "process_cpu.csv")
    system_df = pd.read_csv(run_dir / "host_metrics" / "system_cpu.csv")

    db_bench_log = (run_dir / "db_bench.log").read_text(errors="replace")
    stall_match = STALL_RE.search(db_bench_log)
    stall_count = int(stall_match.group(1)) if stall_match else 0
    stall_sum_us = int(stall_match.group(2)) if stall_match else 0

    compaction_log = run_dir / "metadata" / "rocksdb_LOG_after_fillrandom.txt"
    events, start_info = parse_compaction_events(str(compaction_log))
    comp_df = events_to_dataframe(events, start_info)
    if "input_summary" in comp_df.columns:
        l0_df = comp_df[comp_df["input_summary"].str.contains("L0", na=False)]
    else:
        l0_df = comp_df.iloc[0:0]
    if not comp_df.empty:
        comp_df = comp_df.copy()
        comp_df["label"] = label
        comp_df["plot_label"] = short_label
        comp_df["pattern"] = compaction_pattern(comp_df)
        comp_df["wall_ms"] = comp_df["wall_us"] / 1000.0
        comp_df["cpu_ms"] = comp_df["cpu_us"] / 1000.0
        comp_df["read_io_ms"] = comp_df["read_io_us"] / 1000.0
        comp_df["write_io_ms"] = comp_df["write_io_us"] / 1000.0
    else:
        comp_df = pd.DataFrame(
            columns=[
                "label",
                "plot_label",
                "pattern",
                "input_summary",
                "output_level",
                "num_output_files",
                "input_data_size_bytes",
                "total_output_size_bytes",
                "num_subcompactions",
                "wall_ms",
                "cpu_ms",
                "read_io_ms",
                "write_io_ms",
                "compaction_reason",
            ]
        )

    avg_role_cpu = host_summary.get("avg_thread_role_cpu_pct", {})
    background_cpu_avg = sum(
        value for key, value in avg_role_cpu.items() if key != "foreground"
    )

    metrics = {
        "run_dir": str(run_dir),
        "subcomp": subcomp,
        "bgcomp": bgcomp,
        "label": label,
        "plot_label": short_label,
        "duration_s": int(report_df["secs_elapsed"].max()),
        "avg_qps": float(report_df["interval_qps"].mean()),
        "max_qps": int(report_df["interval_qps"].max()),
        "min_qps": int(report_df["interval_qps"].min()),
        "stall_count": stall_count,
        "stall_s": stall_sum_us / 1e6,
        "stall_pct": (stall_sum_us / 1e6) / len(report_df) * 100 if len(report_df) else 0,
        "avg_process_cpu_pct": float(host_summary.get("avg_process_cpu_pct", 0)),
        "avg_system_cpu_busy_pct": float(host_summary.get("avg_system_cpu_busy_pct", 0)),
        "avg_system_cpu_iowait_pct": float(host_summary.get("avg_system_cpu_iowait_pct", 0)),
        "avg_device_util_pct": float(host_summary.get("avg_device_util_pct", 0)),
        "avg_device_rkib_per_sec": float(host_summary.get("avg_device_rkib_per_sec", 0)),
        "avg_device_wkib_per_sec": float(host_summary.get("avg_device_wkib_per_sec", 0)),
        "avg_foreground_cpu_pct": float(avg_role_cpu.get("foreground", 0)),
        "avg_background_cpu_pct": float(background_cpu_avg),
        "avg_compaction_cpu_pct": float(avg_role_cpu.get("rocksdb_compaction", 0)),
        "avg_flush_cpu_pct": float(avg_role_cpu.get("rocksdb_flush", 0)),
        "compaction_count": int(len(comp_df)),
        "compaction_avg_ms": float(comp_df["wall_us"].mean() / 1000) if not comp_df.empty else 0,
        "compaction_p95_ms": float(comp_df["wall_us"].quantile(0.95) / 1000) if not comp_df.empty else 0,
        "compaction_max_ms": float(comp_df["wall_us"].max() / 1000) if not comp_df.empty else 0,
        "compaction_total_wall_s": float(comp_df["wall_us"].sum() / 1e6) if not comp_df.empty else 0,
        "compaction_total_cpu_s": float(comp_df["cpu_us"].sum() / 1e6) if not comp_df.empty else 0,
        "compaction_total_read_io_s": float(comp_df["read_io_us"].sum() / 1e6) if not comp_df.empty else 0,
        "compaction_total_write_io_s": float(comp_df["write_io_us"].sum() / 1e6) if not comp_df.empty else 0,
        "l0_compaction_count": int(len(l0_df)),
        "l0_p50_ms": float(l0_df["wall_us"].quantile(0.50) / 1000) if not l0_df.empty else 0,
        "l0_avg_ms": float(l0_df["wall_us"].mean() / 1000) if not l0_df.empty else 0,
        "l0_p95_ms": float(l0_df["wall_us"].quantile(0.95) / 1000) if not l0_df.empty else 0,
        "l0_p99_ms": float(l0_df["wall_us"].quantile(0.99) / 1000) if not l0_df.empty else 0,
        "l0_max_ms": float(l0_df["wall_us"].max() / 1000) if not l0_df.empty else 0,
        "actual_subcompactions_avg": float(comp_df["num_subcompactions"].mean()) if not comp_df.empty else 0,
        "actual_subcompactions_max": int(comp_df["num_subcompactions"].max()) if not comp_df.empty else 0,
    }

    report_df = report_df.copy()
    report_df["label"] = label
    report_df["plot_label"] = short_label
    device_df = device_df.copy()
    device_df["label"] = label
    device_df["plot_label"] = short_label
    process_df = process_df.copy()
    process_df["label"] = label
    process_df["plot_label"] = short_label
    role_pivot = role_pivot.copy()
    role_pivot["label"] = label
    role_pivot["plot_label"] = short_label
    system_df = system_df.copy()
    system_df["label"] = label
    system_df["plot_label"] = short_label

    return metrics, {
        "throughput": report_df,
        "device": device_df,
        "process": process_df,
        "roles": role_pivot,
        "system": system_df,
        "compactions": comp_df,
    }


def save_throughput_plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, grp in df.groupby("plot_label"):
        ax.plot(grp["secs_elapsed"], grp["interval_qps"], label=label, linewidth=2)
    ax.set_title("Foreground Write Throughput Over Time")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Interval QPS")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_device_plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, grp in df.groupby("plot_label"):
        axes[0].plot(grp["secs_elapsed"], grp["util_pct"], label=label, linewidth=2)
        axes[1].plot(grp["secs_elapsed"], grp["wkib_per_sec"] / 1024.0, label=label, linewidth=2)
    axes[0].set_title("Device Utilization Over Time")
    axes[0].set_ylabel("Util %")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Device Write Throughput Over Time")
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("MiB/s")
    axes[1].grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_cpu_plot(process_df: pd.DataFrame, role_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for label, grp in process_df.groupby("plot_label"):
        axes[0].plot(grp["secs_elapsed"], grp["cpu_pct"], label=label, linewidth=2)
    for label, grp in role_df.groupby("plot_label"):
        axes[1].plot(grp["secs_elapsed"], grp["foreground_cpu_pct"], label=label, linewidth=2)
        axes[2].plot(grp["secs_elapsed"], grp["background_cpu_pct"], label=label, linewidth=2)
    axes[0].set_title("db_bench CPU Over Time")
    axes[0].set_ylabel("CPU %")
    axes[1].set_title("Foreground Writer CPU Over Time")
    axes[1].set_ylabel("CPU %")
    axes[2].set_title("Background CPU Over Time (Compaction + Flush)")
    axes[2].set_ylabel("CPU %")
    axes[2].set_xlabel("Seconds")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_summary_plot(df: pd.DataFrame, out: Path) -> None:
    labels = df["plot_label"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].bar(labels, df["avg_qps"] / 1e6)
    axes[0, 0].set_title("Average Foreground Throughput")
    axes[0, 0].set_ylabel("M ops/s")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].bar(labels, df["stall_s"])
    axes[0, 1].set_title("Total Write Stall Time")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].tick_params(axis="x", rotation=20)

    x = range(len(labels))
    width = 0.24
    axes[1, 0].bar([i - width for i in x], df["l0_p50_ms"], width=width, label="p50")
    axes[1, 0].bar(list(x), df["l0_p95_ms"], width=width, label="p95")
    axes[1, 0].bar([i + width for i in x], df["l0_p99_ms"], width=width, label="p99")
    axes[1, 0].set_title("L0-input Compaction Latency")
    axes[1, 0].set_ylabel("Milliseconds")
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(labels, rotation=20)
    axes[1, 0].legend()

    axes[0, 2].axis("off")

    width = 0.38
    axes[1, 1].bar(
        [i - width / 2 for i in x],
        df["compaction_total_wall_s"],
        width=width,
        label="wall",
    )
    axes[1, 1].bar(
        [i + width / 2 for i in x],
        df["compaction_total_cpu_s"],
        width=width,
        label="cpu",
    )
    axes[1, 1].set_title("Total Compaction Time")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].legend()

    axes[1, 2].bar(labels, df["avg_device_util_pct"])
    axes[1, 2].set_title("Average Device Utilization")
    axes[1, 2].set_ylabel("Util %")
    axes[1, 2].tick_params(axis="x", rotation=20)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.3)
    fig.text(0.99, 0.01, "x-axis label = (sub,bg)", ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_compaction_breakdown_plot(df: pd.DataFrame, out: Path) -> None:
    labels = df["plot_label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(labels))
    width = 0.38
    axes[0].bar([i - width / 2 for i in x], df["compaction_total_wall_s"], width=width, label="wall")
    axes[0].bar([i + width / 2 for i in x], df["compaction_total_cpu_s"], width=width, label="cpu")
    axes[0].set_title("Total Compaction Time")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, df["compaction_total_read_io_s"], label="read_io")
    axes[1].bar(
        labels,
        df["compaction_total_write_io_s"],
        bottom=df["compaction_total_read_io_s"],
        label="write_io",
    )
    axes[1].bar(
        labels,
        df["compaction_total_cpu_s"],
        bottom=df["compaction_total_read_io_s"] + df["compaction_total_write_io_s"],
        label="cpu",
    )
    axes[1].set_title("Compaction Time Breakdown")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.text(0.99, 0.01, "x-axis label = (sub,bg)", ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_compaction_pattern_plot(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return

    counts = (
        df.groupby(["plot_label", "pattern"])
        .size()
        .reset_index(name="count")
        .sort_values(["plot_label", "count"], ascending=[True, False])
    )
    top_patterns = counts.groupby("pattern")["count"].sum().nlargest(6).index.tolist()
    counts["pattern"] = counts["pattern"].where(counts["pattern"].isin(top_patterns), "other")
    counts = counts.groupby(["plot_label", "pattern"], as_index=False)["count"].sum()
    pivot = counts.pivot(index="plot_label", columns="pattern", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = pd.Series(0, index=pivot.index)
    for pattern in pivot.columns:
        ax.bar(pivot.index, pivot[pattern], bottom=bottom, label=pattern)
        bottom += pivot[pattern]
    ax.set_title("Completed Compactions by Input/Output Pattern")
    ax.set_ylabel("Compaction Count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Pattern")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.99, 0.01, "x-axis label = (sub,bg)", ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_compaction_pattern_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No compaction events found.\n"

    lines = []
    for label, grp in df.groupby("label"):
        lines.append(label)
        lines.append("-" * len(label))
        by_pattern = (
            grp.groupby("pattern")
            .agg(
                count=("pattern", "size"),
                avg_wall_ms=("wall_ms", "mean"),
                p95_wall_ms=("wall_ms", lambda s: s.quantile(0.95)),
                avg_cpu_ms=("cpu_ms", "mean"),
                avg_input_mb=("input_data_size_bytes", lambda s: s.mean() / (1024 * 1024)),
                avg_output_mb=("total_output_size_bytes", lambda s: s.mean() / (1024 * 1024)),
                avg_subcompactions=("num_subcompactions", "mean"),
            )
            .sort_values("count", ascending=False)
        )
        for pattern, row in by_pattern.iterrows():
            lines.append(
                f"{pattern}: count={int(row['count'])}, "
                f"wall avg/p95={row['avg_wall_ms']:.1f}/{row['p95_wall_ms']:.1f} ms, "
                f"cpu avg={row['avg_cpu_ms']:.1f} ms, "
                f"input={row['avg_input_mb']:.1f} MiB, "
                f"output={row['avg_output_mb']:.1f} MiB, "
                f"avg_subcompactions={row['avg_subcompactions']:.1f}"
            )
        lines.append("")
    return "\n".join(lines)


def build_summary_text(summary_df: pd.DataFrame) -> str:
    baseline = summary_df[(summary_df["subcomp"] == 0) & (summary_df["bgcomp"] == 1)]
    lines = []
    if baseline.empty:
        baseline_row = None
        lines.append("No sub=0/bg=1 baseline was found.")
    else:
        baseline_row = baseline.iloc[0]
        lines.append(
            "Baseline (sub=0, bg=1): "
            f"{baseline_row['avg_qps'] / 1e6:.3f} Mops/s, "
            f"{baseline_row['avg_foreground_cpu_pct']:.0f}% foreground CPU, "
            f"{baseline_row['avg_device_util_pct']:.1f}% device util."
        )

    focus_order = [(1, 1), (8, 1), (16, 1), (8, 2)]
    for subcomp, bgcomp in focus_order:
        row_df = summary_df[
            (summary_df["subcomp"] == subcomp) & (summary_df["bgcomp"] == bgcomp)
        ]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        line = (
            f"{row['label']}: {row['avg_qps'] / 1e6:.3f} Mops/s, "
            f"{row['stall_s']:.1f}s stalls ({row['stall_pct']:.1f}% of run), "
            f"L0-input compaction p50/p95/p99 {row['l0_p50_ms']:.1f}/{row['l0_p95_ms']:.1f}/{row['l0_p99_ms']:.1f} ms, "
            f"total compaction wall/cpu {row['compaction_total_wall_s']:.1f}/{row['compaction_total_cpu_s']:.1f}s, "
            f"foreground CPU {row['avg_foreground_cpu_pct']:.0f}%, "
            f"background CPU {row['avg_background_cpu_pct']:.0f}%, "
            f"device util {row['avg_device_util_pct']:.1f}%."
        )
        if baseline_row is not None:
            qps_gap = baseline_row["avg_qps"] - row["avg_qps"]
            fg_gap = baseline_row["avg_foreground_cpu_pct"] - row["avg_foreground_cpu_pct"]
            line += (
                f" Throughput remains {qps_gap / 1e6:.3f} Mops/s below the no-compaction "
                f"baseline, while foreground CPU is lower by {fg_gap:.0f} points."
            )
        lines.append(line)

    sub8 = summary_df[(summary_df["subcomp"] == 8) & (summary_df["bgcomp"] == 1)]
    sub16 = summary_df[(summary_df["subcomp"] == 16) & (summary_df["bgcomp"] == 1)]
    if not sub8.empty and not sub16.empty:
        r8 = sub8.iloc[0]
        r16 = sub16.iloc[0]
        lines.append(
            "sub=16 does not buy much over sub=8 here: "
            f"L0 p95 compaction only moves from {r8['l0_p95_ms']:.1f} ms to "
            f"{r16['l0_p95_ms']:.1f} ms, while foreground CPU falls from "
            f"{r8['avg_foreground_cpu_pct']:.0f}% to {r16['avg_foreground_cpu_pct']:.0f}% "
            f"and average throughput slips from {r8['avg_qps'] / 1e6:.3f} to "
            f"{r16['avg_qps'] / 1e6:.3f} Mops/s. The actual average number of "
            f"subcompactions stays low ({r16['actual_subcompactions_avg']:.1f}, "
            f"max {r16['actual_subcompactions_max']})."
        )
        lines.append(
            "That suggests this workload's compactions are not heavy enough to keep 16 "
            "subcompactions busy. Larger values, larger compaction inputs, or larger "
            "memtables/SST targets are the cases where 16 would be more likely to help."
        )

    sub82 = summary_df[(summary_df["subcomp"] == 8) & (summary_df["bgcomp"] == 2)]
    if baseline_row is not None and not sub82.empty:
        r82 = sub82.iloc[0]
        lines.append(
            "sub=8, bg=2 gives the best compaction-enabled result because it cuts "
            f"stall time to {r82['stall_s']:.1f}s and pushes throughput to "
            f"{r82['avg_qps'] / 1e6:.3f} Mops/s, but it is still below the no-compaction "
            f"baseline by {(baseline_row['avg_qps'] - r82['avg_qps']) / 1e6:.3f} Mops/s. "
            f"Foreground CPU drops from {baseline_row['avg_foreground_cpu_pct']:.0f}% to "
            f"{r82['avg_foreground_cpu_pct']:.0f}%, while background CPU rises to "
            f"{r82['avg_background_cpu_pct']:.0f}% and device util rises from "
            f"{baseline_row['avg_device_util_pct']:.1f}% to {r82['avg_device_util_pct']:.1f}%."
        )

    lines.append(
        "Here, 'L0-input compaction' means any compaction whose input files include L0. "
        "It does not include every compaction in the tree. It is the more relevant subset "
        "for this write-stall analysis because stalls are driven by L0 pressure."
    )

    return "\n\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    run_dirs = sorted(experiment_root.glob("subcomp_*/bgcomp_*"))
    if not run_dirs:
        raise SystemExit(f"no runs found under {experiment_root}")

    metrics_rows = []
    throughput_frames = []
    device_frames = []
    process_frames = []
    role_frames = []
    system_frames = []
    compaction_frames = []

    for run_dir in run_dirs:
        metrics, frames = parse_run(run_dir)
        metrics_rows.append(metrics)
        throughput_frames.append(frames["throughput"])
        device_frames.append(frames["device"])
        process_frames.append(frames["process"])
        role_frames.append(frames["roles"])
        system_frames.append(frames["system"])
        compaction_frames.append(frames["compactions"])

    summary_df = pd.DataFrame(metrics_rows).sort_values(["subcomp", "bgcomp"])
    baseline = summary_df[(summary_df["subcomp"] == 0) & (summary_df["bgcomp"] == 1)]
    if not baseline.empty:
        baseline_qps = baseline.iloc[0]["avg_qps"]
        baseline_fg_cpu = baseline.iloc[0]["avg_foreground_cpu_pct"]
        summary_df["qps_pct_of_baseline"] = summary_df["avg_qps"] / baseline_qps * 100
        summary_df["qps_loss_vs_baseline"] = baseline_qps - summary_df["avg_qps"]
        summary_df["foreground_cpu_loss_vs_baseline"] = (
            baseline_fg_cpu - summary_df["avg_foreground_cpu_pct"]
        )

    analysis_dir = experiment_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(analysis_dir / "summary_metrics.csv", index=False)
    pd.concat(throughput_frames, ignore_index=True).to_csv(
        analysis_dir / "throughput_timeseries.csv", index=False
    )
    compaction_df = pd.concat(compaction_frames, ignore_index=True)
    compaction_df.to_csv(analysis_dir / "compaction_events.csv", index=False)

    save_throughput_plot(pd.concat(throughput_frames, ignore_index=True), analysis_dir / "throughput_over_time.png")
    save_device_plot(pd.concat(device_frames, ignore_index=True), analysis_dir / "device_io_over_time.png")
    save_cpu_plot(
        pd.concat(process_frames, ignore_index=True),
        pd.concat(role_frames, ignore_index=True),
        analysis_dir / "cpu_over_time.png",
    )
    save_summary_plot(summary_df, analysis_dir / "summary_bars.png")
    save_compaction_breakdown_plot(summary_df, analysis_dir / "compaction_breakdown.png")
    save_compaction_pattern_plot(compaction_df, analysis_dir / "compaction_patterns.png")

    summary_text = build_summary_text(summary_df)
    (analysis_dir / "analysis_summary.txt").write_text(summary_text, encoding="utf-8")
    (analysis_dir / "compaction_patterns_summary.txt").write_text(
        build_compaction_pattern_text(compaction_df), encoding="utf-8"
    )
    print(summary_text)
    print(f"Wrote analysis to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
