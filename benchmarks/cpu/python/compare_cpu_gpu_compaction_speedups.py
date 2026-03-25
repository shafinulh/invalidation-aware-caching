#!/usr/bin/env python3
"""Compare CPU subcompaction speedups against GPComp modes for one sweep case."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}") from exc


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

CPU_SUBCOMP_COLORS = {
    2: "#dbe9f6",
    4: "#b6d3eb",
    8: "#8dbfe0",
    16: "#5f9ed1",
    32: "#2f6ba7",
    64: "#1d4f7a",
}

VALUE_RE = re.compile(r"value bytes:\s*(\d+)")
MODE_RE = re.compile(r"gpu mode:\s*([^\n]+)")
INPUT_SSTS_RE = re.compile(r"input SSTs:\s*(\d+)")
SPEEDUP_RE = re.compile(r"Speedup:\s*([\d.]+)x")
CPU_TOTAL_RE = re.compile(r"CPU total \(Wall\):\s*min=[\d.]+\s*ms\s*mean=([\d.]+)\s*\+-\s*([\d.]+)\s*ms")
GPU_TOTAL_RE = re.compile(r"GPU total \(Wall\):\s*min=[\d.]+\s*ms\s*mean=([\d.]+)\s*\+-\s*([\d.]+)\s*ms")


def parse_int_list(raw_value: str) -> list[int]:
    return [int(token) for token in raw_value.split() if token.strip()]


def parse_str_list(raw_value: str) -> list[str]:
    return [token for token in raw_value.split() if token.strip()]


def normalize_prefix(sst_size_mb: int, input_sst_count: int) -> str:
    return f"{sst_size_mb}mb-sst_{input_sst_count}sst"


def parse_gpu_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")

    value_match = VALUE_RE.search(text)
    mode_match = MODE_RE.search(text)
    input_ssts_match = INPUT_SSTS_RE.search(text)
    speedup_match = SPEEDUP_RE.search(text)
    cpu_total_match = CPU_TOTAL_RE.search(text)
    gpu_total_match = GPU_TOTAL_RE.search(text)

    if value_match is None:
        raise ValueError(f"missing value bytes in {path}")
    if mode_match is None:
        raise ValueError(f"missing gpu mode in {path}")
    if input_ssts_match is None:
        raise ValueError(f"missing input SST count in {path}")
    if speedup_match is None:
        raise ValueError(f"missing speedup in {path}")
    if cpu_total_match is None or gpu_total_match is None:
        raise ValueError(f"missing CPU/GPU timing summary in {path}")

    cpu_mean_ms = float(cpu_total_match.group(1))
    cpu_std_ms = float(cpu_total_match.group(2))
    gpu_mean_ms = float(gpu_total_match.group(1))
    gpu_std_ms = float(gpu_total_match.group(2))
    speedup_mean = cpu_mean_ms / gpu_mean_ms if gpu_mean_ms > 0 else 0.0
    speedup_std = 0.0
    if cpu_mean_ms > 0 and gpu_mean_ms > 0:
        speedup_std = speedup_mean * math.sqrt(
            (cpu_std_ms / cpu_mean_ms) ** 2 + (gpu_std_ms / gpu_mean_ms) ** 2
        )

    return {
        "value_size": int(value_match.group(1)),
        "mode": mode_match.group(1).strip(),
        "input_sst_count": int(input_ssts_match.group(1)),
        "speedup": speedup_mean,
        "speedup_std": speedup_std,
        "speedup_min": float(speedup_match.group(1)),
        "cpu_mean_ms": cpu_mean_ms,
        "cpu_std_ms": cpu_std_ms,
        "gpu_mean_ms": gpu_mean_ms,
        "gpu_std_ms": gpu_std_ms,
        "log_path": str(path),
    }


def load_cpu_rows(
    cpu_experiment_root: Path,
    sst_size_mb: int,
    input_sst_count: int,
    subcomp_threads: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = cpu_experiment_root / "analysis" / "summary_metrics.csv"
    if not summary_path.exists():
        raise SystemExit(f"missing CPU analysis summary: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise SystemExit(f"CPU analysis summary is empty: {summary_path}")

    if "requested_input_sst_count" in summary_df.columns:
        requested_counts = pd.to_numeric(
            summary_df["requested_input_sst_count"], errors="coerce"
        ).fillna(0)
    else:
        requested_counts = pd.Series([0] * len(summary_df))

    case_mask = (
        (summary_df["sst_size_mb"] == sst_size_mb)
        & (
            (requested_counts == input_sst_count)
            | (
                (requested_counts == 0)
                & (summary_df["input_data_mb"] == sst_size_mb * input_sst_count)
            )
        )
    )
    case_df = summary_df[case_mask].copy()
    if case_df.empty:
        raise SystemExit(
            "no CPU rows matched "
            f"sst={sst_size_mb}MB and input_sst_count={input_sst_count} in {summary_path}"
        )

    case_df["speedup_vs_subcomp1"] = pd.to_numeric(
        case_df["speedup_vs_subcomp1"], errors="coerce"
    )
    case_df.loc[case_df["requested_subcompactions"] == 1, "speedup_vs_subcomp1"] = 1.0

    expected_subcomps = sorted(set(subcomp_threads))
    available_subcomps = sorted(case_df["requested_subcompactions"].unique())
    if 1 not in available_subcomps:
        raise SystemExit("CPU summary is missing the required subcompaction=1 baseline")
    missing_subcomps = [sub for sub in expected_subcomps if sub not in available_subcomps]
    if missing_subcomps:
        raise SystemExit(
            "CPU summary is missing requested subcompaction points: "
            + ", ".join(str(sub) for sub in missing_subcomps)
        )

    plot_rows: list[dict[str, object]] = []
    for row in case_df.itertuples(index=False):
        subcomp = int(row.requested_subcompactions)
        if subcomp == 1 or subcomp not in expected_subcomps:
            continue
        speedup = float(row.speedup_vs_subcomp1)
        if math.isnan(speedup):
            raise SystemExit(
                f"missing CPU speedup_vs_subcomp1 for value={row.value_size} subcomp={subcomp}"
            )
        plot_rows.append(
            {
                "sst_size_mb": sst_size_mb,
                "input_sst_count": input_sst_count,
                "value_size": int(row.value_size),
                "method_group": "cpu",
                "method_key": f"subcomp_{subcomp}",
                "method_label": f"sub {subcomp}",
                "speedup": speedup,
                "speedup_std": float(getattr(row, "speedup_vs_subcomp1_std", 0.0) or 0.0),
                "details": f"RocksDB speedup vs subcomp=1 (mean rocksdb.compaction.times.micros ratio), subcomp={subcomp}",
            }
        )

    best_cpu_df = (
        case_df[case_df["requested_subcompactions"] != 1]
        .sort_values(["value_size", "speedup_vs_subcomp1"], ascending=[True, False])
        .groupby("value_size", as_index=False)
        .first()
    )

    return pd.DataFrame(plot_rows), best_cpu_df


def load_gpu_rows(
    gpu_sweep_dir: Path,
    input_sst_count: int,
    gpu_modes: list[str],
    sst_size_mb: int,
) -> pd.DataFrame:
    if not gpu_sweep_dir.exists():
        raise SystemExit(f"no such GPU sweep directory: {gpu_sweep_dir}")

    rows: list[dict[str, object]] = []
    for log_path in sorted(gpu_sweep_dir.glob("result_val*B_*.log")):
        parsed = parse_gpu_log(log_path)
        mode = str(parsed["mode"])
        if mode not in gpu_modes:
            continue
        if int(parsed["input_sst_count"]) != input_sst_count:
            raise SystemExit(
                f"GPU log {log_path} uses input_sst_count={parsed['input_sst_count']}, "
                f"expected {input_sst_count}"
            )
        rows.append(
            {
                "sst_size_mb": sst_size_mb,
                "input_sst_count": input_sst_count,
                "value_size": int(parsed["value_size"]),
                "method_group": "gpu",
                "method_key": mode,
                "method_label": GPU_MODE_LABELS.get(mode, mode),
                "speedup": float(parsed["speedup"]),
                "speedup_std": float(parsed["speedup_std"]),
                "speedup_min": float(parsed["speedup_min"]),
                "details": "GPComp speedup computed from mean CPU/GPU wall time; std uses propagated timing stddev",
                "log_path": parsed["log_path"],
            }
        )

    if not rows:
        raise SystemExit(f"no GPU logs matched the requested modes in {gpu_sweep_dir}")

    gpu_df = pd.DataFrame(rows)
    missing_modes = []
    for value_size, group in gpu_df.groupby("value_size"):
        present_modes = set(group["method_key"])
        for mode in gpu_modes:
            if mode not in present_modes:
                missing_modes.append(f"value={value_size} mode={mode}")
    if missing_modes:
        raise SystemExit(
            "GPU sweep is missing requested value/mode combinations: "
            + ", ".join(missing_modes)
        )

    return gpu_df


def plot_comparison(
    combined_df: pd.DataFrame,
    output_dir: Path,
    sst_size_mb: int,
    input_sst_count: int,
    cpu_subcomp_threads: list[int],
    gpu_modes: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    value_sizes = sorted(combined_df["value_size"].unique())
    method_order = [f"subcomp_{subcomp}" for subcomp in cpu_subcomp_threads if subcomp != 1]
    method_order.extend(gpu_modes)

    labels = {f"subcomp_{subcomp}": f"sub {subcomp}" for subcomp in cpu_subcomp_threads if subcomp != 1}
    labels.update({mode: GPU_MODE_LABELS.get(mode, mode) for mode in gpu_modes})

    colors = {
        f"subcomp_{subcomp}": CPU_SUBCOMP_COLORS.get(subcomp, "#2f6ba7")
        for subcomp in cpu_subcomp_threads
        if subcomp != 1
    }
    colors.update({mode: GPU_MODE_COLORS.get(mode, "#666666") for mode in gpu_modes})

    lookup = {
        (int(row.value_size), str(row.method_key)): {
            "speedup": float(row.speedup),
            "speedup_std": float(getattr(row, "speedup_std", 0.0) or 0.0),
        }
        for row in combined_df.itertuples(index=False)
    }

    x = np.arange(len(value_sizes), dtype=float)
    slot_count = len(method_order)
    width = min(0.82 / max(slot_count, 1), 0.12)
    offsets = (np.arange(slot_count) - (slot_count - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(max(10.5, len(value_sizes) * 1.55), 7.2))
    for idx, method_key in enumerate(method_order):
        heights = []
        errors = []
        for value_size in value_sizes:
            stats = lookup.get((value_size, method_key))
            if stats is None:
                heights.append(float("nan"))
                errors.append(0.0)
            else:
                heights.append(float(stats["speedup"]))
                errors.append(float(stats["speedup_std"]))
        ax.bar(
            x + offsets[idx],
            heights,
            width * 0.92,
            yerr=errors,
            capsize=3,
            label=labels[method_key],
            color=colors[method_key],
            edgecolor="black",
            linewidth=0.7,
        )

    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value_size) for value_size in value_sizes])
    ax.set_xlabel("Value Size (Bytes)")
    ax.set_ylabel("Normalized Speedup")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(
            facecolor=colors[method_key],
            edgecolor="black",
            linewidth=0.7,
            label=labels[method_key],
        )
        for method_key in method_order
    ]
    fig.suptitle(
        f"CPU Subcompaction vs GPU Compaction Speedup\n"
        f"SST={sst_size_mb}MB, Input={input_sst_count} SSTs",
        y=0.975,
        fontsize=17,
    )
    fig.legend(
        handles=legend_handles,
        labels=[labels[method_key] for method_key in method_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=min(len(method_order), 5),
        frameon=False,
        fontsize=13,
        columnspacing=1.8,
        handletextpad=0.8,
    )

    fig.text(
        0.5,
        0.01,
        "CPU bars are normalized to RocksDB subcomp=1. GPU bars are normalized to the GPComp synthetic CPU baseline.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.76))
    prefix = normalize_prefix(sst_size_mb, input_sst_count)
    png_path = output_dir / f"{prefix}_normalized_speedup.png"
    pdf_path = output_dir / f"{prefix}_normalized_speedup.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path


def build_summary_text(
    combined_df: pd.DataFrame,
    best_cpu_df: pd.DataFrame,
    sst_size_mb: int,
    input_sst_count: int,
) -> str:
    lines = [
        f"SST size: {sst_size_mb}MB",
        f"Requested input SSTs: {input_sst_count}",
        "",
        "Best CPU subcompaction speedups vs subcomp=1",
        "============================================",
        "",
    ]

    if best_cpu_df.empty:
        lines.append("No CPU subcompaction rows were available.")
    else:
        for row in best_cpu_df.itertuples(index=False):
            lines.append(
                f"value={int(row.value_size)}B: {float(row.speedup_vs_subcomp1):.2f}x +/- {float(getattr(row, 'speedup_vs_subcomp1_std', 0.0) or 0.0):.2f}x at subcomp={int(row.requested_subcompactions)}"
            )

    lines.extend(
        [
            "",
            "GPU mode speedups vs synthetic CPU baseline",
            "===========================================",
            "",
        ]
    )

    gpu_df = combined_df[combined_df["method_group"] == "gpu"].copy()
    for value_size in sorted(gpu_df["value_size"].unique()):
        lines.append(f"value={value_size}B:")
        value_df = gpu_df[gpu_df["value_size"] == value_size].sort_values("method_label")
        for row in value_df.itertuples(index=False):
            lines.append(
                f"  {row.method_label}: {float(row.speedup):.2f}x +/- {float(getattr(row, 'speedup_std', 0.0) or 0.0):.2f}x"
            )

    lines.extend(
        [
            "",
            "Normalization note",
            "==================",
            "",
            "CPU bars use RocksDB subcomp=1 as the baseline and show mean +/- stddev over repeated rocksdb.compaction.times.micros measurements.",
            "GPU bars use the GPComp benchmark's synthetic CPU baseline and show mean speedup derived from CPU/GPU mean wall time.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-experiment-root", type=Path, required=True)
    parser.add_argument("--gpu-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sst-size-mb", type=int, required=True)
    parser.add_argument("--input-sst-count", type=int, required=True)
    parser.add_argument(
        "--subcomp-threads",
        default="1 2 4 8 16 32",
        help="space-separated CPU subcompaction list; subcomp=1 is used as the CPU baseline",
    )
    parser.add_argument(
        "--gpu-modes",
        default="q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan",
        help="space-separated GPComp mode list",
    )
    args = parser.parse_args()

    cpu_subcomp_threads = parse_int_list(args.subcomp_threads)
    gpu_modes = parse_str_list(args.gpu_modes)

    cpu_plot_df, best_cpu_df = load_cpu_rows(
        args.cpu_experiment_root.resolve(),
        args.sst_size_mb,
        args.input_sst_count,
        cpu_subcomp_threads,
    )
    gpu_df = load_gpu_rows(
        args.gpu_sweep_dir.resolve(),
        args.input_sst_count,
        gpu_modes,
        args.sst_size_mb,
    )

    combined_df = pd.concat([cpu_plot_df, gpu_df], ignore_index=True).sort_values(
        ["value_size", "method_group", "method_label"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = normalize_prefix(args.sst_size_mb, args.input_sst_count)
    csv_path = args.output_dir / f"{prefix}_normalized_speedups.csv"
    summary_path = args.output_dir / f"{prefix}_summary.txt"
    combined_df.to_csv(csv_path, index=False)

    summary_text = build_summary_text(
        combined_df,
        best_cpu_df,
        args.sst_size_mb,
        args.input_sst_count,
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    png_path = plot_comparison(
        combined_df,
        args.output_dir,
        args.sst_size_mb,
        args.input_sst_count,
        cpu_subcomp_threads,
        gpu_modes,
    )

    print(summary_text)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
