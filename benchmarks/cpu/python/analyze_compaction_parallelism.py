#!/usr/bin/env python3
"""Analyze isolated compaction-parallelism runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_compaction_profile import events_to_dataframe, parse_compaction_events


BENCHMARK_LINE_RE = re.compile(
    r"^\s*(compact(?:all|0|1))\s*:\s*([\d.]+)\s+micros/op\s+(\d+)\s+ops/sec\s+([\d.]+)\s+seconds\s+(\d+)\s+operations(?:;(.*))?$"
)


def parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def parse_benchmark_elapsed(log_path: Path) -> tuple[float | None, str | None]:
    benchmark_name = None
    elapsed_s = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = BENCHMARK_LINE_RE.match(line)
        if not match:
            continue
        benchmark_name = match.group(1)
        elapsed_s = float(match.group(4))
    return elapsed_s, benchmark_name


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
        output_level = int(row.get("output_level", -1))
        if not levels:
            return f"unknown->L{output_level}"
        input_levels = "+".join(f"L{level}" for level in levels)
        return f"{input_levels}->L{output_level}"

    return df.apply(classify, axis=1)


def mib_per_sec(byte_count: float, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return byte_count / (1024.0 * 1024.0) / elapsed_s


def records_per_sec(record_count: float, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return record_count / elapsed_s


def million_records_per_sec(record_count: float, elapsed_s: float) -> float:
    return records_per_sec(record_count, elapsed_s) / 1_000_000.0


def pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


def dominant_component(row: pd.Series) -> str:
    parts = {
        "computation": row["compute_pct"],
        "read": row["read_pct"],
        "write": row["write_pct"],
    }
    return max(parts.items(), key=lambda item: item[1])[0]


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def throughput_view(unit: str) -> dict[str, str]:
    views = {
        "mib": {
            "column": "logical_input_throughput_mib_per_sec",
            "baseline_column": "baseline_logical_input_throughput_mib_per_sec",
            "ylabel": "Logical Input Throughput (MiB/s)",
            "title_label": "Logical Input Throughput",
            "summary_unit": "MiB/s",
        },
        "records": {
            "column": "logical_input_throughput_records_per_sec",
            "baseline_column": "baseline_logical_input_throughput_records_per_sec",
            "ylabel": "Compaction Throughput (records/s)",
            "title_label": "Compaction Throughput",
            "summary_unit": "records/s",
        },
        "mrecords": {
            "column": "logical_input_throughput_mrecords_per_sec",
            "baseline_column": "baseline_logical_input_throughput_mrecords_per_sec",
            "ylabel": "Compaction Throughput (Million records/s)",
            "title_label": "Compaction Throughput",
            "summary_unit": "Million records/s",
        },
    }
    return views[unit]


def format_throughput(value: float, unit: str) -> str:
    if unit == "records":
        return f"{value:,.0f} records/s"
    if unit == "mrecords":
        return f"{value:.2f} million records/s"
    return f"{value:.1f} MiB/s"


def parse_run(run_dir: Path) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    metadata_path = run_dir / "metadata" / "compaction_parallelism.env"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")

    meta = parse_env_file(metadata_path)
    value_size = int(meta["VALUE_SIZE"])
    key_size = int(meta["KEY_SIZE"])
    input_data_mb = int(meta["INPUT_DATA_MB"])
    sst_size_mb = int(meta["SST_SIZE_MB"])
    requested_subcompactions = int(meta["REQUESTED_SUBCOMPACTIONS"])
    target_logical_input_bytes = int(meta["TARGET_LOGICAL_INPUT_BYTES"])
    load_entries = int(meta["LOAD_ENTRIES"])

    compaction_log = run_dir / "metadata" / "rocksdb_LOG_after_compact.txt"
    events, start_info = parse_compaction_events(str(compaction_log))
    comp_df = events_to_dataframe(events, start_info)

    if comp_df.empty:
        comp_df = pd.DataFrame(
            columns=[
                "job",
                "output_level",
                "num_output_files",
                "total_output_size_bytes",
                "num_input_records",
                "num_output_records",
                "num_subcompactions",
                "input_summary",
                "compaction_desc",
                "compaction_reason",
                "input_data_size_bytes",
                "wall_us",
                "cpu_us",
                "write_io_us",
                "read_io_us",
                "has_io_stats",
            ]
        )
    else:
        comp_df = comp_df.copy()
        comp_df["pattern"] = compaction_pattern(comp_df)
        comp_df["wall_ms"] = comp_df["wall_us"] / 1000.0
        comp_df["cpu_ms"] = comp_df["cpu_us"] / 1000.0
        comp_df["read_io_ms"] = comp_df["read_io_us"] / 1000.0
        comp_df["write_io_ms"] = comp_df["write_io_us"] / 1000.0

    benchmark_elapsed_s, parsed_benchmark_name = parse_benchmark_elapsed(
        run_dir / "compact.log"
    )

    host_summary = read_json(run_dir / "host_metrics" / "summary.json")
    device_df = read_optional_csv(run_dir / "host_metrics" / "device_io.csv")
    process_df = read_optional_csv(run_dir / "host_metrics" / "process_cpu.csv")
    role_df = read_optional_csv(run_dir / "host_metrics" / "thread_role_cpu.csv")

    role_pivot = pd.DataFrame(columns=["secs_elapsed"])
    if not role_df.empty:
        role_pivot = (
            role_df.pivot_table(
                index="secs_elapsed",
                columns="thread_role",
                values="cpu_pct",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    for role_name in ("foreground", "rocksdb_compaction", "rocksdb_flush", "rocksdb_other_bg", "other"):
        if role_name not in role_pivot.columns:
            role_pivot[role_name] = 0.0

    total_wall_s = comp_df["wall_us"].sum() / 1e6 if not comp_df.empty else 0.0
    total_cpu_s = comp_df["cpu_us"].sum() / 1e6 if not comp_df.empty else 0.0
    total_read_io_s = comp_df["read_io_us"].sum() / 1e6 if not comp_df.empty else 0.0
    total_write_io_s = comp_df["write_io_us"].sum() / 1e6 if not comp_df.empty else 0.0
    total_input_bytes = float(comp_df["input_data_size_bytes"].sum()) if not comp_df.empty else 0.0
    total_output_bytes = float(comp_df["total_output_size_bytes"].sum()) if not comp_df.empty else 0.0
    total_input_records = float(comp_df["num_input_records"].sum()) if not comp_df.empty else 0.0
    total_output_records = float(comp_df["num_output_records"].sum()) if not comp_df.empty else 0.0
    target_logical_input_records = target_logical_input_bytes / float(key_size + value_size)
    total_logical_input_bytes = total_input_records * (key_size + value_size)
    total_logical_output_bytes = total_output_records * (key_size + value_size)

    elapsed_source = "benchmark_log"
    if benchmark_elapsed_s is None or benchmark_elapsed_s <= 0:
        if not device_df.empty:
            benchmark_elapsed_s = float(device_df["secs_elapsed"].max())
            elapsed_source = "host_metrics"
        elif total_wall_s > 0:
            benchmark_elapsed_s = total_wall_s
            elapsed_source = "compaction_wall_sum"
        else:
            benchmark_elapsed_s = 0.0
            elapsed_source = "missing"

    avg_role_cpu = host_summary.get("avg_thread_role_cpu_pct", {})
    max_role_cpu = host_summary.get("max_thread_role_cpu_pct", {})
    background_cpu_avg = sum(
        float(value) for key, value in avg_role_cpu.items() if key != "foreground"
    )

    read_pct = pct(total_read_io_s, total_wall_s)
    write_pct = pct(total_write_io_s, total_wall_s)
    cpu_resource_pct = pct(total_cpu_s, total_wall_s)
    compute_pct = max(0.0, 100.0 - read_pct - write_pct)

    metrics = {
        "run_dir": str(run_dir),
        "benchmark": meta["COMPACTION_BENCH"],
        "parsed_benchmark_name": parsed_benchmark_name or meta["COMPACTION_BENCH"],
        "elapsed_source": elapsed_source,
        "benchmark_elapsed_s": benchmark_elapsed_s,
        "sst_size_mb": sst_size_mb,
        "input_data_mb": input_data_mb,
        "value_size": value_size,
        "key_size": key_size,
        "requested_subcompactions": requested_subcompactions,
        "requested_bg_threads": int(meta["COMPACTION_BG_THREADS"]),
        "preload_entries": load_entries,
        "target_logical_input_bytes": target_logical_input_bytes,
        "preload_sst_count": int(meta["PRELOAD_SST_COUNT"]),
        "preload_sst_bytes": int(meta["PRELOAD_SST_BYTES"]),
        "preload_db_bytes": int(meta["PRELOAD_DB_BYTES"]),
        "compaction_count": int(len(comp_df)),
        "actual_subcompactions_avg": float(comp_df["num_subcompactions"].mean()) if not comp_df.empty else 0.0,
        "actual_subcompactions_max": int(comp_df["num_subcompactions"].max()) if not comp_df.empty else 0,
        "total_compaction_wall_s": total_wall_s,
        "total_compaction_cpu_s": total_cpu_s,
        "total_compaction_read_io_s": total_read_io_s,
        "total_compaction_write_io_s": total_write_io_s,
        "read_pct": read_pct,
        "write_pct": write_pct,
        "compute_pct": compute_pct,
        "cpu_resource_pct": cpu_resource_pct,
        "total_input_bytes": total_input_bytes,
        "total_output_bytes": total_output_bytes,
        "total_input_records": total_input_records,
        "total_output_records": total_output_records,
        "target_logical_input_records": target_logical_input_records,
        "total_logical_input_bytes": total_logical_input_bytes,
        "total_logical_output_bytes": total_logical_output_bytes,
        "input_throughput_mib_per_sec": mib_per_sec(total_input_bytes, benchmark_elapsed_s),
        "output_throughput_mib_per_sec": mib_per_sec(total_output_bytes, benchmark_elapsed_s),
        "logical_input_throughput_mib_per_sec": mib_per_sec(total_logical_input_bytes, benchmark_elapsed_s),
        "logical_target_throughput_mib_per_sec": mib_per_sec(target_logical_input_bytes, benchmark_elapsed_s),
        "total_rw_throughput_mib_per_sec": mib_per_sec(total_input_bytes + total_output_bytes, benchmark_elapsed_s),
        "logical_input_throughput_records_per_sec": records_per_sec(total_input_records, benchmark_elapsed_s),
        "logical_output_throughput_records_per_sec": records_per_sec(total_output_records, benchmark_elapsed_s),
        "logical_target_throughput_records_per_sec": records_per_sec(target_logical_input_records, benchmark_elapsed_s),
        "total_rw_throughput_records_per_sec": records_per_sec(total_input_records + total_output_records, benchmark_elapsed_s),
        "logical_input_throughput_mrecords_per_sec": million_records_per_sec(total_input_records, benchmark_elapsed_s),
        "logical_output_throughput_mrecords_per_sec": million_records_per_sec(total_output_records, benchmark_elapsed_s),
        "logical_target_throughput_mrecords_per_sec": million_records_per_sec(target_logical_input_records, benchmark_elapsed_s),
        "total_rw_throughput_mrecords_per_sec": million_records_per_sec(total_input_records + total_output_records, benchmark_elapsed_s),
        "avg_device_util_pct": float(host_summary.get("avg_device_util_pct", 0.0)),
        "max_device_util_pct": float(host_summary.get("max_device_util_pct", 0.0)),
        "avg_device_queue_depth": float(host_summary.get("avg_device_queue_depth", 0.0)),
        "avg_device_rkib_per_sec": float(host_summary.get("avg_device_rkib_per_sec", 0.0)),
        "avg_device_wkib_per_sec": float(host_summary.get("avg_device_wkib_per_sec", 0.0)),
        "avg_process_cpu_pct": float(host_summary.get("avg_process_cpu_pct", 0.0)),
        "max_process_cpu_pct": float(host_summary.get("max_process_cpu_pct", 0.0)),
        "avg_system_cpu_busy_pct": float(host_summary.get("avg_system_cpu_busy_pct", 0.0)),
        "avg_system_cpu_iowait_pct": float(host_summary.get("avg_system_cpu_iowait_pct", 0.0)),
        "avg_foreground_cpu_pct": float(avg_role_cpu.get("foreground", 0.0)),
        "avg_compaction_cpu_pct": float(avg_role_cpu.get("rocksdb_compaction", 0.0)),
        "avg_flush_cpu_pct": float(avg_role_cpu.get("rocksdb_flush", 0.0)),
        "avg_background_cpu_pct": float(background_cpu_avg),
        "max_compaction_cpu_pct": float(max_role_cpu.get("rocksdb_compaction", 0.0)),
        "dominant_breakdown": "none" if total_wall_s <= 0 else "",
    }
    if total_wall_s > 0:
        metrics["dominant_breakdown"] = dominant_component(pd.Series(metrics))

    label = (
        f"sst={sst_size_mb}MB, input={input_data_mb}MB, "
        f"value={value_size}B, sub={requested_subcompactions}"
    )

    if not comp_df.empty:
        comp_df = comp_df.copy()
        comp_df["label"] = label
        comp_df["sst_size_mb"] = sst_size_mb
        comp_df["input_data_mb"] = input_data_mb
        comp_df["value_size"] = value_size
        comp_df["requested_subcompactions"] = requested_subcompactions

    if not device_df.empty:
        device_df = device_df.copy()
        device_df["sst_size_mb"] = sst_size_mb
        device_df["input_data_mb"] = input_data_mb
        device_df["value_size"] = value_size
        device_df["requested_subcompactions"] = requested_subcompactions
    if not process_df.empty:
        process_df = process_df.copy()
        process_df["sst_size_mb"] = sst_size_mb
        process_df["input_data_mb"] = input_data_mb
        process_df["value_size"] = value_size
        process_df["requested_subcompactions"] = requested_subcompactions
    if not role_pivot.empty:
        role_pivot = role_pivot.copy()
        role_pivot["sst_size_mb"] = sst_size_mb
        role_pivot["input_data_mb"] = input_data_mb
        role_pivot["value_size"] = value_size
        role_pivot["requested_subcompactions"] = requested_subcompactions

    return metrics, {
        "compactions": comp_df,
        "device": device_df,
        "process": process_df,
        "roles": role_pivot,
    }


def best_run_per_configuration(summary_df: pd.DataFrame, throughput_column: str) -> pd.DataFrame:
    ordered = summary_df.sort_values(
        [
            "sst_size_mb",
            "input_data_mb",
            "value_size",
            throughput_column,
            "requested_subcompactions",
        ],
        ascending=[True, True, True, False, True],
    )
    best = ordered.groupby(["sst_size_mb", "input_data_mb", "value_size"], as_index=False).first()
    best = best.rename(columns={"requested_subcompactions": "best_subcompactions"})
    return best


def add_speedup_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    baseline = (
        df[df["requested_subcompactions"] == 1]
        .set_index(["sst_size_mb", "input_data_mb", "value_size"])[
            [
                "logical_input_throughput_mib_per_sec",
                "input_throughput_mib_per_sec",
                "logical_input_throughput_records_per_sec",
                "logical_input_throughput_mrecords_per_sec",
            ]
        ]
        .rename(
            columns={
                "logical_input_throughput_mib_per_sec": "baseline_logical_input_throughput_mib_per_sec",
                "input_throughput_mib_per_sec": "baseline_input_throughput_mib_per_sec",
                "logical_input_throughput_records_per_sec": "baseline_logical_input_throughput_records_per_sec",
                "logical_input_throughput_mrecords_per_sec": "baseline_logical_input_throughput_mrecords_per_sec",
            }
        )
    )
    df = df.join(baseline, on=["sst_size_mb", "input_data_mb", "value_size"])
    df["speedup_vs_subcomp1"] = (
        df["logical_input_throughput_records_per_sec"]
        / df["baseline_logical_input_throughput_records_per_sec"]
    )
    df["efficiency_vs_subcomp1"] = df["speedup_vs_subcomp1"] / df["requested_subcompactions"]
    return df


def save_value_size_lines(
    summary_df: pd.DataFrame,
    out_dir: Path,
    throughput_column: str,
    throughput_ylabel: str,
    throughput_title_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (sst_size_mb, input_data_mb), group in summary_df.groupby(["sst_size_mb", "input_data_mb"]):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for subcomp, sub_df in sorted(group.groupby("requested_subcompactions"), key=lambda item: item[0]):
            ordered = sub_df.sort_values("value_size")
            ax.plot(
                ordered["value_size"],
                ordered[throughput_column],
                marker="o",
                linewidth=2,
                label=f"sub={subcomp}",
            )
        ax.set_title(
            f"{throughput_title_label} vs Value Size\nSST={sst_size_mb}MB, Input={input_data_mb}MB"
        )
        ax.set_xlabel("Value Size (Bytes)")
        ax.set_ylabel(throughput_ylabel)
        ax.set_xticks(sorted(group["value_size"].unique()))
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"throughput_sst_{sst_size_mb}MB_input_{input_data_mb}MB.png",
            dpi=170,
        )
        plt.close(fig)


def save_breakdown_bars(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
            "compute_pct": "#fb8072",
            "read_pct": "#1bb7b3",
            "write_pct": "#4666d5",
        }
    for (sst_size_mb, input_data_mb, requested_subcompactions), group in summary_df.groupby(
        ["sst_size_mb", "input_data_mb", "requested_subcompactions"]
    ):
        ordered = group.sort_values("value_size")
        x = range(len(ordered))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x, ordered["compute_pct"], color=colors["compute_pct"], label="Computation")
        ax.bar(
            x,
            ordered["read_pct"],
            bottom=ordered["compute_pct"],
            color=colors["read_pct"],
            label="Read",
        )
        ax.bar(
            x,
            ordered["write_pct"],
            bottom=ordered["compute_pct"] + ordered["read_pct"],
            color=colors["write_pct"],
            label="Write",
        )
        ax.set_title(
            f"Normalized Compaction Time Breakdown\nSST={sst_size_mb}MB, Input={input_data_mb}MB, Sub={requested_subcompactions}",
            pad=26,
        )
        ax.set_xlabel("Value Size (Bytes)")
        ax.set_ylabel("Normalized Execution Time (%)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(v) for v in ordered["value_size"]])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        fig.legend(
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(
            out_dir
            / f"breakdown_sst_{sst_size_mb}MB_input_{input_data_mb}MB_sub_{requested_subcompactions}.png",
            dpi=170,
        )
        plt.close(fig)


def save_scaling_lines(
    summary_df: pd.DataFrame,
    out_dir: Path,
    throughput_column: str,
    throughput_ylabel: str,
    throughput_title_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (sst_size_mb, value_size), group in summary_df.groupby(["sst_size_mb", "value_size"]):
        ordered_inputs = sorted(group["input_data_mb"].unique())
        ordered_subs = sorted(group["requested_subcompactions"].unique())

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for input_data_mb in ordered_inputs:
            sub_df = group[group["input_data_mb"] == input_data_mb].sort_values(
                "requested_subcompactions"
            )
            ax.plot(
                sub_df["requested_subcompactions"],
                sub_df[throughput_column],
                marker="o",
                linewidth=2,
                label=f"input={input_data_mb}MB",
            )
        ax.set_title(
            f"{throughput_title_label} Scaling vs Subcompactions\nSST={sst_size_mb}MB, Value={value_size}B"
        )
        ax.set_xlabel("Requested Subcompactions")
        ax.set_ylabel(throughput_ylabel)
        ax.set_xticks(ordered_subs)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"scaling_sst_{sst_size_mb}MB_value_{value_size}B.png",
            dpi=170,
        )
        plt.close(fig)


def save_latency_small_multiples(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sst_sizes = sorted(summary_df["sst_size_mb"].unique())
    subcompactions = sorted(summary_df["requested_subcompactions"].unique())
    value_sizes = sorted(summary_df["value_size"].unique())
    input_sizes = sorted(summary_df["input_data_mb"].unique())

    fig, axes = plt.subplots(
        len(sst_sizes),
        len(subcompactions),
        figsize=(max(15, len(subcompactions) * 3.2), max(5, len(sst_sizes) * 3.4)),
        sharex=True,
        sharey=True,
    )

    if len(sst_sizes) == 1 and len(subcompactions) == 1:
        axes_grid = [[axes]]
    elif len(sst_sizes) == 1:
        axes_grid = [list(axes)]
    elif len(subcompactions) == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    legend_handles: dict[str, object] = {}
    positive_latency = summary_df[summary_df["total_compaction_wall_s"] > 0]["total_compaction_wall_s"]

    for row_idx, sst_size_mb in enumerate(sst_sizes):
        for col_idx, requested_subcompactions in enumerate(subcompactions):
            ax = axes_grid[row_idx][col_idx]
            panel = summary_df[
                (summary_df["sst_size_mb"] == sst_size_mb)
                & (summary_df["requested_subcompactions"] == requested_subcompactions)
            ]
            plotted_any = False
            for value_size in value_sizes:
                value_df = panel[panel["value_size"] == value_size].sort_values("input_data_mb")
                value_df = value_df[value_df["total_compaction_wall_s"] > 0]
                if value_df.empty:
                    continue
                (line,) = ax.plot(
                    value_df["input_data_mb"],
                    value_df["total_compaction_wall_s"],
                    marker="o",
                    linewidth=1.8,
                    label=f"value={value_size}B",
                )
                legend_handles.setdefault(f"value={value_size}B", line)
                plotted_any = True

            if col_idx == 0:
                ax.set_ylabel(f"SST={sst_size_mb}MB\nWall Time (s)")
            if row_idx == len(sst_sizes) - 1:
                ax.set_xlabel("Input Data (MB)")
            if row_idx == 0:
                ax.set_title(f"sub={requested_subcompactions}")

            ax.set_xscale("log", base=2)
            ax.set_xticks(input_sizes)
            ax.set_xticklabels([str(size) for size in input_sizes], rotation=45)
            if not positive_latency.empty:
                ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")

            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    "No compaction",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#666666",
                )

    handles = [legend_handles[label] for label in sorted(legend_handles)]
    labels = sorted(legend_handles)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(6, len(labels)),
            frameon=False,
        )

    fig.suptitle(
        "Compaction Wall Time vs Input Size\nRows = SST size, Columns = Requested Subcompactions",
        y=0.94,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out_dir / "latency_small_multiples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_resource_lines(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (sst_size_mb, value_size), group in summary_df.groupby(["sst_size_mb", "value_size"]):
        ordered_inputs = sorted(group["input_data_mb"].unique())
        ordered_subs = sorted(group["requested_subcompactions"].unique())

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for input_data_mb in ordered_inputs:
            sub_df = group[group["input_data_mb"] == input_data_mb].sort_values(
                "requested_subcompactions"
            )
            label = f"input={input_data_mb}MB"
            axes[0].plot(
                sub_df["requested_subcompactions"],
                sub_df["avg_process_cpu_pct"],
                marker="o",
                linewidth=2,
                label=label,
            )
            axes[1].plot(
                sub_df["requested_subcompactions"],
                sub_df["avg_device_util_pct"],
                marker="o",
                linewidth=2,
                label=label,
            )
        axes[0].set_title(
            f"CPU Utilization vs Subcompactions\nSST={sst_size_mb}MB, Value={value_size}B"
        )
        axes[0].set_ylabel("Avg Process CPU %")
        axes[0].grid(alpha=0.3)
        axes[1].set_title("Backing Device Utilization vs Subcompactions")
        axes[1].set_xlabel("Requested Subcompactions")
        axes[1].set_ylabel("Avg Device Util %")
        axes[1].set_xticks(ordered_subs)
        axes[1].grid(alpha=0.3)
        axes[0].legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"resources_sst_{sst_size_mb}MB_value_{value_size}B.png",
            dpi=170,
        )
        plt.close(fig)


def save_heatmap(
    data: pd.DataFrame,
    value_column: str,
    annotation_fn,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    inputs = sorted(data["input_data_mb"].unique())
    values = sorted(data["value_size"].unique())
    matrix = []
    annotations = []
    for input_data_mb in inputs:
        row = []
        ann_row = []
        for value_size in values:
            row_df = data[
                (data["input_data_mb"] == input_data_mb) & (data["value_size"] == value_size)
            ]
            if row_df.empty:
                row.append(float("nan"))
                ann_row.append("")
            else:
                record = row_df.iloc[0]
                row.append(float(record[value_column]))
                ann_row.append(annotation_fn(record))
        matrix.append(row)
        annotations.append(ann_row)

    fig, ax = plt.subplots(figsize=(max(7, len(values) * 1.2), max(5, len(inputs) * 0.85)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values])
    ax.set_yticks(range(len(inputs)))
    ax.set_yticklabels([str(v) for v in inputs])
    ax.set_xlabel("Value Size (Bytes)")
    ax.set_ylabel("Input Data Size (MB)")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    for row_idx, ann_row in enumerate(annotations):
        for col_idx, text in enumerate(ann_row):
            if not text:
                continue
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_best_run_overviews(
    best_df: pd.DataFrame,
    out_dir: Path,
    throughput_column: str,
    throughput_unit: str,
    throughput_title_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sst_size_mb, group in best_df.groupby("sst_size_mb"):
        save_heatmap(
            group,
            throughput_column,
            lambda row: f"{format_throughput(row[throughput_column], throughput_unit).split()[0]}\nsub={int(row['best_subcompactions'])}",
            f"Best {throughput_title_label}\nSST={sst_size_mb}MB",
            throughput_view(throughput_unit)["summary_unit"],
            out_dir / f"best_throughput_sst_{sst_size_mb}MB.png",
        )
        speedup_group = group.copy()
        speedup_group["speedup_vs_subcomp1"] = speedup_group["speedup_vs_subcomp1"].fillna(1.0)
        save_heatmap(
            speedup_group,
            "speedup_vs_subcomp1",
            lambda row: f"{row['speedup_vs_subcomp1']:.2f}x\nsub={int(row['best_subcompactions'])}",
            f"Best Speedup vs Sub=1\nSST={sst_size_mb}MB",
            "Speedup",
            out_dir / f"best_speedup_sst_{sst_size_mb}MB.png",
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(group['input_data_mb'].unique()) * 0.85)))
        for ax, column, label in zip(
            axes,
            ["compute_pct", "read_pct", "write_pct"],
            ["Computation %", "Read %", "Write %"],
        ):
            inputs = sorted(group["input_data_mb"].unique())
            values = sorted(group["value_size"].unique())
            matrix = []
            for input_data_mb in inputs:
                row = []
                for value_size in values:
                    row_df = group[
                        (group["input_data_mb"] == input_data_mb)
                        & (group["value_size"] == value_size)
                    ]
                    row.append(float(row_df.iloc[0][column]) if not row_df.empty else float("nan"))
                matrix.append(row)
            image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=100)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(v) for v in values])
            ax.set_yticks(range(len(inputs)))
            ax.set_yticklabels([str(v) for v in inputs])
            ax.set_xlabel("Value Size (Bytes)")
            ax.set_ylabel("Input Data Size (MB)")
            ax.set_title(label)
            for row_idx, input_data_mb in enumerate(inputs):
                for col_idx, value_size in enumerate(values):
                    row_df = group[
                        (group["input_data_mb"] == input_data_mb)
                        & (group["value_size"] == value_size)
                    ]
                    if row_df.empty:
                        continue
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{row_df.iloc[0][column]:.0f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Best-Run Time Breakdown\nSST={sst_size_mb}MB", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"best_breakdown_sst_{sst_size_mb}MB.png", dpi=180)
        plt.close(fig)


def build_patterns_outputs(compaction_df: pd.DataFrame, analysis_dir: Path) -> None:
    if compaction_df.empty:
        (analysis_dir / "compaction_patterns_summary.txt").write_text(
            "No compaction events found.\n", encoding="utf-8"
        )
        return

    pattern_counts = (
        compaction_df.groupby(
            ["sst_size_mb", "input_data_mb", "value_size", "requested_subcompactions", "pattern"],
            as_index=False,
        )
        .agg(
            count=("pattern", "size"),
            avg_wall_ms=("wall_ms", "mean"),
            avg_cpu_ms=("cpu_ms", "mean"),
            avg_input_mb=("input_data_size_bytes", lambda s: s.mean() / (1024 * 1024)),
            avg_output_mb=("total_output_size_bytes", lambda s: s.mean() / (1024 * 1024)),
            avg_subcompactions=("num_subcompactions", "mean"),
        )
        .sort_values(
            ["sst_size_mb", "input_data_mb", "value_size", "requested_subcompactions", "count"],
            ascending=[True, True, True, True, False],
        )
    )
    pattern_counts.to_csv(analysis_dir / "compaction_pattern_counts.csv", index=False)

    overall = (
        compaction_df.groupby("pattern", as_index=False)
        .agg(
            count=("pattern", "size"),
            avg_wall_ms=("wall_ms", "mean"),
            avg_cpu_ms=("cpu_ms", "mean"),
            avg_input_mb=("input_data_size_bytes", lambda s: s.mean() / (1024 * 1024)),
            avg_output_mb=("total_output_size_bytes", lambda s: s.mean() / (1024 * 1024)),
        )
        .sort_values("count", ascending=False)
    )

    lines = ["Overall compaction pattern frequencies", "====================================", ""]
    for _, row in overall.iterrows():
        lines.append(
            f"{row['pattern']}: count={int(row['count'])}, "
            f"avg wall={row['avg_wall_ms']:.1f} ms, avg cpu={row['avg_cpu_ms']:.1f} ms, "
            f"avg input={row['avg_input_mb']:.1f} MiB, avg output={row['avg_output_mb']:.1f} MiB"
        )
    lines.append("")

    top_runs = (
        pattern_counts.sort_values("count", ascending=False)
        .head(25)
        .itertuples(index=False)
    )
    lines.append("Most frequent per-run pattern slices")
    lines.append("================================")
    lines.append("")
    for row in top_runs:
        lines.append(
            f"sst={row.sst_size_mb}MB input={row.input_data_mb}MB value={row.value_size}B "
            f"sub={row.requested_subcompactions} {row.pattern}: count={int(row.count)}, "
            f"avg wall={row.avg_wall_ms:.1f} ms, avg input={row.avg_input_mb:.1f} MiB, "
            f"avg actual subcomp={row.avg_subcompactions:.1f}"
        )

    (analysis_dir / "compaction_patterns_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def heuristic_gpu_candidates(best_df: pd.DataFrame) -> pd.DataFrame:
    return best_df[
        (best_df["compute_pct"] >= 45.0)
        & (best_df["best_subcompactions"] >= 4)
        & (best_df["speedup_vs_subcomp1"].fillna(1.0) >= 1.15)
        & (best_df["cpu_resource_pct"] >= 100.0)
    ].sort_values(
        ["compute_pct", "speedup_vs_subcomp1", "logical_input_throughput_records_per_sec"],
        ascending=[False, False, False],
    )


def build_summary_text(
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    throughput_column: str,
    baseline_column: str,
    throughput_unit: str,
    throughput_title_label: str,
) -> str:
    lines: list[str] = []
    lines.append(
        f"Analyzed {len(summary_df)} compaction runs across {len(best_df)} unique "
        f"(SST size, input size, value size) configurations."
    )
    lines.append("")

    lines.append(f"Best absolute {throughput_title_label.lower()}")
    lines.append("=" * len(lines[-1]))
    lines.append("")
    for row in best_df.nlargest(10, throughput_column).itertuples(index=False):
        speedup = 1.0 if math.isnan(row.speedup_vs_subcomp1) else row.speedup_vs_subcomp1
        lines.append(
            f"sst={row.sst_size_mb}MB input={row.input_data_mb}MB value={row.value_size}B: "
            f"{format_throughput(getattr(row, throughput_column), throughput_unit)} at sub={int(row.best_subcompactions)} "
            f"(speedup {speedup:.2f}x, compute/read/write {row.compute_pct:.0f}/{row.read_pct:.0f}/{row.write_pct:.0f}%, "
            f"cpu resource {row.cpu_resource_pct:.0f}% of wall)"
        )
    lines.append("")

    lines.append("Largest speedups over sub=1")
    lines.append("===========================")
    lines.append("")
    ranked_speedups = best_df.copy()
    ranked_speedups["speedup_vs_subcomp1"] = ranked_speedups["speedup_vs_subcomp1"].fillna(1.0)
    for row in ranked_speedups.nlargest(10, "speedup_vs_subcomp1").itertuples(index=False):
        lines.append(
            f"sst={row.sst_size_mb}MB input={row.input_data_mb}MB value={row.value_size}B: "
            f"{row.speedup_vs_subcomp1:.2f}x at sub={int(row.best_subcompactions)} "
            f"(baseline sub=1 -> {format_throughput(getattr(row, baseline_column), throughput_unit)}, "
            f"best -> {format_throughput(getattr(row, throughput_column), throughput_unit)})"
        )
    lines.append("")

    dominant_counts = Counter(best_df["dominant_breakdown"])
    lines.append("Dominant time component at the best-subcomp point")
    lines.append("=================================================")
    lines.append("")
    for name in ("computation", "read", "write", "none"):
        if dominant_counts.get(name, 0):
            lines.append(f"{name}: {dominant_counts[name]} configurations")
    lines.append("")

    lines.append("Configurations whose best point stayed at sub=1")
    lines.append("===============================================")
    lines.append("")
    stayed_single = best_df[best_df["best_subcompactions"] == 1]
    if stayed_single.empty:
        lines.append("None.")
    else:
        for row in stayed_single.head(20).itertuples(index=False):
            lines.append(
                f"sst={row.sst_size_mb}MB input={row.input_data_mb}MB value={row.value_size}B: "
                f"{format_throughput(getattr(row, throughput_column), throughput_unit)}, "
                f"compute/read/write {row.compute_pct:.0f}/{row.read_pct:.0f}/{row.write_pct:.0f}%, "
                f"cpu resource {row.cpu_resource_pct:.0f}% of wall"
            )
    lines.append("")

    lines.append("Potential GPU-offload candidate slices (heuristic)")
    lines.append("==================================================")
    lines.append("")
    lines.append(
        "Heuristic used: best CPU point still has >=45% compute wall-share, "
        "best_subcompactions >= 4, speedup over sub=1 >= 1.15x, and aggregate "
        "CPU time >= 100% of wall time."
    )
    lines.append("")
    gpu_candidates = heuristic_gpu_candidates(best_df)
    if gpu_candidates.empty:
        lines.append("No slices met the heuristic.")
    else:
        for row in gpu_candidates.head(15).itertuples(index=False):
            lines.append(
                f"sst={row.sst_size_mb}MB input={row.input_data_mb}MB value={row.value_size}B: "
                f"best sub={int(row.best_subcompactions)}, speedup={row.speedup_vs_subcomp1:.2f}x, "
                f"throughput={format_throughput(getattr(row, throughput_column), throughput_unit)}, "
                f"compute/read/write={row.compute_pct:.0f}/{row.read_pct:.0f}/{row.write_pct:.0f}%, "
                f"cpu resource={row.cpu_resource_pct:.0f}% of wall, "
                f"avg proc CPU={row.avg_process_cpu_pct:.1f}%, avg device util={row.avg_device_util_pct:.1f}%."
            )
    lines.append("")
    lines.append(
        "These candidate labels are an inference from the collected CPU and IO breakdowns, "
        "not a direct measurement of GPU benefit."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--throughput-unit",
        choices=["mib", "records", "mrecords"],
        default="mrecords",
        help="display throughput as MiB/s, records/s, or million records/s (default: mrecords)",
    )
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    if not experiment_root.exists():
        raise SystemExit(f"no such directory: {experiment_root}")
    throughput_cfg = throughput_view(args.throughput_unit)

    run_dirs = sorted(
        p for p in experiment_root.rglob("subcomp_*") if (p / "metadata" / "compaction_parallelism.env").exists()
    )
    if not run_dirs:
        raise SystemExit(f"no compaction_parallelism runs found under {experiment_root}")

    metrics_rows: list[dict[str, object]] = []
    compaction_frames: list[pd.DataFrame] = []
    device_frames: list[pd.DataFrame] = []
    process_frames: list[pd.DataFrame] = []
    role_frames: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        metrics, frames = parse_run(run_dir)
        metrics_rows.append(metrics)
        if not frames["compactions"].empty:
            compaction_frames.append(frames["compactions"])
        if not frames["device"].empty:
            device_frames.append(frames["device"])
        if not frames["process"].empty:
            process_frames.append(frames["process"])
        if not frames["roles"].empty:
            role_frames.append(frames["roles"])

    summary_df = pd.DataFrame(metrics_rows).sort_values(
        ["sst_size_mb", "input_data_mb", "value_size", "requested_subcompactions"]
    )
    summary_df = add_speedup_columns(summary_df)
    best_df = best_run_per_configuration(summary_df, throughput_cfg["column"])

    analysis_dir = experiment_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(analysis_dir / "summary_metrics.csv", index=False)
    best_df.to_csv(analysis_dir / "best_runs.csv", index=False)
    heuristic_gpu_candidates(best_df).to_csv(analysis_dir / "gpu_candidate_runs.csv", index=False)

    if compaction_frames:
        compaction_df = pd.concat(compaction_frames, ignore_index=True)
        compaction_df.to_csv(analysis_dir / "compaction_events.csv", index=False)
        build_patterns_outputs(compaction_df, analysis_dir)
    else:
        compaction_df = pd.DataFrame()
        (analysis_dir / "compaction_patterns_summary.txt").write_text(
            "No compaction events found.\n", encoding="utf-8"
        )

    if device_frames:
        pd.concat(device_frames, ignore_index=True).to_csv(
            analysis_dir / "device_io_timeseries.csv", index=False
        )
    if process_frames:
        pd.concat(process_frames, ignore_index=True).to_csv(
            analysis_dir / "process_cpu_timeseries.csv", index=False
        )
    if role_frames:
        pd.concat(role_frames, ignore_index=True).to_csv(
            analysis_dir / "thread_role_cpu_timeseries.csv", index=False
        )

    save_value_size_lines(
        summary_df,
        analysis_dir / "throughput_vs_value_size",
        throughput_cfg["column"],
        throughput_cfg["ylabel"],
        throughput_cfg["title_label"],
    )
    save_breakdown_bars(summary_df, analysis_dir / "time_breakdown_vs_value_size")
    save_scaling_lines(
        summary_df,
        analysis_dir / "throughput_vs_subcompactions",
        throughput_cfg["column"],
        throughput_cfg["ylabel"],
        throughput_cfg["title_label"],
    )
    save_latency_small_multiples(summary_df, analysis_dir / "latency_vs_input_size")
    save_resource_lines(summary_df, analysis_dir / "resource_utilization_vs_subcompactions")
    save_best_run_overviews(
        best_df,
        analysis_dir / "overview",
        throughput_cfg["column"],
        args.throughput_unit,
        throughput_cfg["title_label"],
    )

    summary_text = build_summary_text(
        summary_df,
        best_df,
        throughput_cfg["column"],
        throughput_cfg["baseline_column"],
        args.throughput_unit,
        throughput_cfg["title_label"],
    )
    (analysis_dir / "analysis_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"\nWrote analysis to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
