#!/usr/bin/env python3
"""
Parse RocksDB LOG files to produce a compaction IO/CPU breakdown.

Extracts per-compaction events from the RocksDB LOG (event log JSON lines)
and computes a Read / Write / Computation breakdown analogous to GPComp Fig. 2.

Metrics source (when --report_bg_io_stats=true):
  - compaction_time_micros       → wall-clock time
  - compaction_time_cpu_micros   → CPU time (IO CPU time already subtracted)
  - file_write_nanos             → time in write()/pwrite()
  - file_fsync_nanos             → time in fsync()
  - file_range_sync_nanos        → time in sync_file_range()
  - file_prepare_write_nanos     → time in fallocate()

Derived:
  - write_io_us  = (file_write_nanos + file_fsync_nanos
                     + file_range_sync_nanos + file_prepare_write_nanos) / 1000
  - read_io_us   = wall_us - cpu_us - write_io_us
  - compute_us   = cpu_us

Usage:
  python3 parse_compaction_profile.py <RUN_DIR> [--plot] [--csv output.csv]

  RUN_DIR should contain metadata/rocksdb_LOG_after_compact.txt or
  metadata/rocksdb_LOG_after_fillrandom.txt (the script tries both).

  If a strace_compact.log exists in RUN_DIR, strace-based validation
  numbers are also printed.
"""

import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd

# ── Try optional plotting ──────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =====================================================================
# 1.  Parse compaction events from RocksDB LOG
# =====================================================================

# RocksDB event log lines look like:
#   2025/06/15-12:00:00.123456 ... EVENT_LOG_v1 {"job":1,"event":"compaction_finished",...}
EVENT_LOG_RE = re.compile(r"EVENT_LOG_v1\s+(\{.+\})\s*$")


# Human-readable compaction start line:
#   [default] [JOB 6] Compacting 4@0 files to L1, score 1.00
#   [default] [JOB 11] Compacting 4@0 + 3@1 files to L1, score 1.00
COMPACTING_RE = re.compile(
    r"\[JOB (\d+)\] Compacting (.+?) files? to L(\d+)"
)


def parse_compaction_events(log_path: str) -> tuple[list[dict], dict[int, dict]]:
    """Return (finished_events, start_info_by_job) from a RocksDB LOG.

    start_info_by_job maps job_id → {
        'input_summary': '4@L0 + 3@L1',   # human-readable
        'output_level': 1,
        'compaction_reason': '...',
        'input_data_size': ...,
        'input_files_per_level': {0: 4, 1: 3},  # level → file count
    }
    """
    finished_events = []
    start_info: dict[int, dict] = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # ── Human-readable "Compacting" line ──
            cm = COMPACTING_RE.search(line)
            if cm:
                job_id = int(cm.group(1))
                raw_input = cm.group(2)          # e.g. "4@0 + 3@1"
                out_level = int(cm.group(3))

                # Normalise "4@0 + 3@1" → "4@L0 + 3@L1"
                parts = [p.strip() for p in raw_input.split("+")]
                normalised_parts = []
                files_per_level: dict[int, int] = {}
                for part in parts:
                    m2 = re.match(r"(\d+)@(\d+)", part)
                    if m2:
                        n_files = int(m2.group(1))
                        level = int(m2.group(2))
                        normalised_parts.append(f"{n_files}@L{level}")
                        files_per_level[level] = n_files

                input_summary = " + ".join(normalised_parts)

                if job_id not in start_info:
                    start_info[job_id] = {}
                start_info[job_id].update({
                    "input_summary": input_summary,
                    "output_level": out_level,
                    "input_files_per_level": files_per_level,
                })
                continue  # don't re-check EVENT_LOG on this line

            # ── JSON event log lines ──
            em = EVENT_LOG_RE.search(line)
            if not em:
                continue
            try:
                obj = json.loads(em.group(1))
            except json.JSONDecodeError:
                continue

            event_type = obj.get("event")
            if event_type == "compaction_started":
                job_id = obj.get("job", -1)
                if job_id not in start_info:
                    start_info[job_id] = {}
                start_info[job_id]["compaction_reason"] = obj.get(
                    "compaction_reason", ""
                )
                start_info[job_id]["input_data_size"] = obj.get(
                    "input_data_size", 0
                )
                # Also extract file counts from JSON arrays (files_L0, etc.)
                if "input_files_per_level" not in start_info[job_id]:
                    fpl: dict[int, int] = {}
                    for key, val in obj.items():
                        m3 = re.match(r"files_L(\d+)", key)
                        if m3 and isinstance(val, list):
                            fpl[int(m3.group(1))] = len(val)
                    if fpl:
                        start_info[job_id]["input_files_per_level"] = fpl
                        parts_str = " + ".join(
                            f"{cnt}@L{lvl}"
                            for lvl, cnt in sorted(fpl.items())
                        )
                        start_info[job_id].setdefault("input_summary", parts_str)

            elif event_type == "compaction_finished":
                finished_events.append(obj)

    return finished_events, start_info


def events_to_dataframe(
    events: list[dict], start_info: dict[int, dict] | None = None
) -> pd.DataFrame:
    """Convert compaction_finished events into a tidy DataFrame with the
    Read / Write / Computation breakdown and input/output level info."""
    if start_info is None:
        start_info = {}

    rows = []
    for ev in events:
        wall_us = ev.get("compaction_time_micros", 0)
        cpu_us = ev.get("compaction_time_cpu_micros", 0)

        # Write-side IO nanos (only present with report_bg_io_stats)
        write_nanos = ev.get("file_write_nanos", 0)
        fsync_nanos = ev.get("file_fsync_nanos", 0)
        range_sync_nanos = ev.get("file_range_sync_nanos", 0)
        prepare_write_nanos = ev.get("file_prepare_write_nanos", 0)

        total_write_io_ns = (
            write_nanos + fsync_nanos + range_sync_nanos + prepare_write_nanos
        )
        write_io_us = total_write_io_ns / 1000.0

        # Read IO = wall - CPU - write IO  (everything that isn't CPU or write)
        read_io_us = max(0.0, wall_us - cpu_us - write_io_us)

        has_io_stats = any(
            ev.get(k, 0) > 0
            for k in [
                "file_write_nanos",
                "file_fsync_nanos",
                "file_range_sync_nanos",
                "file_prepare_write_nanos",
            ]
        )

        job_id = ev.get("job", -1)
        sinfo = start_info.get(job_id, {})
        input_summary = sinfo.get("input_summary", "")
        out_level = ev.get("output_level", -1)
        num_out = ev.get("num_output_files", 0)

        # Build a compact description like "4@L0 + 3@L1 → 5@L1"
        if input_summary:
            compaction_desc = f"{input_summary} → {num_out}@L{out_level}"
        else:
            compaction_desc = f"→ {num_out}@L{out_level}"

        rows.append(
            {
                "job": job_id,
                "output_level": out_level,
                "num_output_files": num_out,
                "total_output_size_bytes": ev.get("total_output_size", 0),
                "num_input_records": ev.get("num_input_records", 0),
                "num_output_records": ev.get("num_output_records", 0),
                "num_subcompactions": ev.get("num_subcompactions", 1),
                "input_summary": input_summary,
                "compaction_desc": compaction_desc,
                "compaction_reason": sinfo.get("compaction_reason", ""),
                "input_data_size_bytes": sinfo.get("input_data_size", 0),
                "wall_us": wall_us,
                "cpu_us": cpu_us,
                "write_io_us": write_io_us,
                "read_io_us": read_io_us,
                "file_write_nanos": write_nanos,
                "file_fsync_nanos": fsync_nanos,
                "file_range_sync_nanos": range_sync_nanos,
                "file_prepare_write_nanos": prepare_write_nanos,
                "has_io_stats": has_io_stats,
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# 2.  Also parse the "compacted to:" human-readable summary lines
# =====================================================================

COMPACTED_TO_RE = re.compile(
    r"\[(\S+)\] compacted to: .+, MB/sec: ([\d.]+) rd, ([\d.]+) wr, "
    r"level (\d+), files in\((\d+), (\d+)\)"
    r".+MB in\(([\d.]+), ([\d.]+)"
    r".+out\(([\d.]+)"
    r".+records in: (\d+), records dropped: (\d+)"
)


def parse_compacted_to_lines(log_path: str) -> list[dict]:
    """Parse human-readable 'compacted to:' lines for MB/sec and data sizes."""
    results = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = COMPACTED_TO_RE.search(line)
            if m:
                results.append(
                    {
                        "cf_name": m.group(1),
                        "rd_mb_per_sec": float(m.group(2)),
                        "wr_mb_per_sec": float(m.group(3)),
                        "output_level": int(m.group(4)),
                        "input_files_non_output": int(m.group(5)),
                        "input_files_output": int(m.group(6)),
                        "mb_read_non_output": float(m.group(7)),
                        "mb_read_output": float(m.group(8)),
                        "mb_written": float(m.group(9)),
                        "records_in": int(m.group(10)),
                        "records_dropped": int(m.group(11)),
                    }
                )
    return results


# =====================================================================
# 3.  Parse strace output for validation
# =====================================================================


def parse_strace_log(strace_path: str) -> dict:
    """Parse strace -T output to compute per-syscall total time.

    Returns dict mapping syscall name → total seconds spent.
    Expects lines like:
      1234  1719500000.123456 pread64(3, ...) = 4096 <0.000123>
    """
    if not os.path.exists(strace_path):
        return {}

    # Matches: <time_in_seconds> at end of strace -T lines
    time_re = re.compile(r"(\w+)\(.*<([\d.]+)>$")
    totals: dict[str, float] = {}

    with open(strace_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = time_re.search(line.strip())
            if m:
                syscall = m.group(1)
                secs = float(m.group(2))
                totals[syscall] = totals.get(syscall, 0.0) + secs

    return totals


def infer_value_size_from_run_dir(run_dir: str) -> int | None:
    """Infer value size from a path segment like value_32."""
    for part in Path(run_dir).parts:
        match = re.fullmatch(r"value_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def resolve_multi_value_runs(base_path: str) -> tuple[Path, dict[int, Path]]:
    """Resolve sibling run directories that match the same suffix under value_*/."""
    base = Path(base_path).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No such path: {base}")

    direct_value_dirs = sorted(p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"value_\d+", p.name)) if base.is_dir() else []
    if direct_value_dirs and find_log_file(str(base)) is None:
        root = base
        runs: dict[int, Path] = {}
        for value_dir in direct_value_dirs:
            match = re.fullmatch(r"value_(\d+)", value_dir.name)
            if not match:
                continue
            candidates = sorted(p for p in value_dir.rglob("*") if p.is_dir() and find_log_file(str(p)))
            if candidates:
                runs[int(match.group(1))] = candidates[-1]
        return root, runs

    parts = list(base.parts)
    value_index = next((i for i, part in enumerate(parts) if re.fullmatch(r"value_\d+", part)), None)
    if value_index is None:
        return base, {}

    suffix = Path(*parts[value_index + 1 :])
    root = Path(*parts[:value_index])
    runs = {}
    for sibling in sorted(root.glob("value_*")):
        if not sibling.is_dir():
            continue
        match = re.fullmatch(r"value_(\d+)", sibling.name)
        if not match:
            continue
        candidate = sibling / suffix
        if candidate.is_dir() and find_log_file(str(candidate)):
            runs[int(match.group(1))] = candidate
    return root, runs


def resolve_multi_value_output_dir(base_path: str, root_dir: Path) -> Path:
    """Choose a shared output directory for multi-value artifacts."""
    base = Path(base_path).resolve()
    parts = list(base.parts)
    value_index = next((i for i, part in enumerate(parts) if re.fullmatch(r"value_\d+", part)), None)
    if value_index is None:
        return root_dir

    relative_suffix = Path(*parts[value_index + 1 :])
    output_dir = root_dir / relative_suffix.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_experiment_tag(base_path: str) -> str | None:
    """Infer a stable experiment tag from the run path suffix after value_*/."""
    base = Path(base_path).resolve()
    parts = list(base.parts)
    value_index = next((i for i, part in enumerate(parts) if re.fullmatch(r"value_\d+", part)), None)
    if value_index is None:
        return None

    relative_suffix = Path(*parts[value_index + 1 :])
    tag = relative_suffix.name.strip()
    return tag or None


# =====================================================================
# 4.  Reporting
# =====================================================================


def print_per_compaction_table(df: pd.DataFrame) -> None:
    """Print a detailed per-compaction table."""
    has_desc = "compaction_desc" in df.columns

    print("\n" + "=" * 120)
    print("Per-Compaction Breakdown")
    print("=" * 120)

    desc_width = 28
    header = (
        f"{'Job':>5} {'Compaction':<{desc_width}} "
        f"{'Wall(ms)':>10} {'CPU(ms)':>10} "
        f"{'Read IO(ms)':>12} {'Write IO(ms)':>12} "
        f"{'Read%':>7} {'Write%':>7} {'CPU%':>7} "
        f"{'OutSize(MB)':>12}"
    )
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        wall_ms = row["wall_us"] / 1000.0
        cpu_ms = row["cpu_us"] / 1000.0
        read_ms = row["read_io_us"] / 1000.0
        write_ms = row["write_io_us"] / 1000.0
        out_mb = row["total_output_size_bytes"] / (1024 * 1024)

        if wall_ms > 0:
            read_pct = read_ms / wall_ms * 100
            write_pct = write_ms / wall_ms * 100
            cpu_pct = cpu_ms / wall_ms * 100
        else:
            read_pct = write_pct = cpu_pct = 0.0

        desc = row["compaction_desc"] if has_desc else f"→ L{int(row['output_level'])}"
        # Truncate long descriptions
        if len(desc) > desc_width:
            desc = desc[: desc_width - 1] + "…"

        print(
            f"{int(row['job']):>5} {desc:<{desc_width}} "
            f"{wall_ms:>10.1f} {cpu_ms:>10.1f} "
            f"{read_ms:>12.1f} {write_ms:>12.1f} "
            f"{read_pct:>6.1f}% {write_pct:>6.1f}% {cpu_pct:>6.1f}% "
            f"{out_mb:>12.1f}"
        )


def print_aggregate_summary(df: pd.DataFrame) -> None:
    """Print the aggregate breakdown (GPComp Fig. 2 style)."""
    total_wall = df["wall_us"].sum()
    total_cpu = df["cpu_us"].sum()
    total_write = df["write_io_us"].sum()
    total_read = df["read_io_us"].sum()

    print("\n" + "=" * 60)
    print("Aggregate Compaction Time Breakdown (GPComp Fig. 2 style)")
    print("=" * 60)

    if total_wall > 0:
        read_pct = total_read / total_wall * 100
        write_pct = total_write / total_wall * 100
        cpu_pct = total_cpu / total_wall * 100
    else:
        read_pct = write_pct = cpu_pct = 0.0
        print("  WARNING: No compaction wall-clock time recorded.")
        return

    print(f"  Total compactions:     {len(df)}")
    print(f"  Total wall time:       {total_wall / 1e6:.2f} s")
    print(f"  Total CPU time:        {total_cpu / 1e6:.2f} s")
    print(f"  Total read IO time:    {total_read / 1e6:.2f} s")
    print(f"  Total write IO time:   {total_write / 1e6:.2f} s")

    # ── Compaction shape summary ──
    if "compaction_desc" in df.columns:
        print()
        print("  Compaction shapes:")
        # Count how many compactions of each shape
        shape_counts = df["compaction_desc"].value_counts()
        for shape, count in shape_counts.items():
            if count > 1:
                print(f"    {shape}  (×{count})")
            else:
                print(f"    {shape}")

        # Also summarise by input/output level pattern
        if "compaction_reason" in df.columns:
            reasons = df["compaction_reason"].value_counts()
            if len(reasons) > 0:
                print()
                print("  Compaction reasons:")
                for reason, count in reasons.items():
                    if reason:
                        print(f"    {reason}: {count}")

    print()
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Read IO:        {read_pct:>6.2f}%               │")
    print(f"  │  Write IO:       {write_pct:>6.2f}%               │")
    print(f"  │  Computation:    {cpu_pct:>6.2f}%               │")
    print(f"  │  ─────────────────────────               │")
    print(f"  │  Total:          {read_pct + write_pct + cpu_pct:>6.2f}%               │")
    print(f"  └─────────────────────────────────────────┘")

    if not df["has_io_stats"].all():
        n_missing = (~df["has_io_stats"]).sum()
        print(
            f"\n  ⚠  {n_missing}/{len(df)} compactions had no IO stats "
            f"(report_bg_io_stats may not have been enabled)."
        )
        print(
            "     For those, Read IO and Write IO are estimated "
            "(read = wall - cpu, write = 0)."
        )

    # Write IO sub-breakdown
    total_write_ns = df["file_write_nanos"].sum()
    total_fsync_ns = df["file_fsync_nanos"].sum()
    total_range_sync_ns = df["file_range_sync_nanos"].sum()
    total_prepare_ns = df["file_prepare_write_nanos"].sum()

    if total_write_ns + total_fsync_ns + total_range_sync_ns + total_prepare_ns > 0:
        print("\n  Write IO Sub-Breakdown:")
        total_wio_ns = total_write_ns + total_fsync_ns + total_range_sync_ns + total_prepare_ns
        print(f"    file_write:         {total_write_ns / 1e6:>8.1f} ms  ({total_write_ns / total_wio_ns * 100:>5.1f}%)")
        print(f"    file_fsync:         {total_fsync_ns / 1e6:>8.1f} ms  ({total_fsync_ns / total_wio_ns * 100:>5.1f}%)")
        print(f"    file_range_sync:    {total_range_sync_ns / 1e6:>8.1f} ms  ({total_range_sync_ns / total_wio_ns * 100:>5.1f}%)")
        print(f"    file_prepare_write: {total_prepare_ns / 1e6:>8.1f} ms  ({total_prepare_ns / total_wio_ns * 100:>5.1f}%)")


def print_strace_summary(strace_totals: dict) -> None:
    """Print strace-based validation numbers."""
    if not strace_totals:
        return

    print("\n" + "=" * 60)
    print("strace Validation (syscall-level timing)")
    print("=" * 60)

    read_calls = ["pread64", "read"]
    write_calls = ["pwrite64", "write"]
    sync_calls = ["fsync", "fdatasync", "sync_file_range"]

    read_s = sum(strace_totals.get(c, 0) for c in read_calls)
    write_s = sum(strace_totals.get(c, 0) for c in write_calls)
    sync_s = sum(strace_totals.get(c, 0) for c in sync_calls)

    total_io = read_s + write_s + sync_s
    print(f"  Read syscalls:   {read_s:.3f} s")
    print(f"  Write syscalls:  {write_s:.3f} s")
    print(f"  Sync syscalls:   {sync_s:.3f} s")
    print(f"  Total IO:        {total_io:.3f} s")

    if total_io > 0:
        print(f"\n  Read share of IO:  {read_s / total_io * 100:.1f}%")
        print(f"  Write share of IO: {(write_s + sync_s) / total_io * 100:.1f}%")

    print("\n  All syscalls:")
    for syscall, secs in sorted(strace_totals.items(), key=lambda x: -x[1]):
        print(f"    {syscall:<20s} {secs:.3f} s")


def plot_breakdown(df: pd.DataFrame, output_path: str) -> None:
    """Generate a stacked bar chart similar to GPComp Fig. 2."""
    if not HAS_MATPLOTLIB:
        print("  (matplotlib not available — skipping plot)")
        return

    total_wall = df["wall_us"].sum()
    if total_wall == 0:
        return

    total_read = df["read_io_us"].sum()
    total_write = df["write_io_us"].sum()
    total_cpu = df["cpu_us"].sum()

    read_pct = total_read / total_wall * 100
    write_pct = total_write / total_wall * 100
    cpu_pct = total_cpu / total_wall * 100
    value_size = infer_value_size_from_run_dir(output_path)
    xtick_label = f"{value_size}B" if value_size is not None else "Run"

    fig, ax = plt.subplots(figsize=(4, 5))

    bar_width = 0.5
    x = [0]

    ax.bar(x, [read_pct], bar_width, label="Read", color="#4472C4")
    ax.bar(
        x, [write_pct], bar_width, bottom=[read_pct], label="Write", color="#A9D18E"
    )
    ax.bar(
        x,
        [cpu_pct],
        bar_width,
        bottom=[read_pct + write_pct],
        label="Computation",
        color="#FFF2CC",
        edgecolor="#D6DCE4",
    )

    # Annotate percentages
    ax.text(0, read_pct / 2, f"{read_pct:.1f}", ha="center", va="center", fontsize=10)
    ax.text(
        0,
        read_pct + write_pct / 2,
        f"{read_pct + write_pct:.1f}",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(0, 102, "100", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Normalized Execution Time")
    ax.set_xticks(x)
    ax.set_xticklabels([xtick_label])
    ax.set_xlabel("Value Size")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {output_path}")


# =====================================================================
# 5.  Multi-value-size support (scan a parent directory)
# =====================================================================


def find_log_file(run_dir: str) -> str | None:
    """Find the RocksDB LOG file in a run directory (metadata/)."""
    candidates = [
        "metadata/rocksdb_LOG_after_compact.txt",
        "metadata/rocksdb_LOG_after_fillrandom.txt",
        "metadata/rocksdb_LOG_after_mix.txt",
    ]
    for c in candidates:
        p = os.path.join(run_dir, c)
        if os.path.exists(p):
            return p
    return None


def plot_multi_value_breakdown(
    results: dict[int, pd.DataFrame], output_path: str
) -> None:
    """Stacked bar chart for multiple value sizes (GPComp Fig. 2 replica)."""
    if not HAS_MATPLOTLIB:
        print("  (matplotlib not available — skipping multi-value plot)")
        return

    value_sizes = sorted(results.keys())
    read_pcts, write_pcts, cpu_pcts = [], [], []

    for vs in value_sizes:
        df = results[vs]
        tw = df["wall_us"].sum()
        if tw > 0:
            read_pcts.append(df["read_io_us"].sum() / tw * 100)
            write_pcts.append(df["write_io_us"].sum() / tw * 100)
            cpu_pcts.append(df["cpu_us"].sum() / tw * 100)
        else:
            read_pcts.append(0)
            write_pcts.append(0)
            cpu_pcts.append(0)

    fig, ax = plt.subplots(figsize=(max(4, len(value_sizes) * 1.5), 5))
    x = range(len(value_sizes))
    bar_width = 0.6

    ax.bar(x, read_pcts, bar_width, label="Read", color="#4472C4")
    ax.bar(x, write_pcts, bar_width, bottom=read_pcts, label="Write", color="#A9D18E")
    bottoms = [r + w for r, w in zip(read_pcts, write_pcts)]
    ax.bar(
        x,
        cpu_pcts,
        bar_width,
        bottom=bottoms,
        label="Computation",
        color="#FFF2CC",
        edgecolor="#D6DCE4",
    )

    for i, vs in enumerate(value_sizes):
        ax.text(i, read_pcts[i] / 2, f"{read_pcts[i]:.1f}", ha="center", va="center", fontsize=9)
        cum = read_pcts[i] + write_pcts[i]
        ax.text(i, read_pcts[i] + write_pcts[i] / 2, f"{cum:.1f}", ha="center", va="center", fontsize=9)
        ax.text(i, 102, "100", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Normalized Execution Time")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{vs}B" for vs in value_sizes])
    ax.set_xlabel("Value Size")
    ax.set_ylim(0, 115)
    ax.set_title("Compaction Time Breakdown (Key Size = 16 bytes)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Multi-value plot saved to: {output_path}")


# =====================================================================
# 6.  Main
# =====================================================================


def process_single_run(run_dir: str, do_plot: bool, csv_path: str | None) -> pd.DataFrame | None:
    """Process a single run directory and return the DataFrame."""
    log_file = find_log_file(run_dir)
    if not log_file:
        print(f"ERROR: No RocksDB LOG found in {run_dir}/metadata/", file=sys.stderr)
        return None

    print(f"Parsing: {log_file}")
    events, start_info = parse_compaction_events(log_file)
    if not events:
        print(f"  WARNING: No compaction_finished events found in LOG.", file=sys.stderr)
        print(f"  Make sure --report_bg_io_stats=true was used.", file=sys.stderr)

        # Fall back to human-readable lines
        compacted_lines = parse_compacted_to_lines(log_file)
        if compacted_lines:
            print(f"  Found {len(compacted_lines)} 'compacted to:' summary lines.")
            for cl in compacted_lines:
                print(f"    Level {cl['output_level']}: {cl['rd_mb_per_sec']:.1f} MB/s rd, "
                      f"{cl['wr_mb_per_sec']:.1f} MB/s wr, "
                      f"{cl['mb_written']:.1f} MB written")
        return None

    df = events_to_dataframe(events, start_info)
    print(f"  Found {len(df)} compaction events.")
    if start_info:
        print(f"  Matched {sum(1 for j in df['job'] if j in start_info)}/{len(df)} "
              f"jobs to compaction_started info.")

    # Per-compaction detail
    print_per_compaction_table(df)

    # Aggregate summary
    print_aggregate_summary(df)

    # strace validation
    strace_path = os.path.join(run_dir, "strace_compact.log")
    if not os.path.exists(strace_path):
        strace_path = os.path.join(run_dir, "strace.log")
    strace_totals = parse_strace_log(strace_path)
    print_strace_summary(strace_totals)

    # CSV export
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"\n  CSV saved to: {csv_path}")

    # Plot
    if do_plot:
        plot_path = os.path.join(run_dir, "compaction_breakdown.png")
        plot_breakdown(df, plot_path)

    return df


def process_single_run_with_capture(
    run_dir: str,
    do_plot: bool,
    csv_path: str | None,
    summary_path: str | None,
) -> pd.DataFrame | None:
    """Run single-run processing while also persisting terminal output."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        df = process_single_run(run_dir, do_plot=do_plot, csv_path=csv_path)
    text = buffer.getvalue()
    print(text, end="")
    if summary_path:
        Path(summary_path).write_text(text, encoding="utf-8")
        print(f"  Summary saved to: {summary_path}")
    return df


def print_multi_value_summary(all_results: dict[int, pd.DataFrame]) -> None:
    print(f"\n{'═' * 60}")
    print("Multi-Value-Size Summary (GPComp Fig. 2 style)")
    print(f"{'═' * 60}")
    print(f"  {'Value Size':>12} {'Read%':>8} {'Write%':>8} {'Compute%':>10}")
    print(f"  {'─' * 42}")
    for vs in sorted(all_results.keys()):
        df = all_results[vs]
        tw = df["wall_us"].sum()
        if tw > 0:
            rp = df["read_io_us"].sum() / tw * 100
            wp = df["write_io_us"].sum() / tw * 100
            cp = df["cpu_us"].sum() / tw * 100
        else:
            rp = wp = cp = 0
        print(f"  {vs:>10}B {rp:>7.2f}% {wp:>7.2f}% {cp:>9.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Parse RocksDB compaction profile for IO/CPU breakdown."
    )
    parser.add_argument(
        "run_dir",
        help="Path to a single run directory, a base run under value_*/, or a parent containing value_*/ subdirs.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate bar chart(s).")
    parser.add_argument("--csv", default=None, help="Export per-compaction CSV.")
    parser.add_argument(
        "--all-value-sizes",
        action="store_true",
        help="Starting from one run directory, discover sibling value_*/ runs with the same relative suffix and generate a combined figure.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_path = Path(run_dir).resolve()
    multi_requested = args.all_value_sizes
    if not multi_requested and run_path.is_dir() and find_log_file(str(run_path)) is None:
        multi_requested = any(p.is_dir() and re.fullmatch(r"value_\d+", p.name) for p in run_path.iterdir())

    if multi_requested:
        summary_capture = io.StringIO()
        with redirect_stdout(summary_capture), redirect_stderr(summary_capture):
            root_dir, discovered_runs = resolve_multi_value_runs(run_dir)
            shared_output_dir = resolve_multi_value_output_dir(run_dir, root_dir)
            print(f"Scanning multi-value inputs from: {run_path}")
            if root_dir != run_path:
                print(f"Resolved multi-value root: {root_dir}")
            print(f"Shared output directory: {shared_output_dir}")

            if not discovered_runs:
                print("No matching sibling value_* run directories were found.", file=sys.stderr)
            else:
                print(f"Found value sizes: {[f'{vs}B' for vs in sorted(discovered_runs)]}")

            all_results: dict[int, pd.DataFrame] = {}
            for value_size, candidate in sorted(discovered_runs.items()):
                print(f"\n{'─' * 60}")
                print(f"Value size: {value_size}B  →  {candidate}")
                per_run_csv = None
                if args.csv:
                    csv_target = Path(args.csv)
                    if len(discovered_runs) > 1 and csv_target.suffix:
                        per_run_csv = str(csv_target.with_name(f"{csv_target.stem}_{value_size}B{csv_target.suffix}"))
                    else:
                        per_run_csv = args.csv
                summary_path = candidate / "compaction_breakdown_summary.txt"
                df = process_single_run_with_capture(
                    str(candidate),
                    do_plot=args.plot,
                    csv_path=per_run_csv,
                    summary_path=str(summary_path),
                )
                if df is not None:
                    all_results[value_size] = df

            if all_results:
                print_multi_value_summary(all_results)
                experiment_tag = resolve_experiment_tag(run_dir)
                tag_suffix = f"_{experiment_tag}" if experiment_tag else ""
                combined_plot_path = shared_output_dir / f"compaction_breakdown_all{tag_suffix}.png"
                if args.plot:
                    plot_multi_value_breakdown(all_results, str(combined_plot_path))

                combined_summary_path = shared_output_dir / f"compaction_breakdown_all_summary{tag_suffix}.txt"
                combined_summary_path.write_text(summary_capture.getvalue(), encoding="utf-8")

                if args.csv:
                    csv_target = Path(args.csv)
                    if len(discovered_runs) > 1 and csv_target.suffix:
                        combined_csv_path = csv_target.with_name(f"{csv_target.stem}_all{csv_target.suffix}")
                    else:
                        combined_csv_path = csv_target
                    combined = pd.concat(
                        [df.assign(value_size=vs) for vs, df in all_results.items()],
                        ignore_index=True,
                    )
                    combined.to_csv(combined_csv_path, index=False)
                    print(f"\n  Combined CSV saved to: {combined_csv_path}")

        captured_text = summary_capture.getvalue()
        print(captured_text, end="")
        if 'combined_summary_path' in locals() and all_results:
            combined_summary_path.write_text(captured_text, encoding="utf-8")
            print(f"  Multi-value summary saved to: {combined_summary_path}")
    else:
        summary_path = str(run_path / "compaction_breakdown_summary.txt") if run_path.is_dir() else None
        process_single_run_with_capture(run_dir, do_plot=args.plot, csv_path=args.csv, summary_path=summary_path)


if __name__ == "__main__":
    main()
