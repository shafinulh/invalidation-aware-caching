#!/usr/bin/env python3
"""Parse [CACHE-EVICT] instrumentation from RocksDB LOG and produce CSV + plots.

Reads the compaction-triggered cache eviction instrumentation added to
PurgeObsoleteFiles() and generates:

  1. eviction_events.csv     – one row per compaction purge event
  2. eviction_per_file.csv   – one row per obsoleted file with cached blocks
  3. eviction_summary.txt    – human-readable summary statistics
  4. eviction_timeline.png   – eviction blocks & % over time
  5. eviction_histogram.png  – distribution of eviction % per event

Optionally overlays eviction events onto cache hit-rate / throughput plots
when --metrics-csv is supplied.

Usage:
    python3 analyze_cache_evictions.py /path/to/rocksdb_LOG_after_mix.txt
    python3 analyze_cache_evictions.py /path/to/run_dir
    python3 analyze_cache_evictions.py /path/to/run_dir --metrics-csv /path/to/metrics.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    sys.exit("error: pandas is required – pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("error: matplotlib is required – pip install matplotlib")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)")
_TS_FMT = "%Y/%m/%d-%H:%M:%S.%f"

# [CACHE-EVICT] Compaction purge: 2173 blocks from 11/11 obsolete files ...
# Total cache entries: 32570, eviction%: 6.67%, cache usage: 1073213200 / 1073741824 bytes (100.0%)
_SUMMARY_RE = re.compile(
    r"\[CACHE-EVICT\] Compaction purge: (\d+) blocks from (\d+)/(\d+) obsolete files.*?"
    r"Total cache entries: (\d+), eviction%: ([0-9.]+)%.*?"
    r"cache usage: (\d+) / (\d+) bytes \(([0-9.]+)%\)"
)

# [CACHE-EVICT] File #379: 147 data blocks in cache
_FILE_RE = re.compile(
    r"\[CACHE-EVICT\] File #(\d+): (\d+) data blocks in cache"
)


def _parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, _TS_FMT)


def parse_eviction_log(log_path: Path):
    """Parse [CACHE-EVICT] lines from a RocksDB LOG file.

    Returns (summary_rows, file_rows) where:
      summary_rows: list of dicts with per-compaction-purge stats
      file_rows:    list of dicts with per-file stats
    """
    base_ts: Optional[datetime] = None
    summary_rows = []
    file_rows = []
    current_event_id = 0

    with open(log_path) as f:
        for line in f:
            ts_match = _TS_RE.match(line)
            if not ts_match:
                continue
            ts = _parse_ts(ts_match.group(1))
            if base_ts is None:
                base_ts = ts

            elapsed = (ts - base_ts).total_seconds()

            sm = _SUMMARY_RE.search(line)
            if sm:
                current_event_id += 1
                summary_rows.append({
                    "event_id": current_event_id,
                    "secs_elapsed": round(elapsed, 3),
                    "blocks_evicted": int(sm.group(1)),
                    "files_with_cached_blocks": int(sm.group(2)),
                    "total_obsolete_files": int(sm.group(3)),
                    "total_cache_entries": int(sm.group(4)),
                    "eviction_pct": float(sm.group(5)),
                    "cache_usage_bytes": int(sm.group(6)),
                    "cache_capacity_bytes": int(sm.group(7)),
                    "cache_usage_pct": float(sm.group(8)),
                })
                continue

            fm = _FILE_RE.search(line)
            if fm:
                file_rows.append({
                    "event_id": current_event_id,
                    "secs_elapsed": round(elapsed, 3),
                    "file_number": int(fm.group(1)),
                    "cached_blocks": int(fm.group(2)),
                })

    return summary_rows, file_rows


# ---------------------------------------------------------------------------
# Analysis & CSV output
# ---------------------------------------------------------------------------

def write_csvs(summary_rows, file_rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    df_summary = pd.DataFrame(summary_rows)
    df_files = pd.DataFrame(file_rows)

    summary_path = out_dir / "eviction_events.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path} ({len(df_summary)} events)")

    files_path = out_dir / "eviction_per_file.csv"
    df_files.to_csv(files_path, index=False)
    print(f"  Wrote {files_path} ({len(df_files)} file entries)")

    return df_summary, df_files


def write_summary(df_summary: pd.DataFrame, df_files: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eviction_summary.txt"

    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  CACHE EVICTION INSTRUMENTATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total compaction purge events:  {len(df_summary)}\n")
        if len(df_summary) == 0:
            f.write("No eviction events found.\n")
            print(f"  Wrote {path}")
            return

        total_evicted = df_summary["blocks_evicted"].sum()
        f.write(f"Total blocks evicted:          {total_evicted}\n")
        f.write(f"Avg cache entries:             {df_summary['total_cache_entries'].mean():.0f}\n")
        f.write(f"Cache capacity:                {df_summary['cache_capacity_bytes'].iloc[0] / (1024**2):.0f} MB\n\n")

        f.write("--- Eviction % per compaction purge ---\n")
        pct = df_summary["eviction_pct"]
        f.write(f"  Min:    {pct.min():.2f}%\n")
        f.write(f"  Max:    {pct.max():.2f}%\n")
        f.write(f"  Mean:   {pct.mean():.2f}%\n")
        f.write(f"  Median: {pct.median():.2f}%\n")
        f.write(f"  Stdev:  {pct.std():.2f}%\n\n")

        f.write("--- Blocks evicted per compaction purge ---\n")
        blk = df_summary["blocks_evicted"]
        f.write(f"  Min:    {blk.min()}\n")
        f.write(f"  Max:    {blk.max()}\n")
        f.write(f"  Mean:   {blk.mean():.0f}\n")
        f.write(f"  Median: {blk.median():.0f}\n\n")

        f.write("--- Events by eviction severity ---\n")
        for threshold in [0, 1, 3, 5, 8, 10]:
            count = (pct > threshold).sum()
            f.write(f"  > {threshold:>2}%%:  {count:>3} events ({100.0 * count / len(pct):.0f}%)\n")
        f.write("\n")

        f.write("--- Obsolete files per purge ---\n")
        obs = df_summary["total_obsolete_files"]
        f.write(f"  Min:  {obs.min()}\n")
        f.write(f"  Max:  {obs.max()}\n")
        f.write(f"  Mean: {obs.mean():.1f}\n\n")

        if len(df_files) > 0:
            f.write("--- Per-file cached block counts ---\n")
            cb = df_files["cached_blocks"]
            f.write(f"  Files with cached blocks: {len(df_files)}\n")
            f.write(f"  Min blocks/file:  {cb.min()}\n")
            f.write(f"  Max blocks/file:  {cb.max()}\n")
            f.write(f"  Mean blocks/file: {cb.mean():.0f}\n")
            f.write(f"  Median blocks/file: {cb.median():.0f}\n\n")

        # Time-based analysis
        duration = df_summary["secs_elapsed"].max() - df_summary["secs_elapsed"].min()
        if duration > 0:
            f.write("--- Temporal analysis ---\n")
            f.write(f"  Observation window: {duration:.1f}s\n")
            f.write(f"  Purge event rate:   {len(df_summary) / duration * 60:.1f} events/min\n")
            f.write(f"  Block eviction rate: {total_evicted / duration:.0f} blocks/sec\n")

    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eviction_timeline(
    df_summary: pd.DataFrame,
    out_dir: Path,
    metrics_df: Optional[pd.DataFrame] = None,
    compaction_events: Optional[List[Tuple[float, float]]] = None,
):
    """Plot eviction events over time, optionally overlaid on hit-rate."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df_summary) == 0:
        return

    secs = df_summary["secs_elapsed"]

    # --- Figure 1: Eviction blocks + % over time ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    # Top: blocks evicted as stem plot
    ax1.stem(secs, df_summary["blocks_evicted"], linefmt="tab:red",
             markerfmt="o", basefmt="k-", label="Blocks evicted")
    ax1.set_ylabel("Blocks evicted")
    ax1.set_title("Block Cache Evictions Due to Compaction")
    ax1.legend(fontsize=8, loc="upper right")

    # Bottom: eviction %
    ax2.stem(secs, df_summary["eviction_pct"], linefmt="tab:orange",
             markerfmt="D", basefmt="k-", label="Eviction %")
    ax2.set_ylabel("% of cache evicted")
    ax2.set_xlabel("Elapsed time (s)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "eviction_timeline.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'eviction_timeline.png'}")

    # --- Figure 2: Histogram of eviction % ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_summary["eviction_pct"], bins=30, color="tab:orange",
            edgecolor="black", alpha=0.8)
    ax.set_xlabel("Eviction % per compaction purge")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Cache Eviction % per Compaction")
    ax.axvline(df_summary["eviction_pct"].mean(), color="red", linestyle="--",
               label=f"Mean: {df_summary['eviction_pct'].mean():.2f}%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "eviction_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'eviction_histogram.png'}")

    # --- Figure 3: If metrics available, overlay evictions on hit-rate ---
    if metrics_df is not None:
        _plot_hitrate_with_evictions(metrics_df, df_summary, out_dir, compaction_events)
        _plot_throughput_with_evictions(metrics_df, df_summary, out_dir, compaction_events)
        _plot_latency_with_evictions(metrics_df, df_summary, out_dir, compaction_events)


def _plot_hitrate_with_evictions(
    metrics_df: pd.DataFrame,
    eviction_df: pd.DataFrame,
    out_dir: Path,
    compaction_events: Optional[List[Tuple[float, float]]] = None,
):
    msecs = metrics_df["secs_elapsed"]
    hit_rate = metrics_df["cache_hit_rate"] * 100.0

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(msecs, hit_rate, linewidth=0.8, color="tab:blue", label="Cache hit rate")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Block-cache hit rate (%)")
    ax1.set_ylim(-2, 105)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(100))

    if compaction_events:
        for start, end in compaction_events:
            ax1.axvspan(start, end, alpha=0.08, color="tab:orange")

    # Overlay eviction events as red vertical bars with height = eviction %
    ax2 = ax1.twinx()
    ax2.bar(eviction_df["secs_elapsed"], eviction_df["eviction_pct"],
            width=0.3, color="tab:red", alpha=0.7, label="Eviction %")
    ax2.set_ylabel("Cache eviction % per compaction")
    ax2.set_ylim(0, max(15, eviction_df["eviction_pct"].max() * 1.2))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    ax1.set_title("Cache Hit Rate with Compaction Eviction Events")
    fig.tight_layout()
    fig.savefig(out_dir / "hitrate_with_evictions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'hitrate_with_evictions.png'}")


def _plot_throughput_with_evictions(
    metrics_df: pd.DataFrame,
    eviction_df: pd.DataFrame,
    out_dir: Path,
    compaction_events: Optional[List[Tuple[float, float]]] = None,
):
    msecs = metrics_df["secs_elapsed"]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(msecs, metrics_df["get_count"], linewidth=0.8, color="tab:green",
             label="Get ops/interval")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Read operations / interval")

    if compaction_events:
        for start, end in compaction_events:
            ax1.axvspan(start, end, alpha=0.08, color="tab:orange")

    ax2 = ax1.twinx()
    ax2.bar(eviction_df["secs_elapsed"], eviction_df["blocks_evicted"],
            width=0.3, color="tab:red", alpha=0.6, label="Blocks evicted")
    ax2.set_ylabel("Blocks evicted per compaction")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title("Read Throughput with Compaction Eviction Events")
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_with_evictions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'throughput_with_evictions.png'}")


def _plot_latency_with_evictions(
    metrics_df: pd.DataFrame,
    eviction_df: pd.DataFrame,
    out_dir: Path,
    compaction_events: Optional[List[Tuple[float, float]]] = None,
):
    msecs = metrics_df["secs_elapsed"]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(msecs, metrics_df["get_p95_us"], linewidth=0.8, color="tab:red",
             label="Get P95")
    ax1.plot(msecs, metrics_df["get_p50_us"], linewidth=0.6, color="tab:pink",
             alpha=0.6, label="Get P50")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Latency (µs)")

    if compaction_events:
        for start, end in compaction_events:
            ax1.axvspan(start, end, alpha=0.08, color="tab:orange")

    ax2 = ax1.twinx()
    ax2.bar(eviction_df["secs_elapsed"], eviction_df["eviction_pct"],
            width=0.3, color="tab:purple", alpha=0.5, label="Eviction %")
    ax2.set_ylabel("Cache eviction % per compaction")
    ax2.set_ylim(0, max(15, eviction_df["eviction_pct"].max() * 1.2))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title("Get Latency with Compaction Eviction Events")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_with_evictions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'latency_with_evictions.png'}")


# ---------------------------------------------------------------------------
# Compaction event parsing (reuse from plot_cache_metrics)
# ---------------------------------------------------------------------------

import json

_EVENT_LOG_RE = re.compile(r"EVENT_LOG_v1\s+(\{.*\})\s*$")

def parse_compaction_events(log_path: Path) -> List[Tuple[float, float]]:
    base_ts: Optional[datetime] = None
    pending: dict[int, float] = {}
    events: List[Tuple[float, float]] = []

    with open(log_path) as f:
        for line in f:
            ts_match = _TS_RE.match(line)
            if not ts_match:
                continue
            ts = _parse_ts(ts_match.group(1))
            if base_ts is None:
                base_ts = ts

            ev_match = _EVENT_LOG_RE.search(line)
            if not ev_match:
                continue
            try:
                payload = json.loads(ev_match.group(1))
            except json.JSONDecodeError:
                continue

            event = payload.get("event", "")
            job = payload.get("job")
            if job is None:
                continue

            elapsed = (ts - base_ts).total_seconds()
            if event == "compaction_started":
                pending[job] = elapsed
            elif event == "compaction_finished":
                start = pending.pop(job, None)
                if start is not None:
                    events.append((start, elapsed))

    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_log_file(path: Path) -> Optional[Path]:
    """Find the RocksDB LOG file from a path (file or directory)."""
    if path.is_file() and "rocksdb_LOG" in path.name:
        return path
    if path.is_dir():
        metadata = path / "metadata"
        if metadata.is_dir():
            for name in ("rocksdb_LOG_after_mix.txt", "rocksdb_LOG_after_load.txt"):
                candidate = metadata / name
                if candidate.is_file():
                    return candidate
            for candidate in sorted(metadata.glob("rocksdb_LOG_*.txt")):
                return candidate
    return None


def find_metrics_csv(path: Path) -> Optional[Path]:
    """Find metrics.csv near a run directory."""
    if path.is_file():
        candidate = path.parent / "metrics.csv"
        if candidate.is_file():
            return candidate
        candidate = path.parent.parent / "metrics.csv"
        if candidate.is_file():
            return candidate
        return None
    if path.is_dir():
        candidate = path / "metrics.csv"
        if candidate.is_file():
            return candidate
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", type=Path,
                   help="Path to RocksDB LOG file or run directory")
    p.add_argument("--metrics-csv", type=Path, default=None,
                   help="Path to metrics.csv (auto-detected if omitted)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: <run_dir>/eviction_analysis/)")
    return p.parse_args()


def main():
    args = parse_args()

    log_path = find_log_file(args.path)
    if log_path is None:
        sys.exit(f"error: could not find RocksDB LOG in {args.path}")

    print(f"Parsing eviction events from: {log_path}")

    summary_rows, file_rows = parse_eviction_log(log_path)
    if not summary_rows:
        sys.exit("No [CACHE-EVICT] events found in LOG file.")

    print(f"  Found {len(summary_rows)} compaction purge events, "
          f"{len(file_rows)} per-file entries")

    # Determine output directory
    if args.output_dir:
        out_dir = args.output_dir
    elif args.path.is_dir():
        out_dir = args.path / "eviction_analysis"
    else:
        out_dir = args.path.parent.parent / "eviction_analysis"

    # Write CSVs and summary
    df_summary, df_files = write_csvs(summary_rows, file_rows, out_dir)
    write_summary(df_summary, df_files, out_dir)

    # Load metrics if available
    metrics_csv = args.metrics_csv or find_metrics_csv(args.path)
    metrics_df = None
    if metrics_csv and metrics_csv.is_file():
        print(f"  Loading metrics from: {metrics_csv}")
        metrics_df = pd.read_csv(metrics_csv)

    # Parse compaction events from LOG for shading
    compaction_events = parse_compaction_events(log_path)

    # Generate plots
    plot_eviction_timeline(df_summary, out_dir, metrics_df, compaction_events)

    print(f"\nDone. All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
