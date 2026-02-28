#!/usr/bin/env python3
"""Plot block-cache, latency, and compaction metrics from metrics.csv files.

Reads metrics.csv files produced by the MetricsCollectorAgent in db_bench
(--metrics_interval_ms) and generates time-series plots showing:

  1. Block-cache hit rate over time (with compaction events shaded)
  2. Get P95 latency over time
  3. Interval throughput (get_count / interval) over time
  4. Cache hit/miss breakdown (data / index / filter)

Usage:
    python3 plot_cache_metrics.py --metrics-csv /path/to/metrics.csv
    python3 plot_cache_metrics.py --metrics-csv /path/to/metrics.csv --compaction-source csv
    python3 plot_cache_metrics.py --metrics-dir /path/to/run_dir   # finds metrics.csv recursively
    python3 plot_cache_metrics.py --metrics-dir /path/to/base --compare  # overlay multiple runs

Output:
    PNG figures written to --output-dir (default: alongside the metrics.csv).
"""

from __future__ import annotations

import argparse
import json
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
# Helpers
# ---------------------------------------------------------------------------

def find_metrics_csvs(base: Path) -> List[Path]:
    """Recursively find all metrics.csv files under *base*."""
    if base.is_file() and base.name == "metrics.csv":
        return [base]
    return sorted(base.rglob("metrics.csv"))


def load_metrics(path: Path) -> pd.DataFrame:
    """Load a single metrics.csv into a DataFrame."""
    df = pd.read_csv(path)
    # Derive convenience columns.
    df["cache_total"] = df["cache_hit"] + df["cache_miss"]
    df["hit_rate_pct"] = df["cache_hit_rate"] * 100.0
    return df


def compaction_events_from_csv(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """Derive (start_secs, end_secs) compaction intervals from the metrics CSV.

    A compaction is considered active for any interval where
    compact_read_bytes > 0 or compact_write_bytes > 0.  Contiguous
    active intervals are merged into a single (start, end) span.
    """
    active = ((df["compact_read_bytes"] > 0) | (df["compact_write_bytes"] > 0)).values
    secs = df["secs_elapsed"].values
    events: List[Tuple[float, float]] = []
    in_span = False
    start = 0.0
    for i, a in enumerate(active):
        if a and not in_span:
            start = float(secs[i])
            in_span = True
        elif not a and in_span:
            events.append((start, float(secs[i])))
            in_span = False
    if in_span:
        events.append((start, float(secs[-1])))
    return events


def _label_from_path(csv_path: Path) -> str:
    """Derive a short human-readable label from the directory structure."""
    parts = csv_path.parent.parts
    keep = []
    for p in parts:
        if p.startswith("subcomp_") or p.startswith("write_") or p.startswith("bgcomp_"):
            keep.append(p)
        # Include the run-id (last dir component before metrics.csv).
    if not keep:
        keep.append(csv_path.parent.name)
    return "/".join(keep)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_single_run(
    df: pd.DataFrame,
    label: str,
    out_dir: Path,
    compaction_events: Optional[List[Tuple[float, float]]] = None,
    compaction_source: str = "log",
) -> None:
    """Generate per-run plots for one metrics.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    secs = df["secs_elapsed"]
    if compaction_events is None:
        compaction_events = []

    # --- 1. Hit-rate + compaction shading ---
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(secs, df["hit_rate_pct"], linewidth=0.8, color="tab:blue", label="Cache hit rate")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Block-cache hit rate (%)")
    ax1.set_ylim(-2, 105)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(100))

    # Shade compaction intervals.
    _shade_compaction(ax1, compaction_events)

    # Overlay compaction bytes on secondary axis.
    ax2 = ax1.twinx()
    compact_mb = (df["compact_read_bytes"] + df["compact_write_bytes"]) / (1024 * 1024)
    ax2.fill_between(secs, 0, compact_mb, alpha=0.15, color="tab:orange", label="Compaction MB")
    ax2.set_ylabel("Compaction I/O (MB / interval)")
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    ax1.set_title(f"Block-Cache Hit Rate – {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "cache_hit_rate.png", dpi=150)
    plt.close(fig)

    # --- 2. Get P95 latency ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(secs, df["get_p95_us"], linewidth=0.8, color="tab:red", label="Get P95")
    ax.plot(secs, df["get_p50_us"], linewidth=0.6, color="tab:pink", alpha=0.6, label="Get P50")
    _shade_compaction(ax, compaction_events)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Latency (µs)")
    ax.legend(fontsize=8)
    ax.set_title(f"Get Latency – {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "get_latency.png", dpi=150)
    plt.close(fig)

    # --- 3. Interval throughput (ops/interval) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(secs, df["get_count"], linewidth=0.8, color="tab:green", label="Get ops/interval")
    _shade_compaction(ax, compaction_events)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Operations")
    ax.legend(fontsize=8)
    ax.set_title(f"Read Throughput – {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "read_throughput.png", dpi=150)
    plt.close(fig)

    # --- 4. Cache hit & miss combined ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(secs, df["cache_hit"], linewidth=0.8, color="tab:blue", label="Total hit")
    ax.plot(secs, df["cache_miss"], linewidth=0.8, color="tab:red", label="Total miss")
    _shade_compaction(ax, compaction_events)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Count / interval")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Cache Hit vs Miss – {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "cache_hit_miss.png", dpi=150)
    plt.close(fig)

    # --- 5. Cache hits breakdown (data / index / filter) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(secs, df["data_hit"], linewidth=0.8, color="tab:blue", label="Data hit")
    ax.plot(secs, df["index_hit"], linewidth=0.8, color="tab:cyan", label="Index hit")
    ax.plot(secs, df["filter_hit"], linewidth=0.8, color="tab:green", label="Filter hit")
    _shade_compaction(ax, compaction_events)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Hits / interval")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Cache Hits Breakdown – {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "cache_hits_breakdown.png", dpi=150)
    plt.close(fig)

    # --- 6. Cache misses breakdown (data / index / filter) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(secs, df["data_miss"], linewidth=0.8, color="tab:red", label="Data miss")
    ax.plot(secs, df["index_miss"], linewidth=0.8, color="tab:orange", label="Index miss")
    ax.plot(secs, df["filter_miss"], linewidth=0.8, color="tab:brown", label="Filter miss")
    _shade_compaction(ax, compaction_events)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Misses / interval")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Cache Misses Breakdown – {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "cache_misses_breakdown.png", dpi=150)
    plt.close(fig)

    # --- 7. Write stalls ---
    if df["stall_micros"].sum() > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(secs, df["stall_micros"] / 1000.0, width=0.8, color="tab:purple", alpha=0.7)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Stall (ms / interval)")
        ax.set_title(f"Write Stalls – {label}")
        fig.tight_layout()
        fig.savefig(out_dir / "write_stalls.png", dpi=150)
        plt.close(fig)

    # --- 8. Compaction detection: CSV bytes vs LOG events ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    # Top panel: compact_read_bytes and compact_write_bytes from metrics CSV.
    compact_read_mb  = df["compact_read_bytes"]  / (1024 * 1024)
    compact_write_mb = df["compact_write_bytes"] / (1024 * 1024)
    ax_top.fill_between(secs, 0, compact_read_mb,  alpha=0.55, color="tab:blue",
                        label="compact_read_bytes (MB/interval)")
    ax_top.fill_between(secs, 0, compact_write_mb, alpha=0.45, color="tab:orange",
                        label="compact_write_bytes (MB/interval)")
    ax_top.set_ylabel("Compaction I/O (MB / interval)")
    ax_top.legend(fontsize=8, loc="upper left")
    source_label = "LOG events" if compaction_source == "log" else "CSV bytes > 0"
    ax_top.set_title(
        f"Compaction Detection ({source_label}) vs CSV bytes – {label}"
    )

    # Overlay active-source start (red) / end (blue) lines on top panel.
    for start, end in compaction_events:
        ax_top.axvline(start, color="red",  linewidth=0.7, alpha=0.6)
        ax_top.axvline(end,   color="blue", linewidth=0.7, alpha=0.6)

    # Bottom panel: binary active/inactive.
    # Grey fill = CSV bytes > 0 (always shown for reference).
    csv_active = ((df["compact_read_bytes"] > 0) | (df["compact_write_bytes"] > 0)).astype(float)
    ax_bot.fill_between(secs, 0, csv_active, step="post", alpha=0.5,
                        color="tab:gray", label="CSV: bytes > 0")

    # Active-source spans drawn as a band.
    for start, end in compaction_events:
        ax_bot.axvspan(start, end, ymin=0.55, ymax=0.95, alpha=0.55,
                       color="tab:orange", label="_nolegend_")
        ax_bot.axvline(start, color="red",  linewidth=0.7, alpha=0.7)
        ax_bot.axvline(end,   color="blue", linewidth=0.7, alpha=0.7)

    # Custom legend entries for the bottom panel.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    span_label = "LOG: compaction span" if compaction_source == "log" else "CSV: compaction span"
    start_label = "LOG: compaction_started" if compaction_source == "log" else "CSV: first byte > 0"
    end_label   = "LOG: compaction_finished" if compaction_source == "log" else "CSV: last byte > 0"
    legend_handles = [
        Patch(facecolor="tab:gray",   alpha=0.5,  label="CSV: compact bytes > 0"),
        Patch(facecolor="tab:orange", alpha=0.55, label=span_label),
        Line2D([0], [0], color="red",  linewidth=1, label=start_label),
        Line2D([0], [0], color="blue", linewidth=1, label=end_label),
    ]
    ax_bot.legend(handles=legend_handles, fontsize=8, loc="upper left")
    ax_bot.set_yticks([0, 1])
    ax_bot.set_yticklabels(["idle", "active"])
    ax_bot.set_ylim(-0.05, 1.25)
    ax_bot.set_xlabel("Elapsed time (s)")
    ax_bot.set_ylabel("Active")

    fig.savefig(out_dir / "compaction_detection_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved plots → {out_dir}/")


def plot_comparison(runs: List[tuple[str, pd.DataFrame]], out_dir: Path) -> None:
    """Overlay hit-rate and P95 latency from multiple runs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Hit rate comparison ---
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in runs:
        ax.plot(df["secs_elapsed"], df["hit_rate_pct"], linewidth=0.8, label=label)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Block-cache hit rate (%)")
    ax.set_ylim(-2, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.legend(fontsize=8)
    ax.set_title("Block-Cache Hit Rate Comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "cache_hit_rate_comparison.png", dpi=150)
    plt.close(fig)

    # --- P95 latency comparison ---
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in runs:
        ax.plot(df["secs_elapsed"], df["get_p95_us"], linewidth=0.8, label=label)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Get P95 latency (µs)")
    ax.legend(fontsize=8)
    ax.set_title("Get P95 Latency Comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "get_p95_comparison.png", dpi=150)
    plt.close(fig)

    print(f"  Saved comparison plots → {out_dir}/")


# ---------------------------------------------------------------------------
# RocksDB LOG parsing – compaction events
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)")
_EVENT_LOG_RE = re.compile(r"EVENT_LOG_v1\s+(\{.*\})\s*$")
_TS_FMT = "%Y/%m/%d-%H:%M:%S.%f"


def _parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, _TS_FMT)


def parse_compaction_events(log_path: Path) -> List[Tuple[float, float]]:
    """Parse compaction start/finish times from a RocksDB LOG file.

    Returns a list of (start_secs, end_secs) tuples relative to the
    first timestamp in the LOG file.
    """
    base_ts: Optional[datetime] = None
    # job_id -> start_secs
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


def _find_rocksdb_log(csv_path: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate rocksdb_LOG_after_mix.txt next to a metrics.csv."""
    if explicit and explicit.is_file():
        return explicit
    metadata = csv_path.parent / "metadata"
    # Try common naming conventions.
    for name in ("rocksdb_LOG_after_mix.txt", "rocksdb_LOG_after_load.txt"):
        candidate = metadata / name
        if candidate.is_file():
            return candidate
    # Fallback: any rocksdb_LOG_*.txt.
    for candidate in sorted(metadata.glob("rocksdb_LOG_*.txt")):
        return candidate
    return None


def _shade_compaction(
    ax,
    compaction_events: List[Tuple[float, float]],
) -> None:
    """Draw compaction spans: red line at start, blue line at end, orange fill."""
    for start, end in compaction_events:
        ax.axvline(start, color="red", linewidth=0.5, alpha=0.5)
        ax.axvline(end, color="blue", linewidth=0.5, alpha=0.5)
        ax.axvspan(start, end, alpha=0.10, color="tab:orange")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--metrics-csv", type=Path, help="Path to a single metrics.csv file.")
    grp.add_argument("--metrics-dir", type=Path, help="Directory to search recursively for metrics.csv files.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory for plots.  Default: alongside each metrics.csv.")
    p.add_argument("--compare", action="store_true",
                   help="When using --metrics-dir, overlay all found runs in comparison plots.")
    p.add_argument("--rocksdb-log", type=Path, default=None,
                   help="Path to rocksdb LOG file.  Auto-detected from metadata/ if omitted.")
    p.add_argument("--compaction-source", choices=["csv", "log"], default="log",
                   help="How to determine compaction intervals: "
                        "'log' (default) parses compaction_started/finished from the RocksDB LOG; "
                        "'csv' uses compact_read_bytes > 0 from the metrics CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Discover metrics files.
    if args.metrics_csv:
        csv_paths = [args.metrics_csv]
    else:
        csv_paths = find_metrics_csvs(args.metrics_dir)

    if not csv_paths:
        sys.exit("No metrics.csv files found.")

    print(f"Found {len(csv_paths)} metrics file(s).")

    runs: List[tuple[str, pd.DataFrame]] = []
    for csv_path in csv_paths:
        label = _label_from_path(csv_path)
        df = load_metrics(csv_path)
        runs.append((label, df))

        # Resolve compaction events from the chosen source.
        compaction_events: List[Tuple[float, float]] = []
        if args.compaction_source == "log":
            log_path = _find_rocksdb_log(csv_path, args.rocksdb_log)
            if log_path:
                compaction_events = parse_compaction_events(log_path)
                print(f"  [log]  Parsed {len(compaction_events)} compaction events from {log_path.name}")
            else:
                print(f"  Warning: no RocksDB LOG found for {csv_path}; falling back to CSV source.")
                compaction_events = compaction_events_from_csv(df)
                print(f"  [csv]  Derived {len(compaction_events)} compaction spans from compact bytes")
        else:
            compaction_events = compaction_events_from_csv(df)
            print(f"  [csv]  Derived {len(compaction_events)} compaction spans from compact bytes")

        out = args.output_dir if args.output_dir else csv_path.parent / "plots"
        plot_single_run(df, label, out, compaction_events, args.compaction_source)

    if args.compare and len(runs) > 1:
        comp_out = args.output_dir if args.output_dir else csv_paths[0].parent.parent / "comparison_plots"
        plot_comparison(runs, comp_out)


if __name__ == "__main__":
    main()
