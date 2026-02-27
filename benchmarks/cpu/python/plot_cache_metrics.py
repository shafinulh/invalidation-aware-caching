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
    python3 plot_cache_metrics.py --metrics-dir /path/to/run_dir   # finds metrics.csv recursively
    python3 plot_cache_metrics.py --metrics-dir /path/to/base --compare  # overlay multiple runs

Output:
    PNG figures written to --output-dir (default: alongside the metrics.csv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

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
    # Detect intervals where compaction was active.
    df["compaction_active"] = (
        (df["compact_read_bytes"] > 0) | (df["compact_write_bytes"] > 0)
    )
    return df


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

def plot_single_run(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    """Generate per-run plots for one metrics.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    secs = df["secs_elapsed"]

    # --- 1. Hit-rate + compaction shading ---
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(secs, df["hit_rate_pct"], linewidth=0.8, color="tab:blue", label="Cache hit rate")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Block-cache hit rate (%)")
    ax1.set_ylim(-2, 105)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(100))

    # Shade compaction intervals.
    _shade_compaction(ax1, df)

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
    _shade_compaction(ax, df)
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
    _shade_compaction(ax, df)
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
    _shade_compaction(ax, df)
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
    _shade_compaction(ax, df)
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
    _shade_compaction(ax, df)
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


def _shade_compaction(ax, df: pd.DataFrame) -> None:
    """Add light orange vertical spans where compaction was active."""
    active = df["compaction_active"].values
    secs = df["secs_elapsed"].values
    in_span = False
    start = 0.0
    for i, a in enumerate(active):
        if a and not in_span:
            start = secs[i]
            in_span = True
        elif not a and in_span:
            ax.axvspan(start, secs[i], alpha=0.10, color="tab:orange")
            in_span = False
    if in_span:
        ax.axvspan(start, secs[-1], alpha=0.10, color="tab:orange")


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

        out = args.output_dir if args.output_dir else csv_path.parent / "plots"
        plot_single_run(df, label, out)

    if args.compare and len(runs) > 1:
        comp_out = args.output_dir if args.output_dir else csv_paths[0].parent.parent / "comparison_plots"
        plot_comparison(runs, comp_out)


if __name__ == "__main__":
    main()
