import os
import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


def _load_env_file(env_file):
    values = {}
    if not os.path.exists(env_file):
        return values

    with open(env_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (
                (value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))
            ):
                value = value[1:-1]
            values[key] = value

    return values


def _extract_metadata_from_path(path_parts):
    value_size = None
    comp_threads = None
    write_ratio = None

    for part in path_parts:
        if part.startswith("value_"):
            value_size = int(part.split("_")[1])
        elif part.startswith("subcomp_") or part.startswith("comp_"):
            comp_threads = int(part.split("_")[1])
        elif part.startswith("write_"):
            write_ratio = int(part.split("_")[1])

    return value_size, comp_threads, write_ratio


def _extract_ops_per_sec_from_log(log_file):
    if not os.path.exists(log_file):
        return None

    last_ops = None
    pattern = re.compile(r"ops and \([0-9.]+,([0-9.]+)\) ops/second")
    with open(log_file, "r") as log:
        for line in log:
            match = pattern.search(line)
            if match:
                last_ops = float(match.group(1))
    return last_ops


def _extract_mb_per_sec_from_log(log_file, benchmark_name):
    if not os.path.exists(log_file):
        return None

    with open(log_file, "r") as log:
        for line in log:
            if benchmark_name in line and ";" in line and "MB/s" in line:
                return float(line.split(";")[-1].split("MB/s")[0].strip())
    return None


def _load_gpu_fillrandom(results_base):
    """Load GPU hook-bench CSVs and return per-value-size throughput data.

    Returns:
        gpu_avg  : {value_size: median MB/s}  (output_bytes / compact_range_us)
        gpu_reps : {value_size: DataFrame with columns [rep, time_s, throughput_mb_s]}
    """
    gpu_avg = {}
    gpu_reps = {}
    script_dir = Path(__file__).resolve().parent
    gpu_results_base = script_dir.parent.parent / "gpu" / "results" / "fillrandom_gpu"

    for vs_dir in sorted(gpu_results_base.glob("value_*")):
        try:
            vs = int(vs_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        hook_base = vs_dir / "rocksdb_hook"
        if not hook_base.is_dir():
            continue
        run_dirs = sorted([d for d in hook_base.iterdir() if d.is_dir()])
        if not run_dirs:
            continue
        csv_path = run_dirs[-1] / "gpu_compaction_hook.csv"
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(str(csv_path))
        except Exception:
            continue
        if "compact_range_us" not in df.columns or "output_bytes" not in df.columns:
            continue
        tput = (df["output_bytes"] / (1024 * 1024)) / (df["compact_range_us"] / 1e6)
        # Skip warm-up rep 0 for the average
        stable = tput[df["rep"] > 0] if len(df) > 1 else tput
        gpu_avg[vs] = float(stable.median())
        cum_time = df["compact_range_us"].cumsum() / 1e6
        gpu_reps[vs] = pd.DataFrame({"rep": df["rep"].values,
                                      "time_s": cum_time.values,
                                      "throughput_mb_s": tput.values})
    return gpu_avg, gpu_reps


def _plot_fillrandom(results_base, out_dir):
    pattern = os.path.join(results_base, "fillrandom", "value_*", "subcomp_*", "*", "report.csv")
    files = glob.glob(pattern)

    all_data = []
    realtime_data = {}  # {value_size: {label: df}}

    for report_file in files:
        parts = report_file.split(os.sep)
        try:
            value_size, comp_threads, _ = _extract_metadata_from_path(parts)
            if value_size is None or comp_threads is None:
                continue
            label = f"CPU-{comp_threads}"

            log_file = os.path.join(os.path.dirname(report_file), "db_bench.log")
            throughput_mb = _extract_mb_per_sec_from_log(log_file, "fillrandom")

            df = pd.read_csv(report_file)
            if throughput_mb is None and "interval_qps" in df.columns:
                throughput_mb = df["interval_qps"].mean() * value_size / (1024 * 1024)

            if throughput_mb is not None:
                all_data.append(
                    {
                        "Value Size": value_size,
                        "Threads": label,
                        "Throughput (MB/s)": throughput_mb,
                    }
                )

            realtime_data.setdefault(value_size, {})[label] = df
        except (IndexError, ValueError, pd.errors.EmptyDataError):
            continue

    # Load GPU results and add a "GPU" bar for each value size
    gpu_avg, gpu_reps = _load_gpu_fillrandom(results_base)
    for vs, tput in sorted(gpu_avg.items()):
        all_data.append({"Value Size": vs, "Threads": "GPU", "Throughput (MB/s)": tput})

    if not all_data:
        print(f"No valid fillrandom report files found in {results_base}")
        return

    df_plot = pd.DataFrame(all_data).sort_values(["Value Size", "Threads"])
    pivot_df = df_plot.pivot_table(
        index="Value Size", columns="Threads", values="Throughput (MB/s)", aggfunc="mean"
    )
    # Reorder columns so CPU-* appear before GPU
    cpu_cols = sorted([c for c in pivot_df.columns if c.startswith("CPU")])
    gpu_cols = [c for c in pivot_df.columns if not c.startswith("CPU")]
    pivot_df = pivot_df[cpu_cols + gpu_cols]

    pivot_df.plot(kind="bar", figsize=(10, 6), width=0.8)
    plt.title("RocksDB Throughput: FillRandom CPU vs GPU Compaction")
    plt.xlabel("Value Size (Bytes)")
    plt.ylabel("Throughput (MB/s)")
    plt.legend(title="Configuration")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    avg_png = os.path.join(out_dir, "throughput_avg.png")
    plt.tight_layout()
    plt.savefig(avg_png)
    plt.close()
    print(f"FillRandom average throughput graph saved to: {avg_png}")

    value_sizes = sorted(realtime_data.keys())
    fig, axes = plt.subplots(len(value_sizes), 1, figsize=(12, 4 * len(value_sizes)))
    if len(value_sizes) == 1:
        axes = [axes]

    for i, value_size in enumerate(value_sizes):
        ax = axes[i]
        for label, df in sorted(realtime_data[value_size].items()):
            if "secs_elapsed" in df.columns and "interval_qps" in df.columns:
                # Convert ops/s → MB/s for a unified y-axis
                mb_s = df["interval_qps"] * value_size / (1024 * 1024)
                ax.plot(df["secs_elapsed"], mb_s, label=label, alpha=0.8, linewidth=1)

        # GPU: per-rep throughput as a dashed line with markers;
        # cumulative time is only a few seconds so we also draw a horizontal
        # median reference that spans the full chart.
        if value_size in gpu_reps:
            grdf = gpu_reps[value_size]
            ax.plot(
                grdf["time_s"], grdf["throughput_mb_s"],
                label="GPU (per-rep)", color="purple",
                linewidth=1.5, marker="o", markersize=5, linestyle="-",
            )
        if value_size in gpu_avg:
            ax.axhline(
                gpu_avg[value_size], color="purple", linestyle="--",
                linewidth=1.0, alpha=0.6,
                label=f"GPU median ({gpu_avg[value_size]:.1f} MB/s)",
            )

        ax.set_title(f"FillRandom Value Size: {value_size} Bytes")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Throughput (MB/s)")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, linestyle=":", alpha=0.6)

    realtime_png = os.path.join(out_dir, "throughput_realtime.png")
    plt.tight_layout()
    plt.savefig(realtime_png)
    plt.close()
    print(f"FillRandom real-time throughput graph saved to: {realtime_png}")


def _plot_readwritemix(results_base, out_dir):
    new_pattern = os.path.join(
        results_base,
        "readwritemix",
        "value_*",
        "comp_*",
        "write_*",
        "*",
        "mix_report.csv",
    )
    old_pattern = os.path.join(
        results_base,
        "readwritemix",
        "value_*",
        "write_*",
        "comp_*",
        "*",
        "mix_report.csv",
    )
    files = sorted(set(glob.glob(new_pattern) + glob.glob(old_pattern)))

    all_data = []
    realtime_data = {}  # {(value_size, write_ratio): {label: df}}

    for report_file in files:
        parts = report_file.split(os.sep)
        try:
            value_size, comp_threads, write_ratio = _extract_metadata_from_path(parts)
            if value_size is None or comp_threads is None or write_ratio is None:
                continue
            label = f"CPU-{comp_threads}"

            df = pd.read_csv(report_file)
            throughput_ops = None
            if "interval_qps" in df.columns:
                throughput_ops = df["interval_qps"].mean()

            if throughput_ops is None:
                log_file = os.path.join(os.path.dirname(report_file), "mix.log")
                throughput_ops = _extract_ops_per_sec_from_log(log_file)

            if throughput_ops is not None:
                all_data.append(
                    {
                        "Value Size": value_size,
                        "Write Ratio": write_ratio,
                        "Threads": label,
                        "Throughput (Ops/s)": throughput_ops,
                    }
                )

            realtime_data.setdefault((value_size, write_ratio), {})[label] = df
        except (IndexError, ValueError, pd.errors.EmptyDataError):
            continue

    if not all_data:
        print(f"No valid readwritemix report files found in {results_base}")
        return

    df_plot = pd.DataFrame(all_data).sort_values(["Value Size", "Write Ratio", "Threads"])
    pivot_df = df_plot.pivot_table(
        index=["Value Size", "Write Ratio"],
        columns="Threads",
        values="Throughput (Ops/s)",
        aggfunc="mean",
    )

    pivot_df.plot(kind="bar", figsize=(12, 6), width=0.85)
    plt.title("RocksDB Throughput: ReadWriteMix CPU Compaction Scaling")
    plt.xlabel("Value Size / Write Ratio (%)")
    plt.ylabel("Throughput (Ops/s)")
    plt.legend(title="Configuration")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    avg_png = os.path.join(out_dir, "readwritemix_throughput_avg.png")
    plt.tight_layout()
    plt.savefig(avg_png)
    plt.close()
    print(f"ReadWriteMix average throughput graph saved to: {avg_png}")

    configs = sorted(realtime_data.keys())
    fig, axes = plt.subplots(len(configs), 1, figsize=(12, 3.5 * len(configs)))
    if len(configs) == 1:
        axes = [axes]

    for i, (value_size, write_ratio) in enumerate(configs):
        ax = axes[i]
        for label, df in sorted(realtime_data[(value_size, write_ratio)].items()):
            if "secs_elapsed" in df.columns and "interval_qps" in df.columns:
                ax.plot(df["secs_elapsed"], df["interval_qps"], label=label, alpha=0.85, linewidth=1)

        ax.set_title(f"ReadWriteMix Value Size: {value_size} Bytes, Write Ratio: {write_ratio}%")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Throughput (Ops/s)")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, linestyle=":", alpha=0.6)

    realtime_png = os.path.join(out_dir, "readwritemix_throughput_realtime.png")
    plt.tight_layout()
    plt.savefig(realtime_png)
    plt.close()
    print(f"ReadWriteMix real-time throughput graph saved to: {realtime_png}")

def _resolve_results_paths():
    script_dir = Path(__file__).resolve().parent
    env_file = script_dir.parent / "config" / ".env.local"
    env_values = _load_env_file(str(env_file))

    output_dir = env_values.get("OUTPUT_DIR")
    if not output_dir:
        raise RuntimeError(
            f"OUTPUT_DIR is not set in {env_file}. "
            "Set OUTPUT_DIR in benchmarks/cpu/config/.env.local before plotting."
        )

    results_base = os.path.abspath(os.path.expandvars(os.path.expanduser(output_dir)))
    out_dir = os.path.join(results_base, "plots")
    os.makedirs(out_dir, exist_ok=True)
    return results_base, out_dir, str(env_file)


def plot_results():
    results_base, out_dir, env_file = _resolve_results_paths()
    print(f"Using results directory from {env_file}: {results_base}")
    print(f"Saving plots to: {out_dir}")
    _plot_fillrandom(results_base, out_dir)
    _plot_readwritemix(results_base, out_dir)


if __name__ == "__main__":
    plot_results()
