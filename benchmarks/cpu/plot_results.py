import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_results():
    # Adjust this path if your bench_results folder is located elsewhere
    results_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bench_results"))
    pattern = os.path.join(results_base, "fillrandom", "value_*", "comp_*", "*", "report.csv")
    files = glob.glob(pattern)

    all_data = []
    realtime_data = {} # {value_size: {label: df}}

    for f in files:
        # Extract metadata from folder structure
        parts = f.split(os.sep)
        try:
            value_size = int(parts[-4].split('_')[1])
            comp_threads = int(parts[-3].split('_')[1])
            label = f"CPU-{comp_threads}"
            
            # 1. Capture overall throughput for bar chart
            log_file = os.path.join(os.path.dirname(f), "db_bench.log")
            throughput = None
            if os.path.exists(log_file):
                with open(log_file, 'r') as log:
                    for line in log:
                        if "fillrandom" in line and ";" in line and "MB/s" in line:
                            throughput = float(line.split(";")[-1].split("MB/s")[0].strip())
                            break
            
            df = pd.read_csv(f)
            if throughput is None:
                col = [c for c in df.columns if 'mb_per_sec' in c.lower() or 'mb/s' in c.lower()]
                if col:
                    throughput = df[col[0]].iloc[0]
            
            if throughput is not None:
                all_data.append({
                    'Value Size': value_size,
                    'Threads': label,
                    'Throughput (MB/s)': throughput
                })
            
            # 2. Capture real-time data for line chart
            if value_size not in realtime_data:
                realtime_data[value_size] = {}
            realtime_data[value_size][label] = df

        except (IndexError, ValueError, pd.errors.EmptyDataError):
            continue

    if not all_data:
        print(f"No valid report files found in {results_base}")
        return

    # Plot 1: Average Throughput Bar Chart
    df_plot = pd.DataFrame(all_data).sort_values(['Value Size', 'Threads'])
    pivot_df = df_plot.pivot(index='Value Size', columns='Threads', values='Throughput (MB/s)')

    ax = pivot_df.plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title('RocksDB Throughput: CPU-based Compaction Scaling')
    plt.xlabel('Value Size (Bytes)')
    plt.ylabel('Throughput (MB/s)')
    plt.legend(title='Configuration')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    avg_png = os.path.join(os.path.dirname(__file__), "throughput_avg.png")
    plt.tight_layout()
    plt.savefig(avg_png)
    print(f"Average throughput graph saved to: {avg_png}")

    # Plot 2: Real-time Throughput Line Charts (matches Fig. 12 structure)
    value_sizes = sorted(realtime_data.keys())
    fig, axes = plt.subplots(len(value_sizes), 1, figsize=(12, 4 * len(value_sizes)))
    if len(value_sizes) == 1: axes = [axes]

    for i, vs in enumerate(value_sizes):
        ax = axes[i]
        for label, df in sorted(realtime_data[vs].items()):
            if 'secs_elapsed' in df.columns and 'interval_qps' in df.columns:
                ax.plot(df['secs_elapsed'], df['interval_qps'], label=label, alpha=0.8, linewidth=1)
        
        ax.set_title(f'Value Size: {vs} Bytes')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Throughput (Ops/s)')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)

    realtime_png = os.path.join(os.path.dirname(__file__), "throughput_realtime.png")
    plt.tight_layout()
    plt.savefig(realtime_png)
    print(f"Real-time throughput graph saved to: {realtime_png}")

if __name__ == "__main__":
    plot_results()
