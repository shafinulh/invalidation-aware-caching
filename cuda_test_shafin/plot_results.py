import os
import glob
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


DEFAULT_BASE_DIR = os.environ.get('GPCOMP_RESULTS_DIR', 'sweep_results')
DEFAULT_GRAPH_DIR = os.environ.get('GPCOMP_GRAPHS_DIR', 'graphs')

MODE_ORDER = [
    'cpu_baseline',
    'q_paper_with_plan',
    'q_paper_without_plan',
    'c_paper_with_plan',
    'c_paper_without_plan',
]

MODE_LABELS = {
    'cpu_baseline': 'CPU Baseline',
    'q_paper_with_plan': 'Q (With Plan)',
    'q_paper_without_plan': 'Q (Without Plan)',
    'c_paper_with_plan': 'C (With Plan)',
    'c_paper_without_plan': 'C (Without Plan)',
}

MODE_COLORS = {
    'cpu_baseline': '#f4e7c5',
    'q_paper_with_plan': '#c6dbef',
    'q_paper_without_plan': '#9ecae1',
    'c_paper_with_plan': '#c7e9c0',
    'c_paper_without_plan': '#74c476',
}


def extract_label(sweep_path):
    """Extract label from sweep directory name like sweep_8mb-sst_4sst."""
    name = os.path.basename(os.path.normpath(sweep_path))
    m = re.match(r'sweep_(.+)$', name)
    if m:
        return m.group(1)
    return name


def parse_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    val_match = re.search(r'value bytes:\s*(\d+)', content)
    key_match = re.search(r'key bytes:\s*(\d+)', content)
    mode_match = re.search(r'gpu mode:\s*([^\n]+)', content)

    val_bytes = int(val_match.group(1)) if val_match else 0
    key_bytes = int(key_match.group(1)) if key_match else 16
    mode = mode_match.group(1).strip() if mode_match else ''

    # Throughput values
    cpu_wall = re.search(r'CPU total \(Wall\):\s*min=([\d.]+)\s*ms', content)
    gpu_wall = re.search(r'GPU total \(Wall\):\s*min=([\d.]+)\s*ms', content)
    cpu_wall_min = float(cpu_wall.group(1)) if cpu_wall else 0.0
    gpu_wall_min = float(gpu_wall.group(1)) if gpu_wall else 0.0

    output_bytes_match = re.search(r'output bytes\s*(\d+)', content)
    output_bytes = float(output_bytes_match.group(1)) if output_bytes_match else 0.0

    ops = output_bytes / (key_bytes + val_bytes) if (key_bytes + val_bytes) > 0 else 0
    cpu_throughput = (ops / (cpu_wall_min / 1000.0)) if cpu_wall_min > 0 else 0
    gpu_throughput = (ops / (gpu_wall_min / 1000.0)) if gpu_wall_min > 0 else 0

    # Pipeline CPU utilization from normal benchmark runs
    cpu_pipeline_util = re.search(r'CPU pipeline \(Wall\):\s*([\d.]+)\s*ms\s*\(CPU-Time\):\s*([\d.]+)\s*ms\s*utilization:\s*([\d.]+)%', content)
    gpu_pipeline_util = re.search(r'GPU pipeline \(Wall\):\s*([\d.]+)\s*ms\s*\(CPU-Time\):\s*([\d.]+)\s*ms\s*utilization:\s*([\d.]+)%', content)

    cpu_pipe_util = float(cpu_pipeline_util.group(3)) if cpu_pipeline_util else 0.0
    gpu_pipe_util = float(gpu_pipeline_util.group(3)) if gpu_pipeline_util else 0.0

    io_profiles = re.findall(
        r'I/O profile:\s*input bytes\s*\d+\s*estimated SSD read BW\s*([\d.]+)\s*MB/s\s*estimated SSD write BW\s*([\d.]+)\s*MB/s',
        content,
    )
    cpu_read_bw = float(io_profiles[0][0]) if len(io_profiles) >= 1 else 0.0
    cpu_write_bw = float(io_profiles[0][1]) if len(io_profiles) >= 1 else 0.0
    gpu_read_bw = float(io_profiles[1][0]) if len(io_profiles) >= 2 else 0.0
    gpu_write_bw = float(io_profiles[1][1]) if len(io_profiles) >= 2 else 0.0

    return {
        'value_bytes': val_bytes,
        'mode': mode,
        'cpu_throughput': cpu_throughput,
        'gpu_throughput': gpu_throughput,
        'cpu_util': cpu_pipe_util,
        'gpu_util': gpu_pipe_util,
        'cpu_read_bw': cpu_read_bw,
        'cpu_write_bw': cpu_write_bw,
        'gpu_read_bw': gpu_read_bw,
        'gpu_write_bw': gpu_write_bw,
    }


def mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def plot_grouped_bars(ax, x, width, series_map, sorted_vals, ylabel, title, ylim=None):
    offsets = np.linspace(-2.0, 2.0, len(MODE_ORDER)) * width
    for idx, mode in enumerate(MODE_ORDER):
        series = [series_map[mode].get(v, 0.0) for v in sorted_vals]
        ax.bar(x + offsets[idx], series, width, label=MODE_LABELS[mode],
               color=MODE_COLORS[mode], edgecolor='black')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Value Size (Bytes)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_vals)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)


def main(sweep_dir, graph_dir, label=None):
    print(f"Reading logs from: {sweep_dir}")

    if label is None:
        label = extract_label(sweep_dir)

    log_files = sorted(glob.glob(os.path.join(sweep_dir, 'result_val*B_*.log')))
    gpu_metrics = {mode: defaultdict(dict) for mode in MODE_ORDER if mode != 'cpu_baseline'}
    cpu_samples = defaultdict(lambda: defaultdict(list))

    for file_path in log_files:
        parsed = parse_log(file_path)
        val = parsed['value_bytes']
        mode = parsed['mode']
        if val <= 0 or mode not in gpu_metrics:
            continue

        gpu_metrics[mode][val]['throughput'] = parsed['gpu_throughput']
        gpu_metrics[mode][val]['util'] = parsed['gpu_util']
        gpu_metrics[mode][val]['read_bw'] = parsed['gpu_read_bw']
        gpu_metrics[mode][val]['write_bw'] = parsed['gpu_write_bw']

        cpu_samples[val]['throughput'].append(parsed['cpu_throughput'])
        cpu_samples[val]['util'].append(parsed['cpu_util'])
        cpu_samples[val]['read_bw'].append(parsed['cpu_read_bw'])
        cpu_samples[val]['write_bw'].append(parsed['cpu_write_bw'])

    # Prepare data for plotting
    sorted_vals = sorted(cpu_samples.keys())
    if not sorted_vals:
        print("No valid plotted data.")
        return

    x = np.arange(len(sorted_vals))
    width = 0.14

    os.makedirs(graph_dir, exist_ok=True)

    throughput_series = {
        'cpu_baseline': {v: mean_or_zero(cpu_samples[v]['throughput']) for v in sorted_vals},
        'q_paper_with_plan': {v: gpu_metrics['q_paper_with_plan'].get(v, {}).get('throughput', 0.0) for v in sorted_vals},
        'q_paper_without_plan': {v: gpu_metrics['q_paper_without_plan'].get(v, {}).get('throughput', 0.0) for v in sorted_vals},
        'c_paper_with_plan': {v: gpu_metrics['c_paper_with_plan'].get(v, {}).get('throughput', 0.0) for v in sorted_vals},
        'c_paper_without_plan': {v: gpu_metrics['c_paper_without_plan'].get(v, {}).get('throughput', 0.0) for v in sorted_vals},
    }

    # --- PLOT 1: THROUGHPUT ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_grouped_bars(ax, x, width, throughput_series, sorted_vals,
                      'Throughput (Ops/s)', f'Throughput vs Value Size ({label})')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)

    plt.tight_layout()
    tp_out_path = os.path.join(graph_dir, f'throughput_{label}_sweep.png')
    plt.savefig(tp_out_path)
    print(f"Throughput graph saved to: {tp_out_path}")
    plt.close()

    # --- PLOT 2: PIPELINE CPU UTILIZATION ---
    util_series = {
        'cpu_baseline': {v: mean_or_zero(cpu_samples[v]['util']) for v in sorted_vals},
        'q_paper_with_plan': {v: gpu_metrics['q_paper_with_plan'].get(v, {}).get('util', 0.0) for v in sorted_vals},
        'q_paper_without_plan': {v: gpu_metrics['q_paper_without_plan'].get(v, {}).get('util', 0.0) for v in sorted_vals},
        'c_paper_with_plan': {v: gpu_metrics['c_paper_with_plan'].get(v, {}).get('util', 0.0) for v in sorted_vals},
        'c_paper_without_plan': {v: gpu_metrics['c_paper_without_plan'].get(v, {}).get('util', 0.0) for v in sorted_vals},
    }

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    plot_grouped_bars(ax2, x, width, util_series, sorted_vals,
                      'CPU Utilization (%)', f'CPU Utilization During Compaction ({label})', ylim=(0, 110))
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)

    plt.tight_layout()
    util_out_path = os.path.join(graph_dir, f'utilization_{label}_sweep.png')
    plt.savefig(util_out_path)
    print(f"Utilization graph saved to: {util_out_path}")
    plt.close()

    # --- PLOT 3: SSD READ/WRITE BANDWIDTH ---
    read_bw_series = {
        'cpu_baseline': {v: mean_or_zero(cpu_samples[v]['read_bw']) for v in sorted_vals},
        'q_paper_with_plan': {v: gpu_metrics['q_paper_with_plan'].get(v, {}).get('read_bw', 0.0) for v in sorted_vals},
        'q_paper_without_plan': {v: gpu_metrics['q_paper_without_plan'].get(v, {}).get('read_bw', 0.0) for v in sorted_vals},
        'c_paper_with_plan': {v: gpu_metrics['c_paper_with_plan'].get(v, {}).get('read_bw', 0.0) for v in sorted_vals},
        'c_paper_without_plan': {v: gpu_metrics['c_paper_without_plan'].get(v, {}).get('read_bw', 0.0) for v in sorted_vals},
    }
    write_bw_series = {
        'cpu_baseline': {v: mean_or_zero(cpu_samples[v]['write_bw']) for v in sorted_vals},
        'q_paper_with_plan': {v: gpu_metrics['q_paper_with_plan'].get(v, {}).get('write_bw', 0.0) for v in sorted_vals},
        'q_paper_without_plan': {v: gpu_metrics['q_paper_without_plan'].get(v, {}).get('write_bw', 0.0) for v in sorted_vals},
        'c_paper_with_plan': {v: gpu_metrics['c_paper_with_plan'].get(v, {}).get('write_bw', 0.0) for v in sorted_vals},
        'c_paper_without_plan': {v: gpu_metrics['c_paper_without_plan'].get(v, {}).get('write_bw', 0.0) for v in sorted_vals},
    }

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    plot_grouped_bars(ax3, x, width, read_bw_series, sorted_vals,
                      'MB/s', f'Estimated SSD Read Bandwidth ({label})')
    plot_grouped_bars(ax4, x, width, write_bw_series, sorted_vals,
                      'MB/s', f'Estimated SSD Write Bandwidth ({label})')
    handles, labels = ax3.get_legend_handles_labels()
    fig3.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                fancybox=True, shadow=True, ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    io_out_path = os.path.join(graph_dir, f'io_utilization_{label}_sweep.png')
    plt.savefig(io_out_path)
    print(f"IO bandwidth graph saved to: {io_out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot GPComp benchmark results.')
    parser.add_argument('--sweep_dir', default=None,
                        help='Path to a specific sweep directory (e.g., sweep_results/sweep_8mb-sst_24sst)')
    parser.add_argument('--results_dir', default=DEFAULT_BASE_DIR,
                        help='Directory containing sweep_* result folders (used if --sweep_dir not given)')
    parser.add_argument('--graphs_dir', default=DEFAULT_GRAPH_DIR,
                        help='Directory for output graphs')
    parser.add_argument('--label', default=None,
                        help='Custom label for output filenames (e.g., 8mb-sst_24sst)')
    args = parser.parse_args()

    if args.sweep_dir:
        main(args.sweep_dir, args.graphs_dir, label=args.label)
    else:
        sweep_dirs = sorted(glob.glob(os.path.join(args.results_dir, 'sweep_*')))[-1:]
        if not sweep_dirs:
            print("No sweep directories found.")
        else:
            main(sweep_dirs[-1], args.graphs_dir, label=args.label)
