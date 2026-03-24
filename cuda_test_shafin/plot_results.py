import os
import glob
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASE_DIR = os.environ.get('GPCOMP_RESULTS_DIR', 'sweep_results')
DEFAULT_GRAPH_DIR = os.environ.get('GPCOMP_GRAPHS_DIR', 'graphs')


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

    val_bytes = int(val_match.group(1)) if val_match else 0
    key_bytes = int(key_match.group(1)) if key_match else 16

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

    # Pipeline-only utilization (excludes I/O)
    cpu_pipeline_util = re.search(r'CPU pipeline \(Wall\):\s*([\d.]+)\s*ms\s*\(CPU-Time\):\s*([\d.]+)\s*ms\s*utilization:\s*([\d.]+)%', content)
    gpu_pipeline_util = re.search(r'GPU pipeline \(Wall\):\s*([\d.]+)\s*ms\s*\(CPU-Time\):\s*([\d.]+)\s*ms\s*utilization:\s*([\d.]+)%', content)

    cpu_pipe_util = float(cpu_pipeline_util.group(3)) if cpu_pipeline_util else 0.0
    gpu_pipe_util = float(gpu_pipeline_util.group(3)) if gpu_pipeline_util else 0.0

    return val_bytes, cpu_throughput, gpu_throughput, cpu_pipe_util, gpu_pipe_util


def main(sweep_dir, graph_dir, label=None):
    print(f"Reading logs from: {sweep_dir}")

    if label is None:
        label = extract_label(sweep_dir)

    # Process files
    data_with_plan = {}
    data_without_plan = {}
    data_cpu = {}

    # Throughput from main benchmark logs (.log, not _util)
    with_plan_files = glob.glob(os.path.join(sweep_dir, '*with_plan.log'))
    for f in with_plan_files:
        if 'without' in f: continue
        if '_util' in f: continue
        val, cpu_tp, gpu_tp, _, _ = parse_log(f)
        if val > 0:
            data_with_plan[val] = gpu_tp
            data_cpu[val] = cpu_tp

    without_plan_files = glob.glob(os.path.join(sweep_dir, '*without_plan.log'))
    for f in without_plan_files:
        if '_util' in f: continue
        val, _, gpu_tp, _, _ = parse_log(f)
        if val > 0:
            data_without_plan[val] = gpu_tp

    # Pipeline utilization from _util logs (no instrumentation)
    pipe_util_cpu = {}
    pipe_util_gpu_wp = {}
    pipe_util_gpu_wop = {}

    wp_util_files = glob.glob(os.path.join(sweep_dir, '*with_plan_util.log'))
    for f in wp_util_files:
        if 'without' in f: continue
        val, _, _, cpu_pipe_ut, gpu_pipe_ut = parse_log(f)
        if val > 0:
            pipe_util_cpu[val] = cpu_pipe_ut
            pipe_util_gpu_wp[val] = gpu_pipe_ut

    wop_util_files = glob.glob(os.path.join(sweep_dir, '*without_plan_util.log'))
    for f in wop_util_files:
        val, _, _, _, gpu_pipe_ut = parse_log(f)
        if val > 0:
            pipe_util_gpu_wop[val] = gpu_pipe_ut

    # Prepare data for plotting
    sorted_vals = sorted(list(data_with_plan.keys()))
    if not sorted_vals:
         print("No valid plotted data.")
         return

    cpu_tp_list = [data_cpu[v] for v in sorted_vals]
    wp_tp_list = [data_with_plan[v] for v in sorted_vals]
    wop_tp_list = [data_without_plan.get(v, 0) for v in sorted_vals]

    x = np.arange(len(sorted_vals))
    width = 0.25

    os.makedirs(graph_dir, exist_ok=True)

    # --- PLOT 1: THROUGHPUT ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width, cpu_tp_list, width, label='CPU Comp', color='#ffffe0', edgecolor='black')
    ax.bar(x, wp_tp_list, width, label='GPComp (With Plan)', color='#e6e6fa', edgecolor='black')
    ax.bar(x + width, wop_tp_list, width, label='GPComp (Without Plan)', color='#b0c4de', edgecolor='black')

    ax.set_ylabel('Throughput (Ops/s)')
    ax.set_xlabel('Value Size (Bytes)')
    ax.set_title(f'Throughput vs Value Size ({label})')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_vals)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)

    plt.tight_layout()
    tp_out_path = os.path.join(graph_dir, f'throughput_{label}_sweep.png')
    plt.savefig(tp_out_path)
    print(f"Throughput graph saved to: {tp_out_path}")
    plt.close()

    # --- PLOT 2: PIPELINE CPU UTILIZATION (I/O excluded) ---
    pipe_cpu_list = [pipe_util_cpu.get(v, 0) for v in sorted_vals]
    pipe_wp_list = [pipe_util_gpu_wp.get(v, 0) for v in sorted_vals]
    pipe_wop_list = [pipe_util_gpu_wop.get(v, 0) for v in sorted_vals]

    if any(v > 0 for v in pipe_wp_list + pipe_wop_list):
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(x - width, pipe_cpu_list, width, label='CPU Compaction', color='#ffe4e1', edgecolor='black')
        ax2.bar(x, pipe_wp_list, width, label='GPComp (With Plan)', color='#e6e6fa', edgecolor='black')
        ax2.bar(x + width, pipe_wop_list, width, label='GPComp (Without Plan)', color='#b0c4de', edgecolor='black')

        ax2.set_ylabel('CPU Utilization (%)')
        ax2.set_xlabel('Value Size (Bytes)')
        ax2.set_title(f'CPU Utilization During Compaction (I/O Excluded) ({label})')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sorted_vals)
        ax2.set_ylim(0, 110)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)

        plt.tight_layout()
        util_out_path = os.path.join(graph_dir, f'utilization_{label}_sweep.png')
        plt.savefig(util_out_path)
        print(f"Utilization graph saved to: {util_out_path}")
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
