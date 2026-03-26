import glob
import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BASE_DIR = os.environ.get('GPCOMP_RESULTS_DIR', 'results')
DEFAULT_GRAPH_DIR = os.environ.get('GPCOMP_GRAPHS_DIR', 'graphs')


def parse_profile(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    val = re.search(r'value bytes:\s*(\d+)', text)
    gpu_wall = re.search(r'GPU total \(Wall\):\s*min=([\d.]+)', text)
    speedup = re.search(r'Speedup:\s*([\d.]+)x', text)
    non_kernel = re.search(r'non-kernel overhead \(ms\):\s*unpack\s*([\d.]+)\s*sort\(merge\)\s*([\d.]+)\s*bloom\s*([\d.]+)\s*pack\s*([\d.]+)', text)
    transfer_bw = re.search(r'estimated transfer\+sync BW \(lower-bound\):\s*([\d.]+)\s*MB/s', text)
    measured_transfer = re.search(
        r'measured transfer totals:\s*H2D\s*([\d.]+)\s*ms\s*\((\d+)\s*B,\s*([\d.]+)\s*MB/s\)\s*D2H\s*([\d.]+)\s*ms\s*\((\d+)\s*B,\s*([\d.]+)\s*MB/s\)',
        text,
    )
    data_blocks = re.search(r'output bytes\s*\d+\s*data blocks\s*(\d+)', text)

    if not val:
        return None

    non_kernel_total = 0.0
    if non_kernel:
        non_kernel_total = sum(float(non_kernel.group(i)) for i in range(1, 5))

    return {
        'value_bytes': int(val.group(1)),
        'gpu_wall_min_ms': float(gpu_wall.group(1)) if gpu_wall else 0.0,
        'speedup': float(speedup.group(1)) if speedup else 0.0,
        'non_kernel_total_ms': non_kernel_total,
        'transfer_bw_mb_s': float(transfer_bw.group(1)) if transfer_bw else 0.0,
        'transfer_h2d_ms': float(measured_transfer.group(1)) if measured_transfer else 0.0,
        'transfer_h2d_bytes': int(measured_transfer.group(2)) if measured_transfer else 0,
        'transfer_h2d_bw_mb_s': float(measured_transfer.group(3)) if measured_transfer else 0.0,
        'transfer_d2h_ms': float(measured_transfer.group(4)) if measured_transfer else 0.0,
        'transfer_d2h_bytes': int(measured_transfer.group(5)) if measured_transfer else 0,
        'transfer_d2h_bw_mb_s': float(measured_transfer.group(6)) if measured_transfer else 0.0,
        'data_blocks': int(data_blocks.group(1)) if data_blocks else 0,
    }


def mode_from_file_name(path):
    name = os.path.basename(path)
    return 'without_plan' if 'without_plan' in name else 'with_plan'


def plot_latest_sweep_profile_metrics(base_dir, graph_dir):
    sweep_dirs = sorted(glob.glob(os.path.join(base_dir, 'sweep_*')))
    if not sweep_dirs:
        print('No sweep directories found.')
        return

    latest = sweep_dirs[-1]
    files = sorted(glob.glob(os.path.join(latest, 'result_val*plan.txt')))
    if not files:
        print('No result files found in latest sweep.')
        return

    by_mode = {'with_plan': {}, 'without_plan': {}}
    for file_path in files:
        parsed = parse_profile(file_path)
        if not parsed:
            continue
        mode = mode_from_file_name(file_path)
        by_mode[mode][parsed['value_bytes']] = parsed

    values = sorted(set(by_mode['with_plan'].keys()) | set(by_mode['without_plan'].keys()))
    if not values:
        print('No parsed values found.')
        return

    def series(mode, key):
        return [by_mode[mode].get(v, {}).get(key, 0.0) for v in values]

    x = np.arange(len(values))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics = [
        ('speedup', 'Speedup (CPU/GPU)', 'x'),
        ('gpu_wall_min_ms', 'GPU Wall Time (min)', 'ms'),
        ('non_kernel_total_ms', 'GPU Non-Kernel Overhead', 'ms'),
        ('data_blocks', 'Output Data Blocks', 'count'),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        wp = series('with_plan', key)
        wop = series('without_plan', key)
        ax.bar(x - width / 2, wp, width, label='with_plan', color='#d9c8b4', edgecolor='black')
        ax.bar(x + width / 2, wop, width, label='without_plan', color='#98b4d4', edgecolor='black')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(values)
        ax.set_xlabel('Value Size (Bytes)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True)
    fig.suptitle('Fine-Grained Profile Metrics (Latest Sweep)')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(graph_dir, exist_ok=True)
    stamp = os.path.basename(latest).replace('sweep_', '')
    out = os.path.join(graph_dir, f'profile_metrics_{stamp}.png')
    plt.savefig(out)
    plt.close()
    print(f'Profile metrics graph saved to: {out}')

    # Optional second chart focused on transfer lower-bound BW.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    wp_bw = series('with_plan', 'transfer_bw_mb_s')
    wop_bw = series('without_plan', 'transfer_bw_mb_s')
    ax2.bar(x - width / 2, wp_bw, width, label='with_plan', color='#d9c8b4', edgecolor='black')
    ax2.bar(x + width / 2, wop_bw, width, label='without_plan', color='#98b4d4', edgecolor='black')
    ax2.set_title('Estimated Transfer+Sync BW (Lower-Bound)')
    ax2.set_ylabel('MB/s')
    ax2.set_xticks(x)
    ax2.set_xticklabels(values)
    ax2.set_xlabel('Value Size (Bytes)')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.legend()
    plt.tight_layout()

    out2 = os.path.join(graph_dir, f'profile_transfer_bw_{stamp}.png')
    plt.savefig(out2)
    plt.close()
    print(f'Transfer BW graph saved to: {out2}')

    # Third chart: measured H2D and D2H transfer times.
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    wp_h2d = series('with_plan', 'transfer_h2d_ms')
    wop_h2d = series('without_plan', 'transfer_h2d_ms')
    wp_d2h = series('with_plan', 'transfer_d2h_ms')
    wop_d2h = series('without_plan', 'transfer_d2h_ms')

    ax3.bar(x - width / 2, wp_h2d, width, label='with_plan', color='#d9c8b4', edgecolor='black')
    ax3.bar(x + width / 2, wop_h2d, width, label='without_plan', color='#98b4d4', edgecolor='black')
    ax3.set_title('Measured H2D Transfer Time')
    ax3.set_ylabel('ms')
    ax3.set_xticks(x)
    ax3.set_xticklabels(values)
    ax3.set_xlabel('Value Size (Bytes)')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax4.bar(x - width / 2, wp_d2h, width, label='with_plan', color='#d9c8b4', edgecolor='black')
    ax4.bar(x + width / 2, wop_d2h, width, label='without_plan', color='#98b4d4', edgecolor='black')
    ax4.set_title('Measured D2H Transfer Time')
    ax4.set_ylabel('ms')
    ax4.set_xticks(x)
    ax4.set_xticklabels(values)
    ax4.set_xlabel('Value Size (Bytes)')
    ax4.grid(True, axis='y', linestyle='--', alpha=0.6)

    handles3, labels3 = ax3.get_legend_handles_labels()
    fig3.legend(handles3, labels3, loc='upper center', ncol=2, frameon=True)
    fig3.suptitle('Measured CPU<->GPU Transfer Time')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out3 = os.path.join(graph_dir, f'profile_transfer_time_{stamp}.png')
    plt.savefig(out3)
    plt.close()
    print(f'Transfer time graph saved to: {out3}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot fine-grained GPComp profile metrics.')
    parser.add_argument('--results_dir', default=DEFAULT_BASE_DIR,
                        help='Directory containing sweep_* result folders')
    parser.add_argument('--graphs_dir', default=DEFAULT_GRAPH_DIR,
                        help='Directory for output graphs')
    args = parser.parse_args()
    plot_latest_sweep_profile_metrics(args.results_dir, args.graphs_dir)
