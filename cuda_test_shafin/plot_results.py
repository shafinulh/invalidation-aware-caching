import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def extract_sweep_timestamp(sweep_path):
    name = os.path.basename(os.path.normpath(sweep_path))
    m = re.match(r'sweep_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$', name)
    if not m:
        return datetime.now().strftime('%Y%m%d_%H%M%S'), None
    raw = m.group(1)
    dt = datetime.strptime(raw, '%Y-%m-%d_%H-%M-%S')
    return dt.strftime('%Y%m%d_%H%M%S'), dt.strftime('%Y-%m-%d %H:%M:%S')

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
    
    # Utilization values
    cpu_cpu_time = re.search(r'CPU total \(CPU-Time\):\s*([\d.]+)', content)
    gpu_cpu_time = re.search(r'GPU total \(CPU-Time\):\s*([\d.]+)', content)
    cpu_mean_wall = re.search(r'CPU total \(Wall\):.*mean=([\d.]+)', content)
    gpu_mean_wall = re.search(r'GPU total \(Wall\):.*mean=([\d.]+)', content)

    # Utilization = (CPU-Time / Mean Wall Time) * 100
    cpu_util = 0.0
    if cpu_cpu_time and cpu_mean_wall and float(cpu_mean_wall.group(1)) > 0:
        cpu_util = (float(cpu_cpu_time.group(1)) / float(cpu_mean_wall.group(1))) * 100

    gpu_util = 0.0
    if gpu_cpu_time and gpu_mean_wall and float(gpu_mean_wall.group(1)) > 0:
        gpu_util = (float(gpu_cpu_time.group(1)) / float(gpu_mean_wall.group(1))) * 100

    return val_bytes, cpu_throughput, gpu_throughput, cpu_util, gpu_util


def parse_log_metrics(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    val_match = re.search(r'value bytes:\s*(\d+)', content)
    val_bytes = int(val_match.group(1)) if val_match else 0

    gpu_wall = re.search(r'GPU total \(Wall\):\s*min=([\d.]+)\s*ms', content)
    speedup = re.search(r'Speedup:\s*([\d.]+)x', content)
    blocks = re.search(r'data blocks\s*(\d+)', content)

    gpu_pack = 0.0
    m_gpu = re.search(r'Best GPU run breakdown \(ms\):\s*\n\s*read\+parse[^\n]*pack\+assemble\s*([\d.]+)', content)
    if m_gpu:
        gpu_pack = float(m_gpu.group(1))

    return {
        'val': val_bytes,
        'gpu_wall_min_ms': float(gpu_wall.group(1)) if gpu_wall else 0.0,
        'speedup': float(speedup.group(1)) if speedup else 0.0,
        'data_blocks': int(blocks.group(1)) if blocks else 0,
        'gpu_pack_assemble_ms': gpu_pack,
    }


def collect_without_plan_metrics(sweep_dir):
    metrics = {}
    files = glob.glob(os.path.join(sweep_dir, '*without_plan.txt'))
    for file_path in files:
        m = parse_log_metrics(file_path)
        if m['val'] > 0:
            metrics[m['val']] = m
    return metrics


def plot_without_plan_comparison(base_sweep, opt_sweep, out_dir):
    base = collect_without_plan_metrics(base_sweep)
    opt = collect_without_plan_metrics(opt_sweep)
    vals = sorted(set(base.keys()) & set(opt.keys()))
    if not vals:
        print('No overlapping WITHOUT PLAN value sizes between sweeps.')
        return

    x = np.arange(len(vals))
    width = 0.36

    base_wall = [base[v]['gpu_wall_min_ms'] for v in vals]
    opt_wall = [opt[v]['gpu_wall_min_ms'] for v in vals]
    base_spd = [base[v]['speedup'] for v in vals]
    opt_spd = [opt[v]['speedup'] for v in vals]
    base_blocks = [base[v]['data_blocks'] for v in vals]
    opt_blocks = [opt[v]['data_blocks'] for v in vals]
    base_pack = [base[v]['gpu_pack_assemble_ms'] for v in vals]
    opt_pack = [opt[v]['gpu_pack_assemble_ms'] for v in vals]

    ts_base, human_base = extract_sweep_timestamp(base_sweep)
    ts_opt, human_opt = extract_sweep_timestamp(opt_sweep)
    stamp = f'{ts_base}_vs_{ts_opt}'

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(x - width / 2, base_wall, width, label='Baseline', color='#d9c8b4', edgecolor='black')
    axes[0, 0].bar(x + width / 2, opt_wall, width, label='Optimized', color='#98b4d4', edgecolor='black')
    axes[0, 0].set_title('GPU Total Wall (min)')
    axes[0, 0].set_ylabel('ms')

    axes[0, 1].bar(x - width / 2, base_spd, width, label='Baseline', color='#d9c8b4', edgecolor='black')
    axes[0, 1].bar(x + width / 2, opt_spd, width, label='Optimized', color='#98b4d4', edgecolor='black')
    axes[0, 1].set_title('Speedup (CPU/GPU)')
    axes[0, 1].set_ylabel('x')

    axes[1, 0].bar(x - width / 2, base_blocks, width, label='Baseline', color='#d9c8b4', edgecolor='black')
    axes[1, 0].bar(x + width / 2, opt_blocks, width, label='Optimized', color='#98b4d4', edgecolor='black')
    axes[1, 0].set_title('Output Data Blocks')
    axes[1, 0].set_ylabel('count')

    axes[1, 1].bar(x - width / 2, base_pack, width, label='Baseline', color='#d9c8b4', edgecolor='black')
    axes[1, 1].bar(x + width / 2, opt_pack, width, label='Optimized', color='#98b4d4', edgecolor='black')
    axes[1, 1].set_title('GPU Pack+Assemble (best run)')
    axes[1, 1].set_ylabel('ms')

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(vals)
        ax.set_xlabel('Value Size (Bytes)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True)
    if human_base and human_opt:
        fig.suptitle(f'WITHOUT PLAN: Baseline vs Optimized\n{human_base} vs {human_opt}')
    else:
        fig.suptitle('WITHOUT PLAN: Baseline vs Optimized')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(out_dir, f'without_plan_compare_{stamp}.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Comparison graph saved to: {out_path}')

def main():
    base_dir = '/nfs/ug/groups/ece1755_w26_group1/rocksdb/cuda_test_shafin/results'
    # Find the latest sweep directory
    sweep_dirs = sorted(glob.glob(os.path.join(base_dir, 'sweep_*')))[-1:]
    if not sweep_dirs:
        print("No sweep directories found.")
        return
    
    latest_sweep = sweep_dirs[-1]
    print(f"Reading logs from: {latest_sweep}")
    
    # Process files
    data_with_plan = {}
    data_without_plan = {}
    data_cpu = {}
    
    # Util files
    util_cpu = {}
    util_gpu_wp = {}
    util_gpu_wop = {}
    
    with_plan_files = glob.glob(os.path.join(latest_sweep, '*with_plan.txt'))
    for f in with_plan_files:
        if 'without' in f: continue
        val, cpu_tp, gpu_tp, cpu_ut, gpu_ut = parse_log(f)
        if val > 0:
            data_with_plan[val] = gpu_tp
            data_cpu[val] = cpu_tp
            util_cpu[val] = cpu_ut
            util_gpu_wp[val] = gpu_ut
            
    without_plan_files = glob.glob(os.path.join(latest_sweep, '*without_plan.txt'))
    for f in without_plan_files:
        val, cpu_tp, gpu_tp, cpu_ut, gpu_ut = parse_log(f)
        if val > 0:
            data_without_plan[val] = gpu_tp
            util_gpu_wop[val] = gpu_ut
            
    # Prepare data for plotting
    sorted_vals = sorted(list(data_with_plan.keys()))
    if not sorted_vals:
         print("No valid plotted data.")
         return
         
    cpu_tp_list = [data_cpu[v] for v in sorted_vals]
    wp_tp_list = [data_with_plan[v] for v in sorted_vals]
    wop_tp_list = [data_without_plan.get(v, 0) for v in sorted_vals]
    
    cpu_util_list = [util_cpu[v] for v in sorted_vals]
    wp_util_list = [util_gpu_wp[v] for v in sorted_vals]
    wop_util_list = [util_gpu_wop.get(v, 0) for v in sorted_vals]
    
    x = np.arange(len(sorted_vals))
    width = 0.25
    
    timestamp, human_ts = extract_sweep_timestamp(latest_sweep)
    out_dir = '/nfs/ug/groups/ece1755_w26_group1/rocksdb/cuda_test_shafin/graphs'
    os.makedirs(out_dir, exist_ok=True)
    
    # --- PLOT 1: THROUGHPUT ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width, cpu_tp_list, width, label='CPU Comp', color='#ffffe0', edgecolor='black')
    ax.bar(x, wp_tp_list, width, label='GPComp (With Plan)', color='#e6e6fa', edgecolor='black')
    ax.bar(x + width, wop_tp_list, width, label='GPComp (Without Plan)', color='#b0c4de', edgecolor='black')
    
    ax.set_ylabel('Throughput (Ops/s)')
    ax.set_xlabel('Value Size (Bytes)')
    title_tp = 'Throughput vs Value Size'
    if human_ts:
        title_tp += f' ({human_ts})'
    ax.set_title(title_tp)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_vals)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)
    
    plt.tight_layout()
    tp_out_path = os.path.join(out_dir, f'throughput_{timestamp}.png')
    plt.savefig(tp_out_path)
    print(f"Throughput graph saved to: {tp_out_path}")
    plt.close()
    
    # --- PLOT 2: HARDWARE UTILIZATION ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(x - width, cpu_util_list, width, label='CPU Run Util', color='#ffe4e1', edgecolor='black')
    ax2.bar(x, wp_util_list, width, label='GPU Run Util (With Plan)', color='#e6e6fa', edgecolor='black')
    ax2.bar(x + width, wop_util_list, width, label='GPU Run Util (Without Plan)', color='#b0c4de', edgecolor='black')
    
    ax2.set_ylabel('CPU Utilization (%)')
    ax2.set_xlabel('Value Size (Bytes)')
    title_util = 'Actual System CPU Thread Utilization vs Time'
    if human_ts:
        title_util += f' ({human_ts})'
    ax2.set_title(title_util)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_vals)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)
    
    plt.tight_layout()
    util_out_path = os.path.join(out_dir, f'utilization_{timestamp}.png')
    plt.savefig(util_out_path)
    print(f"Utilization graph saved to: {util_out_path}")
    plt.close()

    # Intentionally disabled: no standalone WITHOUT PLAN comparison graph generation.

if __name__ == '__main__':
    main()