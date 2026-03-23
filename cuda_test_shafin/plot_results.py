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

if __name__ == '__main__':
    main()