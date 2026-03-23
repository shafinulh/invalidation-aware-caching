import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    val_match = re.search(r'value bytes:\s*(\d+)', content)
    key_match = re.search(r'key bytes:\s*(\d+)', content)
    
    val_bytes = int(val_match.group(1)) if val_match else 0
    key_bytes = int(key_match.group(1)) if key_match else 16

    cpu_wall = re.search(r'CPU total \(Wall\):\s*min=([\d.]+)\s*ms', content)
    gpu_wall = re.search(r'GPU total \(Wall\):\s*min=([\d.]+)\s*ms', content)
    cpu_wall = float(cpu_wall.group(1)) if cpu_wall else 0.0
    gpu_wall = float(gpu_wall.group(1)) if gpu_wall else 0.0

    output_bytes_match = re.search(r'output bytes\s*(\d+)', content)
    output_bytes = float(output_bytes_match.group(1)) if output_bytes_match else 0.0

    ops = output_bytes / (key_bytes + val_bytes) if (key_bytes + val_bytes) > 0 else 0
    cpu_throughput = (ops / (cpu_wall / 1000.0)) if cpu_wall > 0 else 0
    gpu_throughput = (ops / (gpu_wall / 1000.0)) if gpu_wall > 0 else 0
    
    return val_bytes, cpu_throughput, gpu_throughput

def main():
    base_dir = '/nfs/ug/groups/ece1755_w26_group1/rocksdb/cuda_test_shafin/results'
    # Find the latest sweep directory
    sweep_dirs = sorted(glob.glob(os.path.join(base_dir, 'sweep_*')))[-1:]
    if not sweep_dirs:
        print("No sweep directories found.")
        return
    
    latest_sweep = sweep_dirs[-1]
    
    # Process files
    data_with_plan = {}
    data_without_plan = {}
    data_cpu = {}
    
    with_plan_files = glob.glob(os.path.join(latest_sweep, '*with_plan.txt'))
    for f in with_plan_files:
        if 'without' in f: continue
        val, cpu_tp, gpu_tp = parse_log(f)
        if val > 0:
            data_with_plan[val] = gpu_tp
            data_cpu[val] = cpu_tp
            
    without_plan_files = glob.glob(os.path.join(latest_sweep, '*without_plan.txt'))
    for f in without_plan_files:
        val, cpu_tp, gpu_tp = parse_log(f)
        if val > 0:
            data_without_plan[val] = gpu_tp
            
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Similar to screenshot: CPU Comp, GPComp (With Plan), GPComp (Without Plan)
    rects1 = ax.bar(x - width, cpu_tp_list, width, label='CPU Comp', color='#ffffe0', edgecolor='black')
    rects2 = ax.bar(x, wp_tp_list, width, label='GPComp (With Plan)', color='#e6e6fa', edgecolor='black')
    rects3 = ax.bar(x + width, wop_tp_list, width, label='GPComp (Without Plan)', color='#b0c4de', edgecolor='black')
    
    ax.set_ylabel('Throughput (Ops/s)')
    ax.set_xlabel('Value Size (Bytes)')
    ax.set_title('Throughput vs Value Size')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_vals)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Legend at the top like screenshot
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)
    
    # Graph folder
    out_dir = '/nfs/ug/groups/ece1755_w26_group1/rocksdb/cuda_test_shafin/graphs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'throughput_chart.png')
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Graph saved to {out_path}")

if __name__ == '__main__':
    main()