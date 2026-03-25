import matplotlib.pyplot as plt
import numpy as np
import os

# Data from testbench logs
data = {
    '32B': {
        'CPU Baseline': 1139.37,
        'C-Comp w/ plan': 327.12,
        'C-Comp w/o plan': 323.28,
        'Q-Comp w/ plan': 192.96,
        'Q-Comp w/o plan': 189.20
    },
    '1024B': {
        'CPU Baseline': 374.96,
        'C-Comp w/ plan': 326.04,
        'C-Comp w/o plan': 320.37,
        'Q-Comp w/ plan': 181.07,
        'Q-Comp w/o plan': 178.07
    }
}

labels = list(data['32B'].keys())
times_32b = [data['32B'][label] for label in labels]
times_1024b = [data['1024B'][label] for label in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, times_32b, width, label='32B Value Size', color='#1f77b4')
rects2 = ax.bar(x + width/2, times_1024b, width, label='1024B Value Size', color='#ff7f0e')

ax.set_ylabel('Execution Time (ms)')
ax.set_title('Testbench: CPU vs GPU Approaches (Overall Wall Time)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.0f')
ax.bar_label(rects2, padding=3, fmt='%.0f')

fig.tight_layout()
os.makedirs('/nfs/ug/groups/ece1755_w26_group1/rocksdb/cuda_test_shafin/sweep_results/sweep_nsight_nsys_8mb-sst_24sst/graphs', exist_ok=True)
plt.savefig('/nfs/ug/groups/ece1755_w26_group1/rocksdb/cuda_test_shafin/sweep_results/sweep_nsight_nsys_8mb-sst_24sst/graphs/testbench_overall_time_comparison.png', dpi=300)
print("Saved testbench_overall_time_comparison.png")

# Speedups
fig, ax = plt.subplots(figsize=(8, 6))

labels_gpu = ['C-Comp w/ plan', 'C-Comp w/o plan', 'Q-Comp w/ plan', 'Q-Comp w/o plan']
speedup_32b = [data['32B']['CPU Baseline'] / data['32B'][l] for l in labels_gpu]
speedup_1024b = [data['1024B']['CPU Baseline'] / data['1024B'][l] for l in labels_gpu]

x_gpu = np.arange(len(labels_gpu))

rects1_sp = ax.bar(x_gpu - width/2, speedup_32b, width, label='32B Value Size', color='#2ca02c')
rects2_sp = ax.bar(x_gpu + width/2, speedup_1024b, width, label='1024B Value Size', color='#d62728')

ax.set_ylabel('Speedup over CPU Baseline')
ax.set_title('Testbench: GPU Speedup over CPU Baseline')
ax.set_xticks(x_gpu)
ax.set_xticklabels(labels_gpu, rotation=45, ha='right')
ax.legend()

ax.bar_label(rects1_sp, padding=3, fmt='%.2fx')
ax.bar_label(rects2_sp, padding=3, fmt='%.2fx')

ax.axhline(y=1.0, color='black', linestyle='--')

fig.tight_layout()
plt.savefig('/nfs/ug/groups/ece1755_w26_group1/rocksdb/cuda_test_shafin/sweep_results/sweep_nsight_nsys_8mb-sst_24sst/graphs/testbench_speedup_comparison.png', dpi=300)
print("Saved testbench_speedup_comparison.png")
