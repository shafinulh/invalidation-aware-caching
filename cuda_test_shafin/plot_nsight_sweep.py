import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np


GPU_STAGE_KEYS = [
    'gpu_read_parse_ms',
    'gpu_unpack_ms',
    'gpu_sort_merge_ms',
    'gpu_plan_ms',
    'gpu_bloom_ms',
    'gpu_pack_assemble_ms',
    'gpu_write_ms',
]

GPU_STAGE_LABELS = [
    'read+parse',
    'unpack',
    'sort(merge)',
    'plan',
    'bloom',
    'pack+assemble',
    'write',
]

KERNEL_KEYS = [
    'kernel_unpack_ms',
    'kernel_sort_merge_ms',
    'kernel_bloom_ms',
    'kernel_pack_ms',
]

KERNEL_LABELS = [
    'kernel unpack',
    'kernel sort(merge)',
    'kernel bloom',
    'kernel pack',
]

NONKERNEL_KEYS = [
    'nonkernel_unpack_ms',
    'nonkernel_sort_merge_ms',
    'nonkernel_bloom_ms',
    'nonkernel_pack_ms',
]

NONKERNEL_LABELS = [
    'non-kernel unpack',
    'non-kernel sort(merge)',
    'non-kernel bloom',
    'non-kernel pack',
]


def parse_bench_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return {
            'gpu_wall_ms': 0.0,
            'gpu_cpu_util_pct': 0.0,
            'gpu_read_parse_ms': 0.0,
            'gpu_unpack_ms': 0.0,
            'gpu_sort_merge_ms': 0.0,
            'gpu_plan_ms': 0.0,
            'gpu_bloom_ms': 0.0,
            'gpu_pack_assemble_ms': 0.0,
            'gpu_write_ms': 0.0,
            'kernel_unpack_ms': 0.0,
            'kernel_sort_merge_ms': 0.0,
            'kernel_bloom_ms': 0.0,
            'kernel_pack_ms': 0.0,
            'nonkernel_unpack_ms': 0.0,
            'nonkernel_sort_merge_ms': 0.0,
            'nonkernel_bloom_ms': 0.0,
            'nonkernel_pack_ms': 0.0,
            'gpu_io_read_bw_mb_s': 0.0,
            'gpu_io_write_bw_mb_s': 0.0,
            'transfer_lb_bw_mb_s': 0.0,
            'h2d_ms': 0.0,
            'h2d_bw_mb_s': 0.0,
            'd2h_ms': 0.0,
            'd2h_bw_mb_s': 0.0,
        }

    with open(log_path, 'r') as f:
        text = f.read()

    gpu_wall = re.search(r'GPU total \(Wall\):\s*min=([\d.]+)\s*ms', text)
    gpu_util = re.search(r'GPU pipeline .* utilization:\s*([\d.]+)%', text)

    gpu_breakdown = re.search(
        r'Best GPU run breakdown \(ms\):\s*\n\s*read\+parse\s*([\d.]+)\s*unpack\s*([\d.]+)\s*sort\(merge\)\s*([\d.]+)\s*gc\s*([\d.]+)\s*plan\s*([\d.]+)\s*bloom\s*([\d.]+)\s*pack\+assemble\s*([\d.]+)\s*write\s*([\d.]+)',
        text,
    )

    kernel_only = re.search(
        r'kernel-only:\s*unpack\s*([\d.]+)\s*sort\(merge\)\s*([\d.]+)\s*bloom\s*([\d.]+)\s*pack\s*([\d.]+)',
        text,
    )

    non_kernel = re.search(
        r'non-kernel overhead \(ms\):\s*unpack\s*([\d.]+)\s*sort\(merge\)\s*([\d.]+)\s*bloom\s*([\d.]+)\s*pack\s*([\d.]+)',
        text,
    )

    io_profiles = re.findall(
        r'I/O profile:\s*input bytes\s*\d+\s*estimated SSD read BW\s*([\d.]+)\s*MB/s\s*estimated SSD write BW\s*([\d.]+)\s*MB/s',
        text,
    )

    transfer_lb_bw = re.search(r'estimated transfer\+sync BW \(lower-bound\):\s*([\d.]+)\s*MB/s', text)
    measured_transfer = re.search(
        r'measured transfer totals:\s*H2D\s*([\d.]+)\s*ms\s*\(\d+\s*B,\s*([\d.]+)\s*MB/s\)\s*D2H\s*([\d.]+)\s*ms\s*\(\d+\s*B,\s*([\d.]+)\s*MB/s\)',
        text,
    )

    gpu_read_bw = 0.0
    gpu_write_bw = 0.0
    if len(io_profiles) >= 1:
        gpu_read_bw = float(io_profiles[-1][0])
        gpu_write_bw = float(io_profiles[-1][1])

    parsed = {
        'gpu_wall_ms': float(gpu_wall.group(1)) if gpu_wall else 0.0,
        'gpu_cpu_util_pct': float(gpu_util.group(1)) if gpu_util else 0.0,
        'gpu_io_read_bw_mb_s': gpu_read_bw,
        'gpu_io_write_bw_mb_s': gpu_write_bw,
        'transfer_lb_bw_mb_s': float(transfer_lb_bw.group(1)) if transfer_lb_bw else 0.0,
        'h2d_ms': float(measured_transfer.group(1)) if measured_transfer else 0.0,
        'h2d_bw_mb_s': float(measured_transfer.group(2)) if measured_transfer else 0.0,
        'd2h_ms': float(measured_transfer.group(3)) if measured_transfer else 0.0,
        'd2h_bw_mb_s': float(measured_transfer.group(4)) if measured_transfer else 0.0,
    }

    if gpu_breakdown:
        parsed.update({
            'gpu_read_parse_ms': float(gpu_breakdown.group(1)),
            'gpu_unpack_ms': float(gpu_breakdown.group(2)),
            'gpu_sort_merge_ms': float(gpu_breakdown.group(3)),
            'gpu_plan_ms': float(gpu_breakdown.group(5)),
            'gpu_bloom_ms': float(gpu_breakdown.group(6)),
            'gpu_pack_assemble_ms': float(gpu_breakdown.group(7)),
            'gpu_write_ms': float(gpu_breakdown.group(8)),
        })
    else:
        parsed.update({
            'gpu_read_parse_ms': 0.0,
            'gpu_unpack_ms': 0.0,
            'gpu_sort_merge_ms': 0.0,
            'gpu_plan_ms': 0.0,
            'gpu_bloom_ms': 0.0,
            'gpu_pack_assemble_ms': 0.0,
            'gpu_write_ms': 0.0,
        })

    if kernel_only:
        parsed.update({
            'kernel_unpack_ms': float(kernel_only.group(1)),
            'kernel_sort_merge_ms': float(kernel_only.group(2)),
            'kernel_bloom_ms': float(kernel_only.group(3)),
            'kernel_pack_ms': float(kernel_only.group(4)),
        })
    else:
        parsed.update({
            'kernel_unpack_ms': 0.0,
            'kernel_sort_merge_ms': 0.0,
            'kernel_bloom_ms': 0.0,
            'kernel_pack_ms': 0.0,
        })

    if non_kernel:
        parsed.update({
            'nonkernel_unpack_ms': float(non_kernel.group(1)),
            'nonkernel_sort_merge_ms': float(non_kernel.group(2)),
            'nonkernel_bloom_ms': float(non_kernel.group(3)),
            'nonkernel_pack_ms': float(non_kernel.group(4)),
        })
    else:
        parsed.update({
            'nonkernel_unpack_ms': 0.0,
            'nonkernel_sort_merge_ms': 0.0,
            'nonkernel_bloom_ms': 0.0,
            'nonkernel_pack_ms': 0.0,
        })

    return parsed


def _parse_float(cell):
    cleaned = cell.replace(',', '').strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_nsys_sum_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return 0.0

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_idx = None
    header = None
    for idx, row in enumerate(rows):
        lowered = [c.strip().lower() for c in row]
        if any('time' in c for c in lowered) and any('name' in c for c in lowered):
            header_idx = idx
            header = row
            break

    if header_idx is None or header is None:
        return 0.0

    lower_header = [c.strip().lower() for c in header]
    time_col = None
    for i, col in enumerate(lower_header):
        if 'time (ns)' in col or col == 'time' or col.endswith('time'):
            time_col = i
            break

    if time_col is None:
        return 0.0

    total_ns = 0.0
    for row in rows[header_idx + 1:]:
        if time_col >= len(row):
            continue
        val = _parse_float(row[time_col])
        if val is not None:
            total_ns += val

    return total_ns / 1e6


def save_grouped_bar(values, by_mode, title, ylabel, out_path):
    modes = sorted(by_mode.keys())
    x = np.arange(len(values))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, mode in enumerate(modes):
        offset = (idx - (len(modes) - 1) / 2.0) * width
        ys = [by_mode[mode].get(v, 0.0) for v in values]
        ax.bar(x + offset, ys, width=width, label=mode, edgecolor='black')

    ax.set_title(title)
    ax.set_xlabel('Value size (bytes)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(values)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_stacked_by_run(rows, keys, labels, title, ylabel, out_path):
    if not rows:
        return

    run_labels = [f"{r['mode']}\nV{r['value_bytes']}" for r in rows]
    x = np.arange(len(rows))
    bottoms = np.zeros(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(max(11, len(rows) * 0.9), 5.5))
    for key, label in zip(keys, labels):
        ys = np.array([r.get(key, 0.0) for r in rows], dtype=float)
        ax.bar(x, ys, bottom=bottoms, label=label, edgecolor='black')
        bottoms += ys

    ax.set_title(title)
    ax.set_xlabel('Run (mode, value size)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_io_subplots(values, read_by_mode, write_by_mode, out_path):
    modes = sorted(read_by_mode.keys())
    x = np.arange(len(values))
    width = 0.8 / max(len(modes), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for idx, mode in enumerate(modes):
        offset = (idx - (len(modes) - 1) / 2.0) * width
        read_vals = [read_by_mode[mode].get(v, 0.0) for v in values]
        write_vals = [write_by_mode[mode].get(v, 0.0) for v in values]
        ax1.bar(x + offset, read_vals, width=width, label=mode, edgecolor='black')
        ax2.bar(x + offset, write_vals, width=width, label=mode, edgecolor='black')

    ax1.set_title('GPU-estimated SSD Read BW')
    ax1.set_ylabel('MB/s')
    ax1.set_xlabel('Value size (bytes)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(values)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax2.set_title('GPU-estimated SSD Write BW')
    ax2.set_ylabel('MB/s')
    ax2.set_xlabel('Value size (bytes)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(values)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.suptitle('GPU I/O Utilization (from benchmark I/O profile)')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path)
    plt.close()


def main(manifest_path, graphs_dir, out_dir):
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = int(row['value_bytes'])
            mode = row['mode']

            bench = parse_bench_log(row.get('bench_log', ''))
            kernel_ms = parse_nsys_sum_csv(row.get('kernel_csv', ''))
            api_ms = parse_nsys_sum_csv(row.get('api_csv', ''))

            rows.append({
                'value_bytes': value,
                'mode': mode,
                'tool': row.get('tool', ''),
                'report_kind': row.get('report_kind', ''),
                'report_path': row.get('report_path', ''),
                'run_status': row.get('run_status', 'unknown'),
                'exit_code': row.get('exit_code', ''),
                'gpu_wall_ms': bench['gpu_wall_ms'],
                'gpu_cpu_util_pct': bench['gpu_cpu_util_pct'],
                'gpu_read_parse_ms': bench['gpu_read_parse_ms'],
                'gpu_unpack_ms': bench['gpu_unpack_ms'],
                'gpu_sort_merge_ms': bench['gpu_sort_merge_ms'],
                'gpu_plan_ms': bench['gpu_plan_ms'],
                'gpu_bloom_ms': bench['gpu_bloom_ms'],
                'gpu_pack_assemble_ms': bench['gpu_pack_assemble_ms'],
                'gpu_write_ms': bench['gpu_write_ms'],
                'kernel_unpack_ms': bench['kernel_unpack_ms'],
                'kernel_sort_merge_ms': bench['kernel_sort_merge_ms'],
                'kernel_bloom_ms': bench['kernel_bloom_ms'],
                'kernel_pack_ms': bench['kernel_pack_ms'],
                'nonkernel_unpack_ms': bench['nonkernel_unpack_ms'],
                'nonkernel_sort_merge_ms': bench['nonkernel_sort_merge_ms'],
                'nonkernel_bloom_ms': bench['nonkernel_bloom_ms'],
                'nonkernel_pack_ms': bench['nonkernel_pack_ms'],
                'gpu_io_read_bw_mb_s': bench['gpu_io_read_bw_mb_s'],
                'gpu_io_write_bw_mb_s': bench['gpu_io_write_bw_mb_s'],
                'transfer_lb_bw_mb_s': bench['transfer_lb_bw_mb_s'],
                'h2d_ms': bench['h2d_ms'],
                'h2d_bw_mb_s': bench['h2d_bw_mb_s'],
                'd2h_ms': bench['d2h_ms'],
                'd2h_bw_mb_s': bench['d2h_bw_mb_s'],
                'nsys_kernel_total_ms': kernel_ms,
                'nsys_cuda_api_total_ms': api_ms,
            })

    summary_csv = os.path.join(out_dir, 'nsight_summary_metrics.csv')
    with open(summary_csv, 'w', newline='') as f:
        fieldnames = [
            'value_bytes', 'mode', 'tool', 'report_kind', 'report_path',
            'run_status', 'exit_code',
            'gpu_wall_ms', 'gpu_cpu_util_pct',
            'gpu_read_parse_ms', 'gpu_unpack_ms', 'gpu_sort_merge_ms', 'gpu_plan_ms',
            'gpu_bloom_ms', 'gpu_pack_assemble_ms', 'gpu_write_ms',
            'kernel_unpack_ms', 'kernel_sort_merge_ms', 'kernel_bloom_ms', 'kernel_pack_ms',
            'nonkernel_unpack_ms', 'nonkernel_sort_merge_ms', 'nonkernel_bloom_ms', 'nonkernel_pack_ms',
            'gpu_io_read_bw_mb_s', 'gpu_io_write_bw_mb_s', 'transfer_lb_bw_mb_s',
            'h2d_ms', 'h2d_bw_mb_s', 'd2h_ms', 'd2h_bw_mb_s',
            'nsys_kernel_total_ms', 'nsys_cuda_api_total_ms'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [
        r for r in rows
        if r['run_status'] != 'failed' and r['gpu_wall_ms'] > 0
    ]

    if not valid_rows:
        print('No successful benchmark rows with GPU timing found; skipping plot generation.')
        print(f'Summary metrics CSV: {summary_csv}')
        return

    valid_rows = sorted(valid_rows, key=lambda r: (r['mode'], r['value_bytes']))
    values = sorted({r['value_bytes'] for r in valid_rows})
    gpu_wall = {}
    gpu_util = {}
    io_read = {}
    io_write = {}
    transfer_lb_bw = {}
    h2d_bw = {}
    d2h_bw = {}
    nsys_kernel = {}
    nsys_api = {}

    for r in valid_rows:
        mode = r['mode']
        value = r['value_bytes']
        gpu_wall.setdefault(mode, {})[value] = r['gpu_wall_ms']
        gpu_util.setdefault(mode, {})[value] = r['gpu_cpu_util_pct']
        io_read.setdefault(mode, {})[value] = r['gpu_io_read_bw_mb_s']
        io_write.setdefault(mode, {})[value] = r['gpu_io_write_bw_mb_s']
        transfer_lb_bw.setdefault(mode, {})[value] = r['transfer_lb_bw_mb_s']
        h2d_bw.setdefault(mode, {})[value] = r['h2d_bw_mb_s']
        d2h_bw.setdefault(mode, {})[value] = r['d2h_bw_mb_s']
        nsys_kernel.setdefault(mode, {})[value] = r['nsys_kernel_total_ms']
        nsys_api.setdefault(mode, {})[value] = r['nsys_cuda_api_total_ms']

    save_grouped_bar(
        values,
        gpu_wall,
        'Bench GPU Wall Time by Value Size',
        'Wall time (ms)',
        os.path.join(graphs_dir, 'nsight_sweep_gpu_wall_ms.png'),
    )
    save_grouped_bar(
        values,
        gpu_util,
        'Bench GPU Pipeline CPU Utilization',
        'CPU utilization (%)',
        os.path.join(graphs_dir, 'nsight_sweep_gpu_cpu_util.png'),
    )

    save_grouped_bar(
        values,
        transfer_lb_bw,
        'GPU Transfer+Sync BW (Lower-Bound)',
        'MB/s',
        os.path.join(graphs_dir, 'nsight_sweep_transfer_lb_bw_mb_s.png'),
    )

    save_grouped_bar(
        values,
        h2d_bw,
        'Measured H2D Transfer Bandwidth',
        'MB/s',
        os.path.join(graphs_dir, 'nsight_sweep_h2d_bw_mb_s.png'),
    )

    save_grouped_bar(
        values,
        d2h_bw,
        'Measured D2H Transfer Bandwidth',
        'MB/s',
        os.path.join(graphs_dir, 'nsight_sweep_d2h_bw_mb_s.png'),
    )

    save_io_subplots(
        values,
        io_read,
        io_write,
        os.path.join(graphs_dir, 'nsight_sweep_io_bw_mb_s.png'),
    )

    save_stacked_by_run(
        valid_rows,
        GPU_STAGE_KEYS,
        GPU_STAGE_LABELS,
        'GPU End-to-End Stage Breakdown',
        'Time (ms)',
        os.path.join(graphs_dir, 'nsight_sweep_gpu_stage_breakdown_ms.png'),
    )

    save_stacked_by_run(
        valid_rows,
        KERNEL_KEYS,
        KERNEL_LABELS,
        'Kernel-Only Time Breakdown',
        'Time (ms)',
        os.path.join(graphs_dir, 'nsight_sweep_kernel_breakdown_ms.png'),
    )

    save_stacked_by_run(
        valid_rows,
        NONKERNEL_KEYS,
        NONKERNEL_LABELS,
        'Non-Kernel Overhead Breakdown',
        'Time (ms)',
        os.path.join(graphs_dir, 'nsight_sweep_nonkernel_breakdown_ms.png'),
    )

    if any(r['nsys_kernel_total_ms'] > 0 for r in valid_rows):
        save_grouped_bar(
            values,
            nsys_kernel,
            'Nsight Systems Total Kernel Time',
            'Kernel time (ms)',
            os.path.join(graphs_dir, 'nsight_sweep_kernel_total_ms.png'),
        )

    if any(r['nsys_cuda_api_total_ms'] > 0 for r in valid_rows):
        save_grouped_bar(
            values,
            nsys_api,
            'Nsight Systems Total CUDA API Time',
            'CUDA API time (ms)',
            os.path.join(graphs_dir, 'nsight_sweep_cuda_api_total_ms.png'),
        )

    print(f'Summary metrics CSV: {summary_csv}')
    print(f'Graphs directory: {graphs_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build summary figures from lightweight Nsight sweep outputs.')
    parser.add_argument('--manifest', required=True, help='CSV manifest produced by sweep_with_nsight.sh')
    parser.add_argument('--graphs_dir', required=True, help='Directory for generated figures')
    parser.add_argument('--out_dir', required=True, help='Sweep output directory where summary CSV is written')
    args = parser.parse_args()
    main(args.manifest, args.graphs_dir, args.out_dir)
