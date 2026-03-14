#!/usr/bin/env python3
"""Low-overhead host and thread metrics sampler for db_bench runs.

Samples Linux kernel counters from /proc and /sys for:
- backing block-device IO utilization and throughput
- system CPU utilization
- target-process thread CPU utilization

Outputs CSV time series plus a compact JSON summary in the requested output
directory. Designed to run alongside release-mode db_bench with minimal
overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from collections import defaultdict
from pathlib import Path


stop_requested = False


def _handle_stop(signum, frame):
    del signum, frame
    global stop_requested
    stop_requested = True


for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _handle_stop)


def read_proc_stat() -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    total: dict[str, int] | None = None
    per_cpu: dict[str, dict[str, int]] = {}
    fields = (
        "user",
        "nice",
        "system",
        "idle",
        "iowait",
        "irq",
        "softirq",
        "steal",
        "guest",
        "guest_nice",
    )

    with open("/proc/stat", "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            label = parts[0]
            if label == "cpu" or (label.startswith("cpu") and label[3:].isdigit()):
                values = [int(x) for x in parts[1 : 1 + len(fields)]]
                row = {
                    name: values[idx] if idx < len(values) else 0
                    for idx, name in enumerate(fields)
                }
                if label == "cpu":
                    total = row
                else:
                    per_cpu[label[3:]] = row
            elif total is not None:
                break

    if total is None:
        raise RuntimeError("failed to parse /proc/stat")
    return total, per_cpu


def read_device_stat(device: str) -> dict[str, int]:
    stat_path = Path("/sys/class/block") / device / "stat"
    values = [int(x) for x in stat_path.read_text(encoding="utf-8").split()]
    if len(values) < 11:
        raise RuntimeError(f"unexpected block stat format for {device}: {values}")

    mapping = {
        "reads_completed": values[0],
        "reads_merged": values[1],
        "sectors_read": values[2],
        "read_ms": values[3],
        "writes_completed": values[4],
        "writes_merged": values[5],
        "sectors_written": values[6],
        "write_ms": values[7],
        "ios_in_progress": values[8],
        "io_ms": values[9],
        "weighted_io_ms": values[10],
    }
    if len(values) >= 15:
        mapping.update(
            {
                "discards_completed": values[11],
                "discards_merged": values[12],
                "sectors_discarded": values[13],
                "discard_ms": values[14],
            }
        )
    if len(values) >= 17:
        mapping.update(
            {
                "flush_completed": values[15],
                "flush_ms": values[16],
            }
        )
    return mapping


def _parse_task_stat(stat_line: str) -> dict[str, int | str]:
    rparen = stat_line.rfind(")")
    lparen = stat_line.find("(")
    comm = stat_line[lparen + 1 : rparen]
    rest = stat_line[rparen + 2 :].split()
    if len(rest) < 37:
        raise RuntimeError("unexpected /proc task stat format")

    return {
        "comm": comm,
        "state": rest[0],
        "utime": int(rest[11]),
        "stime": int(rest[12]),
        "processor": int(rest[36]),
    }


def read_threads(pid: int) -> dict[int, dict[str, int | str]]:
    tasks_dir = Path("/proc") / str(pid) / "task"
    threads: dict[int, dict[str, int | str]] = {}
    if not tasks_dir.exists():
        return threads

    try:
        task_dirs = list(tasks_dir.iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return threads

    for task_dir in task_dirs:
        if not task_dir.name.isdigit():
            continue
        tid = int(task_dir.name)
        try:
            stat_line = (task_dir / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            continue
        threads[tid] = _parse_task_stat(stat_line)
    return threads


def cpu_percent_from_delta(delta: dict[str, int]) -> dict[str, float]:
    total = sum(delta.values())
    if total <= 0:
        return {
            "user_pct": 0.0,
            "system_pct": 0.0,
            "idle_pct": 0.0,
            "iowait_pct": 0.0,
            "busy_pct": 0.0,
        }

    user_pct = 100.0 * (delta["user"] + delta["nice"]) / total
    system_pct = 100.0 * (delta["system"] + delta["irq"] + delta["softirq"]) / total
    idle_pct = 100.0 * delta["idle"] / total
    iowait_pct = 100.0 * delta["iowait"] / total
    busy_pct = 100.0 - idle_pct - iowait_pct
    return {
        "user_pct": user_pct,
        "system_pct": system_pct,
        "idle_pct": idle_pct,
        "iowait_pct": iowait_pct,
        "busy_pct": busy_pct,
    }


def classify_thread(comm: str) -> str:
    if comm.startswith("rocksdb:high"):
        return "rocksdb_flush"
    if comm.startswith("rocksdb:low") or comm.startswith("rocksdb:bottom"):
        return "rocksdb_compaction"
    if comm.startswith("rocksdb:"):
        return "rocksdb_other_bg"
    if comm.startswith("db_bench"):
        return "foreground"
    return "other"


def process_alive(pid: int) -> bool:
    return (Path("/proc") / str(pid)).exists()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.interval_sec <= 0:
        raise SystemExit("--interval-sec must be > 0")

    clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    sector_bytes = 512.0
    start_wall = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_prev, per_cpu_prev = read_proc_stat()
    dev_prev = read_device_stat(args.device)
    threads_prev = read_threads(args.pid)
    time_prev = time.monotonic()

    system_rows: list[dict[str, object]] = []
    per_cpu_rows: list[dict[str, object]] = []
    io_rows: list[dict[str, object]] = []
    process_rows: list[dict[str, object]] = []
    thread_rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []

    role_interval_sum: defaultdict[str, float] = defaultdict(float)
    role_interval_max: defaultdict[str, float] = defaultdict(float)
    thread_name_cpu_sum: defaultdict[str, float] = defaultdict(float)
    thread_name_cpu_max: defaultdict[str, float] = defaultdict(float)
    thread_name_samples: defaultdict[str, int] = defaultdict(int)

    interval_count = 0
    while not stop_requested and process_alive(args.pid):
        time.sleep(args.interval_sec)
        try:
            total_cur, per_cpu_cur = read_proc_stat()
            dev_cur = read_device_stat(args.device)
            threads_cur = read_threads(args.pid)
        except FileNotFoundError:
            break
        if not threads_cur and not process_alive(args.pid):
            break

        time_cur = time.monotonic()
        elapsed = time_cur - time_prev
        if elapsed <= 0:
            continue

        secs_elapsed = time.time() - start_wall

        total_delta = {k: total_cur[k] - total_prev[k] for k in total_cur}
        total_cpu = cpu_percent_from_delta(total_delta)
        system_rows.append(
            {
                "secs_elapsed": round(secs_elapsed, 3),
                "busy_pct": round(total_cpu["busy_pct"], 4),
                "user_pct": round(total_cpu["user_pct"], 4),
                "system_pct": round(total_cpu["system_pct"], 4),
                "iowait_pct": round(total_cpu["iowait_pct"], 4),
                "idle_pct": round(total_cpu["idle_pct"], 4),
            }
        )

        for cpu_id, cur_values in sorted(per_cpu_cur.items(), key=lambda item: int(item[0])):
            prev_values = per_cpu_prev.get(cpu_id)
            if prev_values is None:
                continue
            delta = {k: cur_values[k] - prev_values[k] for k in cur_values}
            row = cpu_percent_from_delta(delta)
            per_cpu_rows.append(
                {
                    "secs_elapsed": round(secs_elapsed, 3),
                    "cpu": cpu_id,
                    "busy_pct": round(row["busy_pct"], 4),
                    "user_pct": round(row["user_pct"], 4),
                    "system_pct": round(row["system_pct"], 4),
                    "iowait_pct": round(row["iowait_pct"], 4),
                    "idle_pct": round(row["idle_pct"], 4),
                }
            )

        dev_delta = {k: dev_cur.get(k, 0) - dev_prev.get(k, 0) for k in dev_cur}
        io_util_pct = 100.0 * dev_delta["io_ms"] / (elapsed * 1000.0)
        avg_queue_depth = dev_delta["weighted_io_ms"] / (elapsed * 1000.0)
        io_rows.append(
            {
                "secs_elapsed": round(secs_elapsed, 3),
                "device": args.device,
                "rps": round(dev_delta["reads_completed"] / elapsed, 4),
                "wps": round(dev_delta["writes_completed"] / elapsed, 4),
                "rkib_per_sec": round(
                    (dev_delta["sectors_read"] * sector_bytes / 1024.0) / elapsed, 4
                ),
                "wkib_per_sec": round(
                    (dev_delta["sectors_written"] * sector_bytes / 1024.0) / elapsed, 4
                ),
                "util_pct": round(io_util_pct, 4),
                "avg_queue_depth": round(avg_queue_depth, 4),
                "read_await_ms": round(
                    dev_delta["read_ms"] / dev_delta["reads_completed"], 4
                )
                if dev_delta["reads_completed"] > 0
                else 0.0,
                "write_await_ms": round(
                    dev_delta["write_ms"] / dev_delta["writes_completed"], 4
                )
                if dev_delta["writes_completed"] > 0
                else 0.0,
                "ios_in_progress": dev_cur["ios_in_progress"],
            }
        )

        role_totals: defaultdict[str, float] = defaultdict(float)
        for tid, cur in sorted(threads_cur.items()):
            prev = threads_prev.get(tid)
            if prev is None:
                continue
            delta_ticks = (int(cur["utime"]) + int(cur["stime"])) - (
                int(prev["utime"]) + int(prev["stime"])
            )
            cpu_pct = 100.0 * (delta_ticks / clk_tck) / elapsed
            comm = str(cur["comm"])
            role = classify_thread(comm)
            thread_rows.append(
                {
                    "secs_elapsed": round(secs_elapsed, 3),
                    "tid": tid,
                    "thread_name": comm,
                    "thread_role": role,
                    "cpu_pct": round(cpu_pct, 4),
                    "cpu_id": int(cur["processor"]),
                    "state": str(cur["state"]),
                }
            )
            role_totals[role] += cpu_pct
            thread_name_cpu_sum[comm] += cpu_pct
            thread_name_cpu_max[comm] = max(thread_name_cpu_max[comm], cpu_pct)
            thread_name_samples[comm] += 1

        for role, cpu_pct in sorted(role_totals.items()):
            role_rows.append(
                {
                    "secs_elapsed": round(secs_elapsed, 3),
                    "thread_role": role,
                    "cpu_pct": round(cpu_pct, 4),
                }
            )
            role_interval_sum[role] += cpu_pct
            role_interval_max[role] = max(role_interval_max[role], cpu_pct)

        process_rows.append(
            {
                "secs_elapsed": round(secs_elapsed, 3),
                "cpu_pct": round(sum(role_totals.values()), 4),
            }
        )

        total_prev = total_cur
        per_cpu_prev = per_cpu_cur
        dev_prev = dev_cur
        threads_prev = threads_cur
        time_prev = time_cur
        interval_count += 1

    write_csv(
        output_dir / "system_cpu.csv",
        ["secs_elapsed", "busy_pct", "user_pct", "system_pct", "iowait_pct", "idle_pct"],
        system_rows,
    )
    write_csv(
        output_dir / "per_cpu.csv",
        ["secs_elapsed", "cpu", "busy_pct", "user_pct", "system_pct", "iowait_pct", "idle_pct"],
        per_cpu_rows,
    )
    write_csv(
        output_dir / "device_io.csv",
        [
            "secs_elapsed",
            "device",
            "rps",
            "wps",
            "rkib_per_sec",
            "wkib_per_sec",
            "util_pct",
            "avg_queue_depth",
            "read_await_ms",
            "write_await_ms",
            "ios_in_progress",
        ],
        io_rows,
    )
    write_csv(
        output_dir / "process_cpu.csv",
        ["secs_elapsed", "cpu_pct"],
        process_rows,
    )
    write_csv(
        output_dir / "thread_cpu.csv",
        ["secs_elapsed", "tid", "thread_name", "thread_role", "cpu_pct", "cpu_id", "state"],
        thread_rows,
    )
    write_csv(
        output_dir / "thread_role_cpu.csv",
        ["secs_elapsed", "thread_role", "cpu_pct"],
        role_rows,
    )

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "pid": args.pid,
        "device": args.device,
        "interval_sec": args.interval_sec,
        "sample_count": interval_count,
        "avg_system_cpu_busy_pct": round(avg([float(r["busy_pct"]) for r in system_rows]), 4),
        "avg_system_cpu_user_pct": round(avg([float(r["user_pct"]) for r in system_rows]), 4),
        "avg_system_cpu_system_pct": round(avg([float(r["system_pct"]) for r in system_rows]), 4),
        "avg_system_cpu_iowait_pct": round(avg([float(r["iowait_pct"]) for r in system_rows]), 4),
        "avg_device_util_pct": round(avg([float(r["util_pct"]) for r in io_rows]), 4),
        "max_device_util_pct": round(max([float(r["util_pct"]) for r in io_rows], default=0.0), 4),
        "avg_device_rkib_per_sec": round(avg([float(r["rkib_per_sec"]) for r in io_rows]), 4),
        "avg_device_wkib_per_sec": round(avg([float(r["wkib_per_sec"]) for r in io_rows]), 4),
        "avg_device_queue_depth": round(avg([float(r["avg_queue_depth"]) for r in io_rows]), 4),
        "avg_process_cpu_pct": round(avg([float(r["cpu_pct"]) for r in process_rows]), 4),
        "max_process_cpu_pct": round(max([float(r["cpu_pct"]) for r in process_rows], default=0.0), 4),
        "avg_thread_role_cpu_pct": {
            role: round(role_interval_sum[role] / interval_count, 4)
            for role in sorted(role_interval_sum)
        },
        "max_thread_role_cpu_pct": {
            role: round(role_interval_max[role], 4) for role in sorted(role_interval_max)
        },
        "avg_thread_name_cpu_pct": {
            name: round(thread_name_cpu_sum[name] / thread_name_samples[name], 4)
            for name in sorted(thread_name_samples)
        },
        "max_thread_name_cpu_pct": {
            name: round(thread_name_cpu_max[name], 4)
            for name in sorted(thread_name_cpu_max)
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
