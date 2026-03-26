#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean


MARKER_PREFIX = "BENCH_ITERATION_MARKER "

QUERY_COLUMN_MAP = {
    "utilization.gpu": "gpu_util_pct",
    "utilization.memory": "memory_util_pct",
    "memory.used": "memory_used_mb",
    "memory.total": "memory_total_mb",
    "power.draw": "power_draw_w",
    "temperature.gpu": "temperature_c",
    "clocks.sm": "sm_clock_mhz",
    "clocks.mem": "mem_clock_mhz",
}

ITERATION_COLUMNS = [
    "iteration_index",
    "warmup",
    "timed",
    "start_epoch_ms",
    "end_epoch_ms",
    "duration_ms",
    "query_sample_count",
    "gpu_util_pct_avg",
    "gpu_util_pct_min",
    "gpu_util_pct_max",
    "memory_util_pct_avg",
    "memory_util_pct_min",
    "memory_util_pct_max",
    "memory_used_mb_avg",
    "memory_used_mb_min",
    "memory_used_mb_max",
    "memory_total_mb",
    "power_draw_w_avg",
    "power_draw_w_min",
    "power_draw_w_max",
    "temperature_c_avg",
    "temperature_c_min",
    "temperature_c_max",
    "sm_clock_mhz_avg",
    "sm_clock_mhz_min",
    "sm_clock_mhz_max",
    "mem_clock_mhz_avg",
    "mem_clock_mhz_min",
    "mem_clock_mhz_max",
    "pstate_mode",
    "dmon_sample_count",
    "dmon_sm_pct_avg",
    "dmon_sm_pct_min",
    "dmon_sm_pct_max",
    "dmon_mem_pct_avg",
    "dmon_mem_pct_min",
    "dmon_mem_pct_max",
    "sm_active_pct_avg",
    "sm_active_pct_min",
    "sm_active_pct_max",
    "dram_active_pct_avg",
    "dram_active_pct_min",
    "dram_active_pct_max",
    "gpu_total_ms",
    "gpu_pipeline_wall_ms",
    "gpu_pipeline_cpu_ms",
    "gpu_read_parse_ms",
    "gpu_write_ms",
    "gpu_output_bytes",
    "cpu_total_ms",
    "cpu_pipeline_wall_ms",
    "cpu_pipeline_cpu_ms",
    "cpu_read_parse_ms",
    "cpu_write_ms",
    "cpu_output_bytes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw nvidia-smi traces per benchmark iteration.")
    parser.add_argument("--bench_log", required=True, help="Benchmark log containing BENCH_ITERATION_MARKER lines")
    parser.add_argument("--query_csv", required=True, help="Raw nvidia-smi --query-gpu output")
    parser.add_argument("--out_csv", required=True, help="Per-iteration summary CSV")
    parser.add_argument("--out_json", required=True, help="Run-level summary JSON")
    parser.add_argument("--dmon_log", default="", help="Optional nvidia-smi dmon output")
    parser.add_argument("--query_interval_ms", type=int, default=100, help="Sampling interval used for --query-gpu")
    parser.add_argument("--dmon_interval_sec", "--gpm_interval_sec", dest="dmon_interval_sec", type=float, default=1.0, help="Sampling interval used for dmon")
    parser.add_argument(
        "--force_first_run_warmup",
        action="store_true",
        help="Mark iteration 0 as warmup in the summary even if the benchmark did not",
    )
    return parser.parse_args()


def parse_scalar(value: str):
    if value in {"", "-", "N/A", "[Not Supported]"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_marker_fields(payload: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for token in payload.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = parse_scalar(value)
    return result


def parse_bench_iterations(log_path: Path) -> tuple[list[dict[str, object]], list[int]]:
    iterations: dict[int, dict[str, object]] = {}
    incomplete: list[int] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith(MARKER_PREFIX):
                continue
            fields = parse_marker_fields(line[len(MARKER_PREFIX):])
            index = int(fields["index"])
            phase = str(fields["phase"])
            entry = iterations.setdefault(
                index,
                {
                    "iteration_index": index,
                    "warmup": int(fields.get("warmup", 0)),
                    "timed": int(fields.get("timed", 0)),
                },
            )
            for key in ("warmup", "timed"):
                if key in fields:
                    entry[key] = int(fields[key])
            if phase == "start":
                entry["start_epoch_ms"] = int(fields["epoch_ms"])
            elif phase == "end":
                entry["end_epoch_ms"] = int(fields["epoch_ms"])
                for key, value in fields.items():
                    if key in {"phase", "index", "warmup", "timed", "runs", "profile_only", "gpu_only", "epoch_ms"}:
                        continue
                    entry[key] = value

    ordered: list[dict[str, object]] = []
    for index in sorted(iterations):
        entry = iterations[index]
        if "start_epoch_ms" not in entry or "end_epoch_ms" not in entry:
            incomplete.append(index)
            continue
        entry["duration_ms"] = int(entry["end_epoch_ms"]) - int(entry["start_epoch_ms"])
        ordered.append(entry)

    return ordered, incomplete


def parse_query_samples(query_path: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    with query_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("timestamp"):
                continue
            if len(row) < 12:
                continue
            try:
                timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                continue
            sample = {
                "timestamp_ms": int(timestamp.timestamp() * 1000),
                "pstate": row[11].strip(),
            }
            ordered_values = [
                row[3].strip(),
                row[4].strip(),
                row[5].strip(),
                row[6].strip(),
                row[7].strip(),
                row[8].strip(),
                row[9].strip(),
                row[10].strip(),
            ]
            for raw_key, raw_value in zip(QUERY_COLUMN_MAP.values(), ordered_values):
                sample[raw_key] = parse_scalar(raw_value)
            samples.append(sample)
    return samples


def parse_dmon_samples(dmon_path: Path) -> list[dict[str, object]]:
    if not dmon_path.exists() or dmon_path.stat().st_size == 0:
        return []

    samples: list[dict[str, object]] = []
    with dmon_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y%m%d %H:%M:%S")
            except ValueError:
                continue
            samples.append(
                {
                    "timestamp_ms": int(timestamp.timestamp() * 1000),
                    "dmon_sm_pct": parse_scalar(parts[3]),
                    "dmon_mem_pct": parse_scalar(parts[4]),
                }
            )
    return samples


def numeric_summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}_avg": round(mean(values), 3),
        f"{prefix}_min": round(min(values), 3),
        f"{prefix}_max": round(max(values), 3),
    }


def mode_or_empty(values: list[str]) -> str:
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def overlapping_dmon_samples(
    samples: list[dict[str, object]],
    start_ms: int,
    end_ms: int,
    interval_ms: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for sample in samples:
        sample_end = int(sample["timestamp_ms"])
        sample_start = sample_end - interval_ms
        if sample_end >= start_ms and sample_start <= end_ms:
            selected.append(sample)
    return selected


def overlapping_query_samples(
    samples: list[dict[str, object]],
    start_ms: int,
    end_ms: int,
    interval_ms: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for sample in samples:
        sample_end = int(sample["timestamp_ms"])
        sample_start = sample_end - interval_ms
        if sample_end >= start_ms and sample_start <= end_ms:
            selected.append(sample)
    return selected


def populate_query_metrics(entry: dict[str, object], query_samples: list[dict[str, object]]) -> None:
    entry["query_sample_count"] = len(query_samples)
    if query_samples:
        entry["memory_total_mb"] = query_samples[-1].get("memory_total_mb")
        entry["pstate_mode"] = mode_or_empty(
            [str(sample["pstate"]) for sample in query_samples if sample.get("pstate")]
        )
    else:
        entry["memory_total_mb"] = None
        entry["pstate_mode"] = ""

    for field in (
        "gpu_util_pct",
        "memory_util_pct",
        "memory_used_mb",
        "power_draw_w",
        "temperature_c",
        "sm_clock_mhz",
        "mem_clock_mhz",
    ):
        values = [float(sample[field]) for sample in query_samples if sample.get(field) is not None]
        entry.update(numeric_summary(values, field))


def populate_dmon_metrics(
    entry: dict[str, object],
    dmon_samples: list[dict[str, object]],
    start_ms: int,
    end_ms: int,
    interval_ms: int,
) -> None:
    overlapping = overlapping_dmon_samples(dmon_samples, start_ms, end_ms, interval_ms)
    entry["dmon_sample_count"] = len(overlapping)

    dmon_sm_values = [float(sample["dmon_sm_pct"]) for sample in overlapping if sample.get("dmon_sm_pct") is not None]
    dmon_mem_values = [float(sample["dmon_mem_pct"]) for sample in overlapping if sample.get("dmon_mem_pct") is not None]
    entry.update(numeric_summary(dmon_sm_values, "dmon_sm_pct"))
    entry.update(numeric_summary(dmon_mem_values, "dmon_mem_pct"))
    entry.update(numeric_summary(dmon_sm_values, "sm_active_pct"))
    entry.update(numeric_summary(dmon_mem_values, "dram_active_pct"))


def apply_warmup_override(iterations: list[dict[str, object]]) -> None:
    if not iterations:
        return
    for entry in iterations:
        entry["warmup"] = 0
        entry["timed"] = 1
    iterations[0]["warmup"] = 1
    iterations[0]["timed"] = 0


def timed_metric_summary(iterations: list[dict[str, object]], field: str) -> dict[str, float]:
    timed_values = [float(entry[field]) for entry in iterations if entry.get("timed") and entry.get(field) is not None]
    if not timed_values:
        return {}
    return {
        "min": round(min(timed_values), 3),
        "mean": round(mean(timed_values), 3),
        "max": round(max(timed_values), 3),
    }


def write_iteration_csv(out_path: Path, iterations: list[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITERATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for entry in iterations:
            writer.writerow(entry)


def write_summary_json(
    out_path: Path,
    iterations: list[dict[str, object]],
    incomplete_iterations: list[int],
    query_sample_count: int,
    dmon_row_count: int,
    force_first_run_warmup: bool,
) -> None:
    timed_iterations = [entry["iteration_index"] for entry in iterations if entry.get("timed")]
    warmup_iterations = [entry["iteration_index"] for entry in iterations if entry.get("warmup")]
    summary = {
        "iteration_count": len(iterations),
        "timed_iteration_indices": timed_iterations,
        "warmup_iteration_indices": warmup_iterations,
        "incomplete_iteration_indices": incomplete_iterations,
        "query_sample_count": query_sample_count,
        "dmon_sample_count": dmon_row_count,
        "force_first_run_warmup": force_first_run_warmup,
        "timed_duration_ms": timed_metric_summary(iterations, "duration_ms"),
        "timed_gpu_total_ms": timed_metric_summary(iterations, "gpu_total_ms"),
        "timed_gpu_util_pct_avg": timed_metric_summary(iterations, "gpu_util_pct_avg"),
        "timed_gpu_util_pct_max": timed_metric_summary(iterations, "gpu_util_pct_max"),
        "timed_memory_used_mb_max": timed_metric_summary(iterations, "memory_used_mb_max"),
        "timed_power_draw_w_avg": timed_metric_summary(iterations, "power_draw_w_avg"),
        "timed_dmon_sm_pct_avg": timed_metric_summary(iterations, "dmon_sm_pct_avg"),
        "timed_dmon_mem_pct_avg": timed_metric_summary(iterations, "dmon_mem_pct_avg"),
        "timed_sm_active_pct_avg": timed_metric_summary(iterations, "sm_active_pct_avg"),
        "timed_dram_active_pct_avg": timed_metric_summary(iterations, "dram_active_pct_avg"),
        "notes": {
            "query_gpu_metrics": "Sampled from nvidia-smi --query-gpu at subsecond cadence.",
            "dmon_metrics": "Sampled from nvidia-smi dmon; sm_active_pct is derived from dmon sm and dram_active_pct is derived from dmon mem.",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    bench_log = Path(args.bench_log)
    query_csv = Path(args.query_csv)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    dmon_log = Path(args.dmon_log) if args.dmon_log else None

    iterations, incomplete_iterations = parse_bench_iterations(bench_log)
    if args.force_first_run_warmup:
        apply_warmup_override(iterations)

    query_samples = parse_query_samples(query_csv)
    dmon_samples = parse_dmon_samples(dmon_log) if dmon_log else []
    query_interval_ms = max(args.query_interval_ms, 1)
    dmon_interval_ms = int(args.dmon_interval_sec * 1000.0)

    for entry in iterations:
        start_ms = int(entry["start_epoch_ms"])
        end_ms = int(entry["end_epoch_ms"])
        query_window = overlapping_query_samples(query_samples, start_ms, end_ms, query_interval_ms)
        populate_query_metrics(entry, query_window)
        populate_dmon_metrics(entry, dmon_samples, start_ms, end_ms, dmon_interval_ms)

    write_iteration_csv(out_csv, iterations)
    write_summary_json(
        out_json,
        iterations,
        incomplete_iterations,
        len(query_samples),
        len(dmon_samples),
        args.force_first_run_warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
