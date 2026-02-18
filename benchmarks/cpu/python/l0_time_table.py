#!/usr/bin/env python3
"""Build time-weighted L0 occupancy tables from RocksDB EVENT_LOG_v1 logs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EVENT_MARKER = "EVENT_LOG_v1"


@dataclass
class Sample:
    time_us: int
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create plain-text, time-weighted L0 distribution tables from "
            "rocksdb_LOG_after_mix.txt logs."
        )
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="One or more metadata log paths (rocksdb_LOG_after_mix.txt).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_default_out_dir()),
        help="Directory for output .txt tables.",
    )
    parser.add_argument(
        "--source-key",
        default="approx-l0-data-size",
        help=(
            "Preferred JSON key to read from EVENT_LOG_v1 objects. "
            "If missing, lsm_state[0] is used."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "count", "bytes"),
        default="auto",
        help="Interpret source values as count, bytes, or auto-detect.",
    )
    parser.add_argument(
        "--l0-unit-mb",
        type=float,
        default=8.0,
        help="Unit size for converting bytes to approx L0 files (default: 8 MB).",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=1,
        help="Bucket width for L0 file counts (default: 1).",
    )
    parser.add_argument(
        "--window",
        choices=("auto", "full"),
        default="auto",
        help=(
            "Analysis window: auto uses mix.log Date + mix_report.csv duration, "
            "full uses all available samples."
        ),
    )
    return parser.parse_args()


def _default_out_dir() -> Path:
    script = Path(__file__).resolve()
    for ancestor in script.parents:
        bench_results = ancestor / "bench_results"
        if bench_results.exists():
            return bench_results / "analysis" / "approx_l0_data_size_tables"
    return Path.cwd() / "bench_results" / "analysis" / "approx_l0_data_size_tables"


def _resolve_log_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "metadata" / "rocksdb_LOG_after_mix.txt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"{path} is a directory, but {candidate} does not exist."
        )
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _iter_event_json(log_path: Path) -> Iterable[dict[str, Any]]:
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if EVENT_MARKER not in line:
                continue
            idx = line.find("{")
            if idx < 0:
                continue
            payload = line[idx:]
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            time_micros = obj.get("time_micros")
            if isinstance(time_micros, (int, float)):
                yield obj


def _find_numeric_key(obj: Any, key: str) -> float | None:
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        for nested in obj.values():
            found = _find_numeric_key(nested, key)
            if found is not None:
                return found
        return None
    if isinstance(obj, list):
        for nested in obj:
            found = _find_numeric_key(nested, key)
            if found is not None:
                return found
    return None


def _collect_samples(log_path: Path, source_key: str) -> tuple[list[Sample], str]:
    preferred: list[Sample] = []
    fallback: list[Sample] = []

    for obj in _iter_event_json(log_path):
        t = int(obj["time_micros"])

        preferred_value = _find_numeric_key(obj, source_key)
        if preferred_value is not None:
            preferred.append(Sample(time_us=t, value=preferred_value))

        lsm_state = obj.get("lsm_state")
        if (
            isinstance(lsm_state, list)
            and lsm_state
            and isinstance(lsm_state[0], (int, float))
        ):
            fallback.append(Sample(time_us=t, value=float(lsm_state[0])))

    if preferred:
        preferred.sort(key=lambda s: s.time_us)
        return preferred, source_key
    fallback.sort(key=lambda s: s.time_us)
    return fallback, "lsm_state[0]"


def _parse_mix_start_us(run_dir: Path) -> int | None:
    mix_log = run_dir / "mix.log"
    if not mix_log.exists():
        return None
    with mix_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("Date:"):
                continue
            raw = line.split("Date:", 1)[1].strip()
            try:
                start_dt = dt.datetime.strptime(raw, "%a %b %d %H:%M:%S %Y")
            except ValueError:
                return None
            return int(start_dt.timestamp() * 1_000_000)
    return None


def _parse_mix_duration_s(run_dir: Path) -> int | None:
    report = run_dir / "mix_report.csv"
    if not report.exists():
        return None

    max_secs = 0
    with report.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "secs_elapsed" not in reader.fieldnames:
            return None
        for row in reader:
            raw = row.get("secs_elapsed")
            if raw is None:
                continue
            try:
                secs = int(float(raw))
            except ValueError:
                continue
            max_secs = max(max_secs, secs)
    return max_secs if max_secs > 0 else None


def _resolve_window(
    samples: list[Sample],
    run_dir: Path,
    window_mode: str,
) -> tuple[int, int, str]:
    if not samples:
        raise ValueError("No samples found.")

    if window_mode == "full":
        return samples[0].time_us, samples[-1].time_us, "full-samples"

    start_us = _parse_mix_start_us(run_dir)
    duration_s = _parse_mix_duration_s(run_dir)
    if start_us is not None and duration_s is not None and duration_s > 0:
        return (
            start_us,
            start_us + duration_s * 1_000_000,
            f"auto-mix-log+mix-report ({duration_s}s)",
        )

    return samples[0].time_us, samples[-1].time_us, "auto-fallback-full-samples"


def _windowed_series(
    samples: list[Sample], start_us: int, end_us: int
) -> tuple[list[Sample], int]:
    if end_us <= start_us:
        return [], 0

    in_window = [s for s in samples if start_us <= s.time_us <= end_us]
    prior = None
    for s in samples:
        if s.time_us <= start_us:
            prior = s
        else:
            break

    if prior is not None:
        series = [Sample(time_us=start_us, value=prior.value)]
        series.extend(s for s in samples if start_us < s.time_us <= end_us)
        return series, len(in_window)

    if not in_window:
        return [], 0

    effective_start = in_window[0].time_us
    series = [s for s in in_window if s.time_us >= effective_start]
    return series, len(in_window)


def _bucket_value(
    raw_value: float,
    mode: str,
    source_desc: str,
    l0_unit_bytes: float,
    bucket_size: int,
) -> int:
    if mode == "auto":
        if source_desc == "lsm_state[0]":
            detected = "count"
        elif "size" in source_desc.lower() or abs(raw_value) >= 1_000_000:
            detected = "bytes"
        else:
            detected = "count"
    else:
        detected = mode

    if detected == "bytes":
        approx_files = raw_value / l0_unit_bytes
    else:
        approx_files = raw_value

    base = max(0, int(math.floor(approx_files)))
    if bucket_size <= 1:
        return base
    return (base // bucket_size) * bucket_size


def _build_distribution(
    series: list[Sample],
    end_us: int,
    mode: str,
    source_desc: str,
    l0_unit_bytes: float,
    bucket_size: int,
) -> tuple[dict[int, float], float]:
    if not series:
        return {}, 0.0

    durations: dict[int, float] = defaultdict(float)
    total = 0.0

    for idx, sample in enumerate(series):
        t0 = sample.time_us
        t1 = series[idx + 1].time_us if idx + 1 < len(series) else end_us
        if t1 <= t0:
            continue
        dt_s = (t1 - t0) / 1_000_000.0
        bucket = _bucket_value(
            raw_value=sample.value,
            mode=mode,
            source_desc=source_desc,
            l0_unit_bytes=l0_unit_bytes,
            bucket_size=bucket_size,
        )
        durations[bucket] += dt_s
        total += dt_s

    return dict(sorted(durations.items())), total


def _format_table(durations: dict[int, float], total: float) -> list[str]:
    headers = ("L0 files", "Time (s)", "% of run")
    rows: list[tuple[str, str, str]] = []
    for bucket, seconds in durations.items():
        pct = (seconds / total * 100.0) if total > 0 else 0.0
        rows.append((str(bucket), f"{seconds:.3f}", f"{pct:.2f}%"))

    widths = [
        max(len(headers[0]), max((len(r[0]) for r in rows), default=0)),
        max(len(headers[1]), max((len(r[1]) for r in rows), default=0)),
        max(len(headers[2]), max((len(r[2]) for r in rows), default=0)),
    ]

    lines = []
    lines.append(
        f"{headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  {headers[2]:>{widths[2]}}"
    )
    lines.append(
        f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}"
    )
    for r in rows:
        lines.append(f"{r[0]:<{widths[0]}}  {r[1]:>{widths[1]}}  {r[2]:>{widths[2]}}")
    return lines


def _report_text(
    *,
    log_path: Path,
    source_desc: str,
    window_desc: str,
    sample_count: int,
    duration_total: float,
    durations: dict[int, float],
    l0_unit_mb: float,
) -> str:
    if durations:
        min_bucket = min(durations)
        max_bucket = max(durations)
        weighted_avg = sum(b * s for b, s in durations.items()) / duration_total
    else:
        min_bucket = 0
        max_bucket = 0
        weighted_avg = 0.0

    run_name = _run_name_from_log(log_path)
    lines: list[str] = []
    lines.append(run_name)
    lines.append("")
    lines.append(f"- Source log: {log_path}")
    lines.append(f"- Metric source: {source_desc}")
    lines.append(f"- Window: {window_desc}")
    lines.append(f"- L0 unit: {l0_unit_mb:.3f} MB")
    lines.append(f"- Duration covered: {duration_total:.3f}s")
    lines.append(f"- Samples: {sample_count}")
    lines.append(f"- L0 min/max: {min_bucket} / {max_bucket}")
    lines.append(f"- Time-weighted average L0 files: {weighted_avg:.2f}")
    lines.append("")
    lines.extend(_format_table(durations, duration_total))
    lines.append("")
    return "\n".join(lines)


def _run_name_from_log(log_path: Path) -> str:
    run_dir = log_path.parent.parent if log_path.parent.name == "metadata" else log_path.parent
    for part in run_dir.parts:
        if part.startswith("comp_"):
            return part
    return run_dir.name


def _output_path_for_log(log_path: Path, out_dir: Path) -> Path:
    run_dir = log_path.parent.parent if log_path.parent.name == "metadata" else log_path.parent
    picked: list[str] = []
    for part in run_dir.parts:
        if (
            part.startswith("value_")
            or part.startswith("comp_")
            or part.startswith("write_")
            or "fillrandom" in part
            or "trigger" in part
        ):
            picked.append(part)
    if not picked:
        picked = [run_dir.name]
    filename = "__".join(picked) + "__l0_time_table.txt"
    return out_dir / filename


def analyze_one_log(
    *,
    log_path: Path,
    source_key: str,
    mode: str,
    l0_unit_mb: float,
    bucket_size: int,
    window_mode: str,
) -> str:
    samples, source_desc = _collect_samples(log_path, source_key=source_key)
    if not samples:
        raise ValueError("No usable samples found (neither source key nor lsm_state[0]).")

    run_dir = log_path.parent.parent if log_path.parent.name == "metadata" else log_path.parent
    start_us, end_us, window_desc = _resolve_window(samples, run_dir, window_mode)
    series, sample_count = _windowed_series(samples, start_us, end_us)
    if not series:
        raise ValueError("No samples fell inside the selected analysis window.")

    # If the first sample starts after the requested start, cover the available range only.
    effective_start_us = series[0].time_us
    effective_end_us = end_us
    if effective_end_us <= effective_start_us:
        raise ValueError("Window has zero or negative duration after alignment.")

    l0_unit_bytes = l0_unit_mb * 1024.0 * 1024.0
    durations, duration_total = _build_distribution(
        series=series,
        end_us=effective_end_us,
        mode=mode,
        source_desc=source_desc,
        l0_unit_bytes=l0_unit_bytes,
        bucket_size=bucket_size,
    )
    if duration_total <= 0:
        raise ValueError("Computed zero duration after processing samples.")

    return _report_text(
        log_path=log_path,
        source_desc=source_desc,
        window_desc=window_desc,
        sample_count=sample_count,
        duration_total=duration_total,
        durations=durations,
        l0_unit_mb=l0_unit_mb,
    )


def main() -> int:
    args = parse_args()

    if args.bucket_size < 1:
        print("--bucket-size must be >= 1", file=sys.stderr)
        return 2
    if args.l0_unit_mb <= 0:
        print("--l0-unit-mb must be > 0", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    had_error = False
    for raw in args.logs:
        try:
            log_path = _resolve_log_path(raw)
            text = analyze_one_log(
                log_path=log_path,
                source_key=args.source_key,
                mode=args.mode,
                l0_unit_mb=args.l0_unit_mb,
                bucket_size=args.bucket_size,
                window_mode=args.window,
            )
            out_path = _output_path_for_log(log_path, out_dir)
            out_path.write_text(text, encoding="utf-8")
            print(f"Wrote {out_path}")
        except Exception as exc:  # pylint: disable=broad-except
            had_error = True
            print(f"Failed for {raw}: {exc}", file=sys.stderr)

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
