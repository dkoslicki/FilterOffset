"""Parallel, cached star-profile measurement across many frames.

The worker count is chosen from *available memory* rather than core count.  A
QHY600 frame is 123 MB on disk and several hundred MB once a backend has it in
floating point, so a 768-core machine with 3 GB of RAM can still only afford a
handful of concurrent measurements.  Getting this wrong is the difference
between a run that finishes and one that is killed by the OOM reaper.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .cache import FrameCache
from .profiles import ProfileParams, get_backend

#: Backends that spend their time in a subprocess and so can use threads.
_SUBPROCESS_BACKENDS = {"astap"}


def available_memory_bytes() -> int:
    """Best-effort available RAM, respecting cgroup limits where present."""
    limits: list[int] = []
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    limits.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    for path, current in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw in ("max", ""):
                continue
            total = int(raw)
            with open(current) as fh:
                used = int(fh.read().strip())
            if 0 < total < (1 << 60):
                limits.append(max(total - used, 0))
        except (OSError, ValueError):
            continue
    return min(limits) if limits else 2 * (1 << 30)


def estimate_peak_bytes(backend: str, params: ProfileParams, median_file_size: int) -> int:
    """Rough per-worker peak RSS for one frame measurement."""
    if backend in _SUBPROCESS_BACKENDS:
        # ASTAP reads the frame and keeps a working copy.
        return int(max(median_file_size, 1 << 20) * 2.5) + (64 << 20)
    if params.crop:
        pixels = float(params.crop) ** 2
    else:
        # Unknown geometry; assume the frame is as big as the file implies.
        pixels = max(median_file_size / 2.0, 1e6)
    # image + background model + subtracted copy + extraction overhead
    return int(pixels * 4 * 4) + (192 << 20)


def plan_workers(
    backend: str,
    params: ProfileParams,
    n_files: int,
    median_file_size: int,
    requested: int | None = None,
    memory_fraction: float = 0.7,
) -> int:
    """Choose a worker count that fits in memory."""
    cpu = os.cpu_count() or 1
    ceiling = max(1, min(cpu, n_files if n_files else 1))
    per = estimate_peak_bytes(backend, params, median_file_size)
    budget = int(available_memory_bytes() * memory_fraction)
    by_memory = max(1, budget // max(per, 1))
    limit = int(min(ceiling, by_memory))
    if requested is not None and requested > 0:
        return max(1, min(int(requested), ceiling))
    # Beyond ~16 concurrent readers the disk, not the CPU, is the bottleneck.
    return max(1, min(limit, 16))


@dataclass
class MeasureResult:
    profiles: pd.DataFrame
    n_measured: int
    n_cached: int
    failures: list[tuple[str, str, str]]
    workers: dict[str, int]


def _task(args):
    backend, path, params, n1, n2, full_well, timeout, workdir = args
    fn = get_backend(backend)
    try:
        prof = fn(
            path,
            params,
            naxis1=n1,
            naxis2=n2,
            full_well=full_well,
            timeout=timeout,
            workdir=workdir,
        )
    except Exception as exc:  # pragma: no cover
        from .profiles import FrameProfile

        prof = FrameProfile(path, backend, ok=False, error=f"{type(exc).__name__}: {exc}")
    return prof.as_row()


def measure_frames(
    frames: pd.DataFrame,
    backends: Sequence[str] = ("astap", "internal"),
    params: ProfileParams | None = None,
    cache: FrameCache | None = None,
    workers: int | None = None,
    timeout: float = 300.0,
    workdir: str | None = None,
    progress: bool = False,
) -> MeasureResult:
    """Measure star profiles for every frame with every requested backend.

    Returns a long-format table with one row per (frame, backend).
    """
    params = params or ProfileParams()
    key = params.key()
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []
    n_cached = 0
    n_measured = 0
    worker_plan: dict[str, int] = {}

    if frames.empty:
        return MeasureResult(pd.DataFrame(), 0, 0, failures, worker_plan)

    sizes = []
    for path in frames["path"]:
        try:
            sizes.append(os.path.getsize(path))
        except OSError:
            continue
    median_size = int(pd.Series(sizes).median()) if sizes else (1 << 20)

    geom = {
        r.path: (
            _num(getattr(r, "naxis1", None)),
            _num(getattr(r, "naxis2", None)),
        )
        for r in frames.itertuples()
    }

    for backend in backends:
        todo: list[tuple] = []
        for path in frames["path"]:
            if cache is not None:
                hit = cache.get_profile(path, backend, key)
                if hit is not None:
                    rows.append(hit)
                    n_cached += 1
                    continue
            n1, n2 = geom.get(path, (None, None))
            todo.append((backend, path, params, n1, n2, None, timeout, workdir))

        nw = plan_workers(backend, params, len(todo), median_size, requested=workers)
        worker_plan[backend] = nw
        if not todo:
            continue

        fresh: list[dict[str, Any]] = []
        for row in _run(todo, backend, nw, progress):
            rows.append(row)
            fresh.append(row)
            n_measured += 1
            if not row.get("ok", False):
                failures.append((row.get("path", "?"), backend, str(row.get("error"))))
        if cache is not None and fresh:
            cache.put_profiles(fresh, key)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["backend", "path"], kind="mergesort").reset_index(drop=True)
    return MeasureResult(df, n_measured, n_cached, failures, worker_plan)


def _run(todo: list[tuple], backend: str, workers: int, progress: bool) -> Iterable[dict]:
    if workers <= 1:
        seq = (_task(t) for t in todo)
        yield from _progress(seq, len(todo), progress, f"{backend}")
        return
    pool_cls = (
        ThreadPoolExecutor if backend in _SUBPROCESS_BACKENDS else ProcessPoolExecutor
    )
    with pool_cls(max_workers=workers) as pool:
        futures = [pool.submit(_task, t) for t in todo]
        yield from _progress(
            (f.result() for f in as_completed(futures)), len(todo), progress, f"{backend}"
        )


def _progress(iterable, total, progress, desc):
    if not progress:
        yield from iterable
        return
    try:
        from tqdm import tqdm

        yield from tqdm(iterable, total=total, desc=desc, unit="frame")
    except ImportError:  # pragma: no cover
        yield from iterable


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None
