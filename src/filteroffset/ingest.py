"""Directory scan -> tidy per-frame table, with a persistent cache.

Designed for the case the author actually has: hundreds of thousands of subs on
disk, of which only a handful are new on any given night.  The header scan is
therefore cached in SQLite keyed by (path, mtime, size), so a rescan of an
unchanged archive costs one query rather than one file open per frame.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .cache import FrameCache
from .headers import HeaderReadError, iter_fits_files, normalise_header

#: Columns carried through to the modelling stage.
FRAME_COLUMNS = (
    "path", "file", "date_obs", "filter", "image_type", "object", "exptime",
    "focus_pos", "focus_temp", "ambient_temp", "ccd_temp", "set_temp",
    "gain", "offset", "xbinning", "ybinning", "xpixsz", "ypixsz",
    "naxis1", "naxis2", "instrument", "telescope", "focal_len", "aperture",
    "focal_ratio", "pixel_scale", "airmass", "altitude", "azimuth", "rotator",
    "humidity", "dewpoint", "pressure", "software", "guide_rms", "hfr_header",
    "focus_step_size",
)


@dataclass
class ScanResult:
    frames: pd.DataFrame
    n_scanned: int
    n_cached: int
    errors: list[tuple[str, str]]
    provenance: dict[str, str]

    @property
    def n_frames(self) -> int:
        return int(len(self.frames))


def _row_from_path(path: str, extra_aliases) -> tuple[str, dict[str, Any] | None, str | None,
                                                      dict[str, str]]:
    try:
        rec = normalise_header(path, extra_aliases=extra_aliases)
    except HeaderReadError as exc:
        return path, None, str(exc), {}
    except Exception as exc:  # pragma: no cover
        return path, None, f"{type(exc).__name__}: {exc}", {}
    return path, rec.as_row(), None, rec.provenance


def scan_directory(
    roots: Sequence[str | os.PathLike[str]],
    cache: FrameCache | None = None,
    recursive: bool = True,
    workers: int = 8,
    extra_aliases: dict[str, Sequence[str]] | None = None,
    image_types: Iterable[str] | None = ("light",),
    progress: bool = False,
) -> ScanResult:
    """Scan *roots* for FITS frames and return a tidy table.

    *image_types* filters on a lowercase substring of ``IMAGETYP`` so that
    darks, flats and bias frames in the same tree are ignored; pass ``None`` to
    keep everything.
    """
    paths = list(iter_fits_files(roots, recursive=recursive))
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    provenance: dict[str, str] = {}
    n_cached = 0

    pending: list[str] = []
    if cache is not None:
        for path in paths:
            hit = cache.get_header(path)
            if hit is None:
                pending.append(path)
            else:
                rows.append(hit)
                n_cached += 1
    else:
        pending = list(paths)

    if pending:
        iterator = _iter_headers(pending, workers, extra_aliases, progress)
        fresh: list[dict[str, Any]] = []
        for path, row, err, prov in iterator:
            if err is not None or row is None:
                errors.append((path, err or "unknown error"))
                continue
            rows.append(row)
            fresh.append(row)
            for k, v in prov.items():
                provenance.setdefault(k, v)
        if cache is not None and fresh:
            cache.put_headers(fresh)

    df = pd.DataFrame(rows)
    if df.empty:
        return ScanResult(df, len(paths), n_cached, errors, provenance)

    for col in FRAME_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["file"] = df["path"].map(os.path.basename)
    df["date_obs"] = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")

    if image_types:
        wanted = tuple(t.lower() for t in image_types)
        it = df["image_type"].astype("string").str.lower().fillna("")
        keep = it.eq("") | it.apply(lambda v: any(w in v for w in wanted))
        df = df[keep]

    df = df[list(FRAME_COLUMNS)].sort_values("date_obs", kind="mergesort")
    df = df.reset_index(drop=True)
    return ScanResult(df, len(paths), n_cached, errors, provenance)


def _iter_headers(paths, workers, extra_aliases, progress):
    workers = max(1, int(workers))
    if workers == 1:
        seq = (_row_from_path(p, extra_aliases) for p in paths)
        yield from _maybe_progress(seq, len(paths), progress, "headers")
        return
    # Header reads are IO bound and release the GIL, so threads are enough and
    # avoid the memory cost of forking for a hundred thousand tasks.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_row_from_path, p, extra_aliases) for p in paths]
        yield from _maybe_progress(
            (f.result() for f in as_completed(futures)), len(paths), progress, "headers"
        )


def _maybe_progress(iterable, total, progress, desc):
    if not progress:
        yield from iterable
        return
    try:
        from tqdm import tqdm

        yield from tqdm(iterable, total=total, desc=desc, unit="file")
    except ImportError:  # pragma: no cover
        yield from iterable
