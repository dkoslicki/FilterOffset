"""ASTAP ``-extract`` backend.

Notes on ASTAP's command line, learned the hard way and worth recording:

* ``-analyse`` prints ``HFD_MEDIAN`` rounded to **one decimal place**.  That is
  far too coarse for filter offsets, where the whole signal can be a few
  hundredths of a pixel, so this backend uses ``-extract`` and computes the
  statistics from the per-star CSV instead.
* the ``HFD_MEDIAN`` that ``-extract`` prints on stdout does **not** agree with
  the median of the CSV it writes (observed 21.5 vs 4.42 on the same frame).
  The CSV is the trustworthy product; the stdout value is ignored.
* ``-o`` does *not* redirect the ``-extract`` CSV.  ASTAP writes
  ``<input-basename>.csv`` next to the input file, so this backend runs against
  a symlink inside a scratch directory to avoid writing into the user's data
  directories.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .base import FrameProfile, ProfileParams, summarise_stars

DEFAULT_ASTAP = "/usr/local/bin/astap_cli"


def astap_path() -> str | None:
    """Locate the ASTAP CLI, honouring ``$ASTAP_CLI``."""
    env = os.environ.get("ASTAP_CLI")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    if os.path.isfile(DEFAULT_ASTAP) and os.access(DEFAULT_ASTAP, os.X_OK):
        return DEFAULT_ASTAP
    for name in ("astap_cli", "astap"):
        found = shutil.which(name)
        if found:
            return found
    return None


def astap_status() -> str:
    exe = astap_path()
    if not exe:
        return "unavailable: astap_cli not found (set $ASTAP_CLI)"
    try:
        proc = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        first = (proc.stdout or proc.stderr).splitlines()
        ver = next((ln for ln in first if "version" in ln.lower()), first[0] if first else "")
        return f"ok ({exe}; {ver.strip()})"
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {exc}"


def read_astap_csv(csv_path: str) -> pd.DataFrame:
    """Parse an ASTAP ``-extract`` CSV into the shared star-table schema.

    The header advertises ``ra``/``dec`` columns that are absent unless the
    frame was also plate solved, and short rows do occur, so parsing is done
    defensively rather than with a strict reader.
    """
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return pd.DataFrame(columns=["x", "y", "hfd", "snr", "flux"])
    head = [h.strip().lower() for h in rows[0]]
    # Map advertised names to positions; fall back to documented column order.
    def idx(*names: str, default: int | None = None) -> int | None:
        for n in names:
            for j, h in enumerate(head):
                if h == n or h.startswith(n + "["):
                    return j
        return default

    ix = idx("x", default=0)
    iy = idx("y", default=1)
    ih = idx("hfd", default=2)
    isnr = idx("snr", default=3)
    iflux = idx("flux", default=4)
    need = max(v for v in (ix, iy, ih, isnr, iflux) if v is not None)

    recs: list[tuple[float, float, float, float, float]] = []
    for row in rows[1:]:
        if len(row) <= need:
            continue
        try:
            recs.append(
                (
                    float(row[ix]), float(row[iy]), float(row[ih]),
                    float(row[isnr]), float(row[iflux]),
                )
            )
        except (TypeError, ValueError):
            continue
    df = pd.DataFrame(recs, columns=["x", "y", "hfd", "snr", "flux"])
    # ASTAP gives no peak pixel, shape or FWHM.  Saturation is instead screened
    # via flux/HFD below, and shape metrics come from the internal backend.
    df["peak"] = np.nan
    for col in ("a", "b", "theta", "fwhm"):
        df[col] = np.nan
    return df


def measure_frame(
    path: str,
    params: ProfileParams | None = None,
    naxis1: float | None = None,
    naxis2: float | None = None,
    full_well: float | None = None,
    timeout: float = 300.0,
    workdir: str | None = None,
) -> FrameProfile:
    """Measure one frame's star profiles with ASTAP."""
    params = params or ProfileParams()
    exe = astap_path()
    if not exe:
        return FrameProfile(path, "astap", ok=False, error="astap_cli not found")
    if not os.path.isfile(path):
        return FrameProfile(path, "astap", ok=False, error="file not found")

    tmp = tempfile.mkdtemp(prefix="fo_astap_", dir=workdir)
    try:
        # Preserve the real extension: ASTAP dispatches on it.
        base = os.path.basename(path)
        ext = ".fits" if base.lower().endswith((".fits", ".fits.fz")) else ".fit"
        link = os.path.join(tmp, "frame" + ext)
        try:
            os.symlink(os.path.abspath(path), link)
        except OSError:
            shutil.copy2(path, link)

        cmd = [exe, "-f", link, "-extract", f"{params.snr_min:g}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return FrameProfile(path, "astap", ok=False, error=f"timeout after {timeout:g}s")

        csv_path = os.path.splitext(link)[0] + ".csv"
        if not os.path.isfile(csv_path):
            msg = (proc.stderr or proc.stdout or "no output").strip().splitlines()
            return FrameProfile(
                path, "astap", ok=False,
                error=f"no CSV produced (rc={proc.returncode}): {msg[-1] if msg else ''}",
            )
        stars = read_astap_csv(csv_path)
        stats = summarise_stars(
            stars, params, naxis1=naxis1, naxis2=naxis2, full_well=full_well,
            backend="astap",
        )
        stats["astap_stdout_stars"] = _parse_kv(proc.stdout, "STARS")
        # Recorded but never used for science - see module docstring.
        stats["astap_stdout_hfd"] = _parse_kv(proc.stdout, "HFD_MEDIAN")
        ok = stats.get("n_stars", 0) >= params.min_stars
        err = None if ok else f"only {stats.get('n_stars', 0)} usable stars"
        return FrameProfile(path, "astap", ok=ok, error=err, stats=stats)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _parse_kv(text: str, key: str) -> float:
    for line in (text or "").splitlines():
        if line.startswith(key + "="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                return float("nan")
    return float("nan")
