"""Defocus physics: V-curves, critical focus zone, and step-size calibration.

Near best focus the measured star size adds in quadrature with the defocus
blur, which gives the hyperbolic "V-curve" that autofocus routines fit:

.. math::  \\mathrm{HFD}(p) = \\sqrt{H_0^2 + \\left[s\\,(p - p_\\star)\\right]^2}

with :math:`H_0` the in-focus star size (seeing and optics), :math:`p_\\star`
the best-focus position and *s* the V-curve slope in pixels per focuser step.

Two things fall out of that slope which are otherwise guesswork:

* **Step size.**  The geometric defocus blur diameter is
  :math:`\\Delta / N` for a defocus distance :math:`\\Delta` at focal ratio
  *N*, so a measured slope converts directly into microns per step.  This
  matters because a focuser's nominal figure is a property of the motor and
  gearbox, while the quantity that counts is the *drawtube* motion per step,
  which depends on the telescope it is bolted to.
* **Significance.**  An offset only matters if it is comparable to the depth of
  focus.  :func:`focus_tolerances` gives both the diffraction limit and the
  seeing-limited tolerance, which for a fast astrograph are the numbers that
  decide whether a five-step offset is worth setting at all.

When the data contain no deliberate defocus (the usual case - autofocus put
every frame near focus) the slope is not identifiable.  The module then falls
back to the *geometric* slope predicted from the optics, and says so, rather
than inventing a fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Half-flux diameter of a uniformly illuminated disk of diameter D is D/sqrt2,
#: because half the area lies inside radius R/sqrt2.
_DISK_HFD_FACTOR = 1.0 / np.sqrt(2.0)


def hyperbola_hfd(
    position: np.ndarray, best: float, hfd_min: float, slope: float
) -> np.ndarray:
    """The standard autofocus V-curve."""
    e = np.asarray(position, dtype=float) - float(best)
    return np.sqrt(float(hfd_min) ** 2 + (float(slope) * e) ** 2)


def geometric_slope_px_per_micron(focal_ratio: float, pixel_um: float) -> float:
    """Defocus blur growth in pixels per micron of focuser travel.

    A defocus of :math:`\\Delta` microns spreads a star into a disk of diameter
    :math:`\\Delta / N` microns; its half-flux diameter is that divided by
    :math:`\\sqrt 2`, and dividing by the pixel pitch puts it in pixels.
    """
    if not (focal_ratio and pixel_um) or focal_ratio <= 0 or pixel_um <= 0:
        return float("nan")
    return float(_DISK_HFD_FACTOR / (float(focal_ratio) * float(pixel_um)))


def step_size_microns_from_slope(
    slope_px_per_step: float, focal_ratio: float, pixel_um: float
) -> float:
    """Invert a measured V-curve slope into microns of travel per step."""
    g = geometric_slope_px_per_micron(focal_ratio, pixel_um)
    if not np.isfinite(g) or g <= 0 or not np.isfinite(slope_px_per_step):
        return float("nan")
    return float(slope_px_per_step / g)


def focus_tolerances(
    focal_ratio: float,
    wavelength_um: float = 0.55,
    seeing_fwhm_px: float | None = None,
    pixel_um: float | None = None,
    seeing_blur_fraction: float = 1.0 / 3.0,
) -> dict[str, float]:
    """Depth-of-focus tolerances in microns of focuser travel.

    ``diffraction_total`` uses the Rayleigh quarter-wave criterion, total
    depth :math:`4 \\lambda N^2`.  ``seeing_total`` is the usually more
    relevant practical figure: the defocus at which the geometric blur reaches
    ``seeing_blur_fraction`` of the seeing disk, doubled to give a total
    two-sided zone.  For a fast astrograph the diffraction figure is small and
    the seeing figure is what actually governs whether an offset is worth
    applying.
    """
    out: dict[str, float] = {}
    N = float(focal_ratio) if focal_ratio else float("nan")
    out["focal_ratio"] = N
    if np.isfinite(N) and N > 0:
        out["diffraction_total_um"] = float(4.0 * wavelength_um * N**2)
        out["diffraction_half_um"] = out["diffraction_total_um"] / 2.0
    else:
        out["diffraction_total_um"] = float("nan")
        out["diffraction_half_um"] = float("nan")
    if seeing_fwhm_px and pixel_um and np.isfinite(N) and N > 0:
        seeing_um = float(seeing_fwhm_px) * float(pixel_um)
        half = N * seeing_um * float(seeing_blur_fraction)
        out["seeing_fwhm_um"] = seeing_um
        out["seeing_half_um"] = float(half)
        out["seeing_total_um"] = float(2.0 * half)
    return out


def steps_to_microns_table(
    offsets_steps: dict[str, float],
    candidate_step_sizes_um: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0),
    tolerance_um: float | None = None,
) -> pd.DataFrame:
    """Translate step offsets into microns across plausible step sizes.

    Written for the common situation where the focuser's microns-per-step is
    not reliably known.  Capture software only ever needs the step figure, so
    this table exists purely to answer "is this offset even significant?"
    without requiring the user to commit to a calibration.
    """
    rows = []
    for um_per_step in candidate_step_sizes_um:
        row: dict[str, Any] = {"um_per_step": um_per_step}
        for filt, steps in sorted(offsets_steps.items()):
            row[f"{filt}_um"] = float(steps) * float(um_per_step)
        if tolerance_um and np.isfinite(tolerance_um) and tolerance_um > 0:
            worst = max((abs(float(v)) for v in offsets_steps.values()), default=0.0)
            row["max_offset_um"] = worst * um_per_step
            row["fraction_of_tolerance"] = worst * um_per_step / tolerance_um
            row["matters"] = bool(worst * um_per_step > 0.5 * tolerance_um)
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class VCurveFit:
    """Result of fitting a hyperbolic V-curve to one filter's sweep."""

    filter: str
    n_points: int
    best_position: float = float("nan")
    best_position_se: float = float("nan")
    hfd_min: float = float("nan")
    slope_px_per_step: float = float("nan")
    slope_se: float = float("nan")
    position_span: float = float("nan")
    hfd_ratio: float = float("nan")
    identifiable: bool = False
    step_size_um: float = float("nan")
    rms: float = float("nan")
    notes: list[str] = field(default_factory=list)


def fit_vcurve(
    positions: np.ndarray,
    hfd: np.ndarray,
    filter_name: str = "",
    focal_ratio: float | None = None,
    pixel_um: float | None = None,
    min_hfd_ratio: float = 1.3,
) -> VCurveFit:
    """Fit the hyperbola to (position, HFD) pairs for a single filter.

    ``min_hfd_ratio`` guards against the flat-bottom case: if the worst star
    size in the sweep is not at least this multiple of the best, the curvature
    is not sampled and the fit would report a precise-looking best position
    that is really unconstrained.
    """
    p = np.asarray(positions, dtype=float)
    h = np.asarray(hfd, dtype=float)
    ok = np.isfinite(p) & np.isfinite(h) & (h > 0)
    p, h = p[ok], h[ok]
    res = VCurveFit(filter=filter_name, n_points=int(p.size))
    if p.size < 4 or np.ptp(p) <= 0:
        res.notes.append("Need at least 4 distinct focuser positions to fit a V-curve.")
        return res

    res.position_span = float(np.ptp(p))
    res.hfd_ratio = float(np.nanmax(h) / max(np.nanmin(h), 1e-9))
    if res.hfd_ratio < min_hfd_ratio:
        res.notes.append(
            f"Star size varies by only x{res.hfd_ratio:.2f} across the sweep "
            f"(need >= x{min_hfd_ratio:.2f}); the curve's bottom is too flat to "
            "locate best focus. Widen the sweep."
        )

    from scipy.optimize import least_squares

    p0 = [float(p[np.argmin(h)]), float(np.nanmin(h)), 1.0]
    span = max(res.position_span, 1.0)

    def resid(theta):
        return hyperbola_hfd(p, theta[0], theta[1], theta[2]) - h

    try:
        sol = least_squares(
            resid,
            p0,
            bounds=(
                [p.min() - span, 1e-3, 1e-6],
                [p.max() + span, max(np.nanmax(h) * 2, 1.0), 1e3],
            ),
            # Plain least squares, so that the covariance computed below from
            # J^T J is the covariance of the estimator actually used. With a
            # robust loss the two disagree and the interval under-covers.
            loss="linear",
        )
    except Exception as exc:  # pragma: no cover
        res.notes.append(f"V-curve fit failed: {exc}")
        return res

    res.best_position = float(sol.x[0])
    res.hfd_min = float(sol.x[1])
    res.slope_px_per_step = float(sol.x[2])
    r = resid(sol.x)
    dof = max(p.size - 3, 1)
    res.rms = float(np.sqrt(np.mean(r**2)))
    try:
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J) * float(r @ r) / dof
        res.best_position_se = float(np.sqrt(max(cov[0, 0], 0.0)))
        res.slope_se = float(np.sqrt(max(cov[2, 2], 0.0)))
    except Exception:  # pragma: no cover
        pass
    res.identifiable = bool(res.hfd_ratio >= min_hfd_ratio and p.size >= 5)
    if focal_ratio and pixel_um:
        res.step_size_um = step_size_microns_from_slope(
            res.slope_px_per_step, focal_ratio, pixel_um
        )
    return res


def detect_sweeps(
    frames: pd.DataFrame,
    min_positions: int = 4,
    min_frames: int = 4,
    max_dead_time_s: float = 120.0,
    max_span_minutes: float = 90.0,
) -> pd.DataFrame:
    """Find runs of frames that constitute a genuine focus sweep.

    A sweep means *deliberately* stepping the focuser while imaging the same
    filter, so the defining feature is that the position changes from one frame
    to the next **without an autofocus run in between**.

    Simply grouping by (session, filter) and counting distinct positions is not
    enough, and gets it badly wrong: a normal night visits several positions per
    filter because each filter change is followed by its own autofocus run at a
    different temperature.  Fitting a V-curve to those points recovers the
    night's seeing trend rather than a focus curve, and would report a confident
    best-focus position that is pure artefact.  Candidate runs are therefore
    required to be contiguous in time, with small inter-frame dead time and a
    short total duration, so that temperature and seeing are effectively frozen
    across the sweep.
    """
    if frames.empty or "focus_pos" not in frames.columns:
        return pd.DataFrame()
    df = frames.copy()
    df["focus_pos"] = pd.to_numeric(df["focus_pos"], errors="coerce")
    if "date_obs" not in df.columns:
        return pd.DataFrame()
    df["_t"] = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")
    df = df.dropna(subset=["focus_pos", "_t"]).sort_values("_t").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    exptime = pd.to_numeric(df.get("exptime"), errors="coerce").fillna(0.0)
    dead = (df["_t"].diff().dt.total_seconds() - exptime.shift(1)).fillna(1e9)
    filt = df["filter"].astype("string") if "filter" in df else pd.Series("", index=df.index)
    sess = df["session"] if "session" in df else pd.Series(0, index=df.index)

    # A new candidate run starts whenever continuity is broken.
    brk = (
        filt.ne(filt.shift(1)).fillna(True).to_numpy(dtype=bool)
        | sess.ne(sess.shift(1)).fillna(True).to_numpy(dtype=bool)
        | (dead.to_numpy(dtype=float) > float(max_dead_time_s))
    )
    df["_run"] = np.cumsum(brk)

    rows = []
    for run_id, g in df.groupby("_run"):
        n_pos = int(g["focus_pos"].nunique())
        span_min = float((g["_t"].max() - g["_t"].min()).total_seconds() / 60.0)
        if (
            n_pos >= min_positions
            and len(g) >= min_frames
            and span_min <= float(max_span_minutes)
        ):
            row: dict[str, Any] = {
                "n_frames": int(len(g)),
                "n_positions": n_pos,
                "position_span": float(np.ptp(g["focus_pos"])),
                "duration_minutes": span_min,
                "run": int(run_id),
            }
            if "filter" in g:
                row["filter"] = g["filter"].iloc[0]
            if "session" in g:
                row["session"] = g["session"].iloc[0]
            rows.append(row)
    return pd.DataFrame(rows)


def implied_defocus_from_excess(
    excess_fraction: float,
    hfd_in_focus_px: float,
    slope_px_per_step: float,
) -> dict[str, float]:
    """Convert a fractional star-size excess into an implied defocus.

    Inverts the quadrature sum: if a filter's stars are ``(1+x)`` times the
    size the conditions model predicts, and star size grows as
    :math:`\\sqrt{H_0^2 + (s e)^2}`, then

    .. math::  |e| = \\frac{H_0}{s}\\sqrt{(1+x)^2 - 1}

    The **sign is not recoverable** from near-focus data: too far in and too
    far out look identical.  The magnitude is also an upper bound on any
    autofocus error, because part or all of the excess may simply be the
    filter's intrinsic performance.
    """
    out = {
        "excess_fraction": float(excess_fraction),
        "implied_defocus_steps": float("nan"),
        "implied_blur_px": float("nan"),
    }
    if not np.isfinite(excess_fraction) or excess_fraction <= 0:
        out["implied_defocus_steps"] = 0.0
        out["implied_blur_px"] = 0.0
        return out
    if not np.isfinite(hfd_in_focus_px) or hfd_in_focus_px <= 0:
        return out
    factor = (1.0 + float(excess_fraction)) ** 2 - 1.0
    if factor <= 0:
        return out
    blur = hfd_in_focus_px * np.sqrt(factor)
    out["implied_blur_px"] = float(blur)
    if np.isfinite(slope_px_per_step) and slope_px_per_step > 0:
        out["implied_defocus_steps"] = float(blur / slope_px_per_step)
    return out
