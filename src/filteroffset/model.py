"""Statistical models that turn observed focus behaviour into offsets.

Two complementary models are fitted.

**Model A - the autofocus-position model.**  This is the primary estimator.
After each filter change the autofocus routine is run, and the focuser position
it settles on *is* a measurement of best focus for that filter at that
temperature.  So for each focus block *b*

.. math::  p_b = \\mu + \\delta_{f(b)} + k\\,(T_b - T_{ref}) + \\varepsilon_b

where :math:`\\delta_f` is the filter offset (identified up to the choice of
reference filter, exactly as capture software defines it) and *k* is the
temperature coefficient in steps per degree.  Because temperature enters the
same fit, the offsets are estimated *controlling for* the thermal drift rather
than relative to a moving zero point - which is what makes them trustworthy
even though the night warmed or cooled while the sequence ran.

The design only works if the filters are not confounded with temperature.  That
is a property of the observing pattern, not of the fit, so it is measured and
reported: :func:`design_diagnostics` gives per-filter temperature coverage,
variance inflation factors, and an independent within-filter-only estimate of
*k* that uses no filter-offset information at all.

**Model B - the HFD excess model.**  Model A inherits any *bias* in the
autofocus routine: if autofocus systematically lands off best focus for one
filter, that bias is silently absorbed into the offset.  Model B is therefore
fitted to the measured star sizes, asking whether any filter's frames are
sharper or softer than the observing conditions can explain:

.. math::  \\text{HFD}_i = s(t_i) \\cdot a_i^{\\gamma} + \\eta_{f(i)}

with :math:`s(t)` a smooth seeing term and :math:`a` airmass.  A filter whose
:math:`\\eta` is significantly positive is soft for a reason the conditions do
not account for, which is a hint - not a proof - that its autofocus lands off
focus.  The ambiguity is genuine and is reported as such: intrinsic optical
performance and a constant autofocus bias are not separable from near-focus
data alone, and breaking the degeneracy needs a deliberate focus sweep
(:mod:`filteroffset.vcurve`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Preference order when picking a reference filter automatically.
REFERENCE_PREFERENCE = ("L", "G", "R", "B", "Ha", "OIII", "SII")


@dataclass
class ModelSpec:
    """What to fit and how."""

    #: Which header temperature to use.  A focuser-mounted probe tracks the
    #: optical train better than an ambient/weather probe, so it wins when both
    #: are present.
    temp_column: str = "auto"
    #: Use the temperature of the first frame in the block (closest in time to
    #: the autofocus run) rather than the block mean.
    temp_at_block_start: bool = True
    reference_filter: str | None = None
    #: "ols", "robust" (Huber M-estimator) or "auto" (robust when >= 8 blocks).
    method: str = "auto"
    #: Per-night intercepts.  ``"auto"`` (the default) decides from the data
    #: via :func:`zero_point_drift_test`; ``"on"``/``"off"`` force it.
    #:
    #: The working assumption is that the imaging train is not disturbed between
    #: nights - no re-seating the camera, no spacer changes, no focuser losing
    #: its home - so an absolute encoder makes focuser positions directly
    #: comparable across nights.  That assumption is worth a great deal: it is
    #: the *between*-night information that lets a filter used alone on its own
    #: nights be tied to the rest.  Per-night intercepts discard exactly that
    #: information, which is safe but can leave a filter with no measurable
    #: offset at all, so they are used only when the data show the jumps that
    #: would justify them.
    night_effects: str = "auto"
    #: Smallest within-night temperature range (C) that will support a
    #: separate within-night slope.  Below this the split is skipped.
    min_within_night_range_c: float = 0.5
    #: Where the temperature coefficient is estimated from.  ``"within_night"``
    #: uses only variation *inside* each night, which is what temperature
    #: compensation actually acts on and is free of anything that differs
    #: between nights.  ``"pooled"`` also uses between-night contrasts, which
    #: adds precision but imports any between-night confounder.  ``"auto"``
    #: picks within-night whenever there is more than one night to do it with.
    temp_from: str = "auto"
    #: Linear drift of focus position with date, in steps per day.  Slow
    #: mechanical settling shows up this way.  One parameter absorbs it without
    #: costing any offset its identifiability, unlike a level per night.
    drift_term: str | bool = "auto"
    #: Also fit a separate temperature slope per filter, and test it.
    per_filter_slopes: bool = True
    #: Include an altitude term to probe flexure.  Often collinear with
    #: temperature on a single night, which the diagnostics will say.
    altitude_term: bool = False
    #: Thermal lag time constant in minutes.  ``None`` (the default) disables
    #: it, a number applies it, ``"fit"`` profiles it over a grid.
    #:
    #: Fitting it is off by default because the search is not free: shrinking
    #: the within-night temperature amplitude scales the slope up while leaving
    #: the residuals nearly flat, so the grid search drifts toward long time
    #: constants and larger |k|, and the reported standard error is conditional
    #: on the selected value rather than accounting for having selected it. On
    #: simulated data with one true coefficient and no lag at all, fitting the
    #: lag inflated |k| by up to 22% and multiplied its scatter by 45 when the
    #: within-night temperature range was small. The profile remains available
    #: as a diagnostic, and the sensitivity table reports what it would change.
    thermal_lag_minutes: float | str | None = None
    bootstrap: int = 2000
    confidence: float = 0.95
    random_seed: int = 12345


@dataclass
class FitResult:
    """Coefficients, uncertainties and diagnostics from Model A."""

    reference: str
    filters: list[str]
    offsets: dict[str, float] = field(default_factory=dict)
    offsets_se: dict[str, float] = field(default_factory=dict)
    offsets_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    offsets_ci_boot: dict[str, tuple[float, float]] = field(default_factory=dict)
    ci_method: str = "t"
    temp_coeff: float = float("nan")
    temp_coeff_se: float = float("nan")
    temp_coeff_ci: tuple[float, float] = (float("nan"), float("nan"))
    temp_coeff_ci_boot: tuple[float, float] = (float("nan"), float("nan"))
    intercept: float = float("nan")
    temp_ref: float = float("nan")
    temp_column: str = ""
    method: str = ""
    n_blocks: int = 0
    n_params: int = 0
    resid: np.ndarray = field(default_factory=lambda: np.array([]))
    resid_rms: float = float("nan")
    resid_scale: float = float("nan")
    r2: float = float("nan")
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    design_columns: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    thermal_lag_minutes: float | None = None
    per_filter_slopes: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    #: Filters whose offset the design cannot determine (see `identifiability`).
    not_identified: list[str] = field(default_factory=list)
    #: Connected components of the night-filter graph.
    components: list[list[str]] = field(default_factory=list)
    #: Fallback offsets fitted WITHOUT night intercepts, for filters the main
    #: fit cannot identify. Valid only if the focuser zero point is stable
    #: across nights, which is exactly what night intercepts exist to doubt.
    offsets_pooled: dict[str, float] = field(default_factory=dict)
    offsets_pooled_se: dict[str, float] = field(default_factory=dict)
    #: Fitted per-night intercepts, keyed by session id (baseline night = 0.0).
    night_offsets: dict[int, float] = field(default_factory=dict)
    #: Result of the focuser zero-point stability test.
    drift_test: Any = None
    #: Fitted linear drift of focus position with date, steps per day.
    drift_per_day: float = float("nan")
    drift_per_day_se: float = float("nan")
    #: "within_night" or "pooled" - where the temperature coefficient came from.
    temp_from: str = "pooled"
    #: Confidence level the reported intervals were built at.
    confidence: float = 0.95
    #: The alternative (single-slope) estimate, for comparison.
    temp_coeff_pooled: float = float("nan")
    temp_coeff_pooled_se: float = float("nan")
    #: The between-night temperature response, when it is estimated separately.
    temp_coeff_between: float = float("nan")
    temp_coeff_between_se: float = float("nan")
    #: The temperature values actually regressed on, one per block.  These are
    #: the thermally-lagged values when a lag was applied, so anything plotting
    #: the fitted line must use these rather than the raw header column.
    temp_values: np.ndarray = field(default_factory=lambda: np.array([]))

    def offset_table(self) -> pd.DataFrame:
        rows = []
        for f in self.filters:
            lo, hi = self.offsets_ci.get(f, (np.nan, np.nan))
            blo, bhi = self.offsets_ci_boot.get(f, (np.nan, np.nan))
            rows.append(
                {
                    "filter": f,
                    "identified": f not in self.not_identified,
                    "offset_steps": self.offsets.get(f, np.nan),
                    "se_steps": self.offsets_se.get(f, np.nan),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "boot_ci_lo": blo,
                    "boot_ci_hi": bhi,
                    "n_blocks": int(
                        (self.blocks["filter"] == f).sum()
                    ) if not self.blocks.empty else 0,
                    "is_reference": f == self.reference,
                    "offset_pooled_steps": self.offsets_pooled.get(f, np.nan),
                }
            )
        return pd.DataFrame(rows).sort_values("filter").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# temperature handling
# --------------------------------------------------------------------------- #
def assess_temperature_sources(
    blocks: pd.DataFrame, min_range_c: float = 0.2
) -> tuple[str | None, dict[str, str]]:
    """Choose a thermal regressor, and say why the others were rejected.

    Two failure modes are common enough to be worth naming, and both used to
    pass silently:

    * **A sensor that is not reporting.**  A probe that is unplugged or
      unsupported often writes a constant - very often exactly zero - rather
      than nothing at all, so the column looks populated.
    * **A regulated sensor.**  ``CCD-TEMP`` on a cooled camera is held at its
      set point by the cooler.  Its residual wobble is control-loop noise, not
      weather, and regressing focus on it manufactures a coefficient out of
      nothing.  It is only a thermal proxy on an uncooled camera, where it
      drifts with the environment instead of tracking a set point.

    Returns the chosen column (or ``None`` if nothing is usable) together with
    a reason for every candidate, so the report can explain the choice.
    """
    reasons: dict[str, str] = {}
    usable: list[str] = []
    for col in ("focus_temp", "ambient_temp", "ccd_temp"):
        if col not in blocks:
            reasons[col] = "not present in the headers"
            continue
        vals = pd.to_numeric(blocks[col], errors="coerce").dropna()
        if vals.empty:
            reasons[col] = "present but never populated"
            continue
        if len(vals) < 3:
            reasons[col] = f"only {len(vals)} value(s)"
            continue
        spread = float(vals.max() - vals.min())
        if spread <= 0.0:
            zero = " (and identically zero, the usual sign of a probe that is " \
                   "not reporting)" if float(vals.iloc[0]) == 0.0 else ""
            reasons[col] = f"constant at {vals.iloc[0]:g}{zero}"
            continue
        if col == "ccd_temp":
            set_temp = (
                pd.to_numeric(blocks["set_temp"], errors="coerce").dropna()
                if "set_temp" in blocks else pd.Series(dtype=float)
            )
            if len(set_temp):
                offset = abs(float(vals.median()) - float(set_temp.median()))
                if offset < 2.0 and spread < 2.0:
                    reasons[col] = (
                        f"regulated: held within {spread:.2f} C of the "
                        f"{float(set_temp.median()):g} C set point, so its variation "
                        "is cooler noise rather than the temperature of the optics"
                    )
                    continue
        if spread <= min_range_c:
            reasons[col] = f"varies by only {spread:.2f} C"
            continue
        reasons[col] = f"usable: {spread:.2f} C of range"
        usable.append(col)
    return (usable[0] if usable else None), reasons


def choose_temp_column(blocks: pd.DataFrame, spec: ModelSpec) -> str | None:
    """Pick the temperature column with usable, physically meaningful variation."""
    if spec.temp_column and spec.temp_column != "auto":
        return spec.temp_column if spec.temp_column in blocks else None
    return assess_temperature_sources(blocks)[0]


def apply_thermal_lag(
    times: pd.Series, temps: np.ndarray, tau_minutes: float
) -> np.ndarray:
    """Exponentially lag a temperature series.

    The optical train does not follow the air instantly; it relaxes towards it
    with some time constant.  Focus tracks the *glass and tube*, so lagging the
    measured air temperature is usually a better regressor than using it raw.
    Implemented as an irregular-interval exponential moving average.
    """
    if tau_minutes is None or tau_minutes <= 0:
        return np.asarray(temps, dtype=float)
    t = pd.to_datetime(times, utc=True, errors="coerce")
    secs = (t - t.min()).dt.total_seconds().to_numpy(dtype=float)
    temps = np.asarray(temps, dtype=float)
    out = np.empty_like(temps)
    tau_s = float(tau_minutes) * 60.0
    out[0] = temps[0]
    last_good = secs[0] if np.isfinite(secs[0]) else 0.0
    for i in range(1, len(temps)):
        # An unparseable timestamp must not propagate: without this guard one
        # NaT makes every later value NaN, and the design matrix then reaches
        # LAPACK as garbage and fails with an uninterpretable SVD error.
        if np.isfinite(secs[i]) and np.isfinite(last_good):
            dt = max(secs[i] - last_good, 0.0)
        else:
            dt = tau_s  # unknown spacing: assume one time constant has passed
        if np.isfinite(secs[i]):
            last_good = secs[i]
        alpha = 1.0 - np.exp(-dt / tau_s)
        prev = out[i - 1] if np.isfinite(out[i - 1]) else temps[i]
        cur = temps[i] if np.isfinite(temps[i]) else prev
        val = prev + alpha * (cur - prev)
        # Never manufacture a NaN where the input temperature was finite.
        out[i] = val if np.isfinite(val) else (
            temps[i] if np.isfinite(temps[i]) else prev
        )
    return out


# --------------------------------------------------------------------------- #
# design matrix
# --------------------------------------------------------------------------- #
def pick_reference(blocks: pd.DataFrame, spec: ModelSpec) -> str:
    counts = blocks["filter"].value_counts()
    if spec.reference_filter:
        if spec.reference_filter in counts.index:
            return spec.reference_filter
        raise ValueError(
            f"reference filter {spec.reference_filter!r} not present; "
            f"available: {sorted(counts.index)}"
        )
    # Prefer a filter from the largest connected component: that maximises how
    # many offsets the design can actually identify once night effects are in.
    comps = filter_components(blocks)
    pool = list(counts.index)
    if comps:
        largest = comps[0]
        if len(largest) > 1 or len(comps) == 1:
            pool = [f for f in counts.index if f in largest] or pool
    sub = counts.loc[[f for f in pool if f in counts.index]]
    best = int(sub.max())
    eligible = [f for f in sub.index if sub[f] == best]
    for pref in REFERENCE_PREFERENCE:
        if pref in eligible:
            return pref
    return sorted(eligible)[0]


def build_design(
    blocks: pd.DataFrame,
    reference: str,
    temp: np.ndarray,
    temp_ref: float,
    spec: ModelSpec,
    use_night: bool,
    exclude_filters: Sequence[str] = (),
    use_drift: bool = False,
    include_temp: bool = True,
    split_temp: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Assemble the Model A design matrix.

    *exclude_filters* names filters given no dummy of their own.  This is how an
    unidentifiable filter is handled: with per-night intercepts present, a
    filter observed only on nights where it was the *sole* filter has its
    offset perfectly absorbed by those nights' intercepts.  Omitting its dummy
    is then the correct parameterisation rather than a loss - those nights still
    contribute to the temperature slope, and their level is soaked up by the
    intercepts, leaving the remaining offsets clean.
    """
    filters = sorted(blocks["filter"].dropna().unique().tolist())
    others = [f for f in filters if f != reference and f not in set(exclude_filters)]
    cols: list[np.ndarray] = [np.ones(len(blocks))]
    names: list[str] = ["intercept"]

    centred = np.asarray(temp, dtype=float) - float(temp_ref)
    if include_temp:
        if split_temp and "session" in blocks:
            # Within/between decomposition. The within-night part is what
            # temperature compensation corrects; the night-mean part absorbs
            # whatever differs between nights that happens to track temperature
            # (how long the rig had been cooling, the season) so that it cannot
            # leak into the filter offsets. Without this split a single slope is
            # a compromise between two genuinely different responses - on real
            # data -3.0 within nights against -1.3 between them.
            night_mean = (
                pd.Series(centred).groupby(blocks["session"].to_numpy())
                .transform("mean").to_numpy(dtype=float)
            )
            cols.append(centred - night_mean)
            names.append("temp")
            if np.nanstd(night_mean) > 1e-9:
                cols.append(night_mean - np.nanmean(night_mean))
                names.append("temp_between_nights")
        else:
            cols.append(centred)
            names.append("temp")

    for f in others:
        cols.append((blocks["filter"] == f).to_numpy(dtype=float))
        names.append(f"filter[{f}]")

    if use_drift and "t_mid" in blocks:
        t = pd.to_datetime(blocks["t_mid"], utc=True, errors="coerce")
        days = (t - t.min()).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        if np.isfinite(days).any() and np.nanstd(days) > 0:
            cols.append(days - np.nanmean(days))
            names.append("drift_per_day")

    if use_night:
        sessions = sorted(blocks["session"].unique().tolist())
        for s in sessions[1:]:
            cols.append((blocks["session"] == s).to_numpy(dtype=float))
            names.append(f"night[{s}]")

    if spec.altitude_term and "altitude" in blocks:
        alt = pd.to_numeric(blocks["altitude"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(alt).sum() >= 3 and np.nanstd(alt) > 1.0:
            cols.append(alt - np.nanmean(alt))
            names.append("altitude")

    X = np.column_stack(cols)
    return X, names, others


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    if w is None:
        w = np.ones(len(y))
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    s2 = float((w * resid**2).sum() / dof)
    XtX = Xw.T @ Xw
    try:
        cov = s2 * np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:  # pragma: no cover
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    return beta, cov, resid, s2, dof


def _within_night_span(blocks: pd.DataFrame, temp: np.ndarray) -> float:
    """Median per-night temperature range, the material the split is fitted on."""
    t = pd.to_numeric(pd.Series(np.asarray(temp, dtype=float)), errors="coerce")
    if "session" not in blocks or t.isna().all():
        return float("nan")
    spans = t.groupby(blocks["session"].to_numpy()).agg(lambda x: float(x.max() - x.min()))
    spans = spans[spans.notna()]
    return float(spans.median()) if len(spans) else float("nan")


def estimate_k_within_night(
    blocks: pd.DataFrame, temp: np.ndarray, y: np.ndarray
) -> tuple[float, float, int]:
    """Temperature coefficient from variation *inside* nights only.

    Groups are (night, filter), so every contrast is between two blocks of the
    same filter on the same night.  Nothing that differs between nights - the
    focuser's zero point, the season, the target's place in the sky, how long
    the rig had been cooling - can reach this estimate.

    That matters because the two estimators genuinely disagree on real data: on
    the development archive the within-night slope is about -4.2 steps/C while
    pooling in the between-night contrasts gives about -2.7.  Temperature
    compensation corrects focus as the night cools, so the within-night figure
    is the one that describes what the focuser has to do.
    """
    t = np.asarray(temp, dtype=float)
    y = np.asarray(y, dtype=float)
    key = pd.Series(
        [f"{a}|{b}" for a, b in zip(blocks["session"], blocks["filter"].astype(str))]
    )
    ok = np.isfinite(t) & np.isfinite(y)
    t, y, key = t[ok], y[ok], key[ok].reset_index(drop=True)
    # Only groups with at least two blocks carry within-group information.
    sizes = key.map(key.value_counts())
    keep = (sizes >= 2).to_numpy()
    if keep.sum() < 3:
        return float("nan"), float("nan"), 0
    t, y, key = t[keep], y[keep], key[keep].reset_index(drop=True)
    td = t - key.map(pd.Series(t).groupby(key).mean()).to_numpy(dtype=float)
    yd = y - key.map(pd.Series(y).groupby(key).mean()).to_numpy(dtype=float)
    denom = float(td @ td)
    n_groups = int(key.nunique())
    if denom <= 1e-12:
        return float("nan"), float("nan"), n_groups
    k = float((td @ yd) / denom)
    resid = yd - k * td
    dof = max(len(y) - n_groups - 1, 1)
    se = float(np.sqrt((resid @ resid) / dof / denom))
    return k, se, n_groups


def profile_thermal_lag_within_night(
    blocks: pd.DataFrame,
    raw_temp: np.ndarray,
    y: np.ndarray,
    grid: Sequence[float] | None = None,
) -> tuple[float | None, pd.DataFrame]:
    """Choose the thermal time constant from within-night dynamics.

    The lag describes how the glass and tube follow the air *while a night
    cools*, so it has to be fitted on that same variation.  Selecting it from
    pooled residuals instead lets between-night structure drive the choice, and
    the within-night slope it feeds is strongly lag-sensitive (about -3.5
    steps/C at zero lag against -4.3 at ninety minutes on the development
    archive), so the two must be estimated consistently or the coefficient is
    simply wrong.
    """
    if grid is None:
        grid = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0]
    rows = []
    for tau in grid:
        temp = apply_thermal_lag(blocks["t_mid"], raw_temp, tau) if tau else raw_temp
        if np.nanstd(temp) < 1e-9:
            continue
        k, se, n_groups = estimate_k_within_night(blocks, temp, y)
        if not np.isfinite(k):
            continue
        # Residual scatter about the within-night fit at this lag.
        t = np.asarray(temp, dtype=float)
        key = pd.Series(
            [f"{a}|{b}" for a, b in zip(blocks["session"], blocks["filter"].astype(str))]
        )
        sizes = key.map(key.value_counts())
        keep = (sizes >= 2).to_numpy() & np.isfinite(t) & np.isfinite(y)
        tt, yy, kk = t[keep], np.asarray(y, float)[keep], key[keep].reset_index(drop=True)
        td = tt - kk.map(pd.Series(tt).groupby(kk).mean()).to_numpy(float)
        yd = yy - kk.map(pd.Series(yy).groupby(kk).mean()).to_numpy(float)
        resid = yd - k * td
        rows.append(
            {
                "tau_minutes": float(tau), "k": float(k), "k_se": float(se),
                "rss": float(resid @ resid),
                "resid_rms": float(np.sqrt(np.mean(resid**2))),
                "n_groups": int(n_groups),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return None, df
    # Charge one parameter for choosing tau, using the within-night sample size.
    n_obs = max(int(np.isfinite(y).sum()), 3)
    df["aicc"] = n_obs * np.log(df["rss"] / n_obs) + 2 * (df["tau_minutes"] > 0).astype(int)
    chosen = float(df.loc[df["aicc"].idxmin(), "tau_minutes"])
    return (chosen if chosen > 0 else None), df


def _huber(X: np.ndarray, y: np.ndarray, c: float = 1.345, iters: int = 60):
    """Huber M-estimator via IRLS, with a MAD scale.

    Used so that a single failed autofocus run cannot drag the offsets.
    """
    beta, cov, resid, s2, dof = _wls(X, y)

    # Never let the robust scale fall below the resolution of the data itself.
    # Focuser positions are integers and autofocus often returns the same one
    # repeatedly, so at least half the residuals commonly coincide and the
    # median absolute deviation collapses to zero. Huber then judges every
    # non-zero residual against a scale of nothing, weights whole blocks out,
    # and reports an impossible zero scatter with zero-width intervals. Below
    # the quantisation of the measurement there is simply nothing to be robust
    # about.
    floor = _scale_floor(y)

    def _mad(r: np.ndarray) -> float:
        return float(1.4826 * np.median(np.abs(r - np.median(r))))

    scale = _mad(resid)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.sqrt(max(s2, 0.0)))
    scale = max(scale, floor)

    w = np.ones(len(y))
    for _ in range(iters):
        r = (y - X @ beta) / max(scale, 1e-9)
        w_new = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
        if np.allclose(w_new, w, atol=1e-8):
            w = w_new
            break
        w = w_new
        beta, cov, resid, s2, dof = _wls(X, y, w)
        new_scale = _mad(resid)
        if np.isfinite(new_scale):
            scale = max(new_scale, floor)
    beta, cov, resid, s2, dof = _wls(X, y, w)

    # The covariance from a weighted least-squares fit treats the weights as
    # known, which they are not - they were estimated from the residuals. That
    # understates the standard error of an M-estimator by around 13% here
    # (95% intervals covering 92% of the time). Correct it with the usual
    # asymptotic factor E[psi^2] / E[psi']^2.
    r = (y - X @ beta) / max(scale, 1e-9)
    psi = np.clip(r, -c, c)
    psi_prime = (np.abs(r) <= c).astype(float)
    denom = float(psi_prime.mean())
    n, p = X.shape
    if denom > 0:
        correction = float((psi**2).mean()) / (denom**2)
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            cov = (scale**2) * correction * (n / max(n - p, 1)) * XtX_inv
        except np.linalg.LinAlgError:  # pragma: no cover
            pass
    return beta, cov, resid, s2, dof, w, scale


def _scale_floor(y: np.ndarray) -> float:
    """Smallest scale worth distinguishing, from the granularity of *y*.

    Taken as half the smallest non-zero gap between distinct observed values,
    which is half a step for integer focuser positions.
    """
    vals = np.unique(np.asarray(y, dtype=float))
    if vals.size < 2:
        return 0.5
    gaps = np.diff(vals)
    gaps = gaps[gaps > 0]
    return float(0.5 * gaps.min()) if gaps.size else 0.5


# --------------------------------------------------------------------------- #
# design adequacy diagnostics
# --------------------------------------------------------------------------- #
def variance_inflation(X: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    """VIF for every non-intercept column.

    A high VIF on a filter dummy is the quantitative form of the worry "are my
    filter offsets just absorbing the temperature drift?".
    """
    out: dict[str, float] = {}
    for j, name in enumerate(names):
        if name == "intercept":
            continue
        others = [k for k in range(X.shape[1]) if k != j]
        Xo, xj = X[:, others], X[:, j]
        if np.allclose(xj.std(), 0.0):
            out[name] = float("inf")
            continue
        beta, *_ = np.linalg.lstsq(Xo, xj, rcond=None)
        resid = xj - Xo @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((xj - xj.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # A column that is an exact combination of the others is *aliased*, not
        # merely collinear. Reporting 1e12 invites the reader to treat it as a
        # very large number rather than as "this coefficient does not exist".
        out[name] = float("inf") if (1.0 - r2) <= 1e-10 else float(1.0 / (1.0 - r2))
    return out


def within_filter_temp_coeff(
    blocks: pd.DataFrame, temp: np.ndarray, min_span: float = 0.5,
    by_night: bool = False,
) -> dict[str, Any]:
    """Estimate *k* using only within-filter comparisons.

    This is the key independence check.  It never compares one filter with
    another, so it cannot be contaminated by filter offsets; if it agrees with
    the joint fit, the joint fit's separation of filter from temperature is
    real.  Implemented as a fixed-effects (demeaned) regression, which is
    algebraically the pooled version of "compare every pair of same-filter
    blocks".

    With *by_night* the groups are (filter, night) rather than filter alone.
    That matters whenever the main fit carries per-night intercepts: demeaning
    by filter only would let a focuser zero-point shift between nights enter as
    though it were a temperature response, and the "independent check" would
    then disagree with the joint fit for a reason that has nothing to do with
    filters.  The comparison is only meaningful when both estimators condition
    on the same things.
    """
    pos = pd.to_numeric(blocks["focus_pos"], errors="coerce").to_numpy(dtype=float)
    t = np.asarray(temp, dtype=float)
    filt = blocks["filter"].astype(str).to_numpy()
    sess = (
        blocks["session"].astype(str).to_numpy()
        if "session" in blocks.columns else np.zeros(len(blocks), dtype=str)
    )
    ok = np.isfinite(pos) & np.isfinite(t)
    pos, t, filt, sess = pos[ok], t[ok], filt[ok], sess[ok]

    per_filter: dict[str, Any] = {}
    yd, xd = [], []
    n_groups = 0
    for f in sorted(set(filt)):
        m = filt == f
        if m.sum() < 2:
            continue
        span = float(t[m].max() - t[m].min())
        entry: dict[str, Any] = {"n_blocks": int(m.sum()), "temp_span": span}
        if m.sum() >= 3 and span >= min_span:
            cols = [np.ones(m.sum()), t[m]]
            names = ["const", "temp"]
            if by_night:
                nights = sorted(set(sess[m]))
                for nn in nights[1:]:
                    cols.append((sess[m] == nn).astype(float))
                    names.append(f"night[{nn}]")
            A = np.column_stack(cols)
            if A.shape[0] > np.linalg.matrix_rank(A):
                beta, cov, resid, s2, dof = _wls(A, pos[m])
                entry["k"] = float(beta[1])
                entry["k_se"] = float(np.sqrt(max(cov[1, 1], 0.0)))
        per_filter[f] = entry
        # Demeaned contributions to the pooled fixed-effects estimate.
        groups = (
            [(f, s_) for s_ in sorted(set(sess[m]))] if by_night else [(f, None)]
        )
        for _, s_ in groups:
            gm = m if s_ is None else (m & (sess == s_))
            if gm.sum() < 2:
                continue
            n_groups += 1
            yd.append(pos[gm] - pos[gm].mean())
            xd.append(t[gm] - t[gm].mean())

    result: dict[str, Any] = {"per_filter": per_filter}
    if yd:
        Y = np.concatenate(yd)
        Xd = np.concatenate(xd)
        if np.nanstd(Xd) > 1e-9:
            A = Xd[:, None]
            beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
            k = float(beta[0])
            resid = Y - A @ beta
            dof = max(len(Y) - max(n_groups, 1) - 1, 1)
            s2 = float(resid @ resid) / dof
            se = float(np.sqrt(s2 / max(float(Xd @ Xd), 1e-12)))
            result["k_pooled"] = k
            result["k_pooled_se"] = se
            result["dof"] = dof
            result["by_night"] = bool(by_night)
            result["n_groups"] = int(n_groups)
    return result


def design_diagnostics(
    blocks: pd.DataFrame,
    X: np.ndarray,
    names: Sequence[str],
    temp: np.ndarray,
    reference: str,
    by_night: bool = False,
) -> dict[str, Any]:
    """Quantify whether the observing pattern can support the requested fit."""
    diag: dict[str, Any] = {}
    t = np.asarray(temp, dtype=float)
    diag["temp_span"] = float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else 0.0
    diag["n_blocks"] = int(len(blocks))
    diag["n_nights"] = int(blocks["session"].nunique())

    per_filter: dict[str, Any] = {}
    for f in sorted(blocks["filter"].dropna().unique().tolist()):
        m = (blocks["filter"] == f).to_numpy(dtype=bool)
        tf = t[m]
        per_filter[f] = {
            "n_blocks": int(m.sum()),
            "n_frames": int(
                pd.to_numeric(blocks.loc[m, "n_frames"], errors="coerce").sum()
            ),
            "temp_min": float(np.nanmin(tf)) if m.sum() else float("nan"),
            "temp_max": float(np.nanmax(tf)) if m.sum() else float("nan"),
            "temp_span": float(np.nanmax(tf) - np.nanmin(tf)) if m.sum() else float("nan"),
            "temp_mean": float(np.nanmean(tf)) if m.sum() else float("nan"),
            "n_nights": int(blocks.loc[m, "session"].nunique()),
        }
        # Correlation of this filter's indicator with temperature: the direct
        # measure of filter/temperature confounding.
        if m.sum() and m.sum() < len(blocks) and np.nanstd(t) > 0:
            per_filter[f]["corr_with_temp"] = float(
                np.corrcoef(m.astype(float), t)[0, 1]
            )
        else:
            per_filter[f]["corr_with_temp"] = float("nan")
    diag["per_filter"] = per_filter

    diag["vif"] = variance_inflation(X, names)
    # Condition number of the standardised design.
    Xs = X[:, 1:] if names and names[0] == "intercept" else X
    if Xs.size:
        sd = Xs.std(axis=0)
        sd[sd == 0] = 1.0
        sv = np.linalg.svd((Xs - Xs.mean(axis=0)) / sd, compute_uv=False)
        diag["condition_number"] = float(sv.max() / max(sv.min(), 1e-12))
    diag["within_filter_k"] = within_filter_temp_coeff(blocks, t, by_night=by_night)
    diag["reference"] = reference
    return diag


def durbin_watson(resid: np.ndarray) -> float:
    r = np.asarray(resid, dtype=float)
    if r.size < 3:
        return float("nan")
    d = np.diff(r)
    denom = float(r @ r)
    return float((d @ d) / denom) if denom > 0 else float("nan")


def cooks_distance(X: np.ndarray, resid: np.ndarray, s2: float) -> np.ndarray:
    """Per-block influence, for spotting a single dominating autofocus run."""
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.full(len(resid), np.nan)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    p = np.linalg.matrix_rank(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (resid**2 / (p * max(s2, 1e-12))) * (h / np.maximum((1 - h) ** 2, 1e-12))
    return np.asarray(d, dtype=float)


# --------------------------------------------------------------------------- #
# Model A
# --------------------------------------------------------------------------- #
def _profile_thermal_lag(
    blocks: pd.DataFrame,
    raw_temp: np.ndarray,
    reference: str,
    spec: ModelSpec,
    use_night: bool,
    grid: Sequence[float] | None = None,
    exclude: Sequence[str] = (),
    use_drift: bool = False,
    split_temp: bool = False,
) -> tuple[float | None, list[tuple[float, float]]]:
    """Choose a thermal lag time constant by profiling the residual variance."""
    if grid is None:
        grid = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0]
    trace: list[tuple[float, float]] = []
    best: tuple[float, float] | None = None
    y = pd.to_numeric(blocks["focus_pos"], errors="coerce").to_numpy(dtype=float)
    n = len(y)
    for tau in grid:
        temp = apply_thermal_lag(blocks["t_mid"], raw_temp, tau) if tau else raw_temp
        if np.nanstd(temp) < 1e-9:
            continue
        X, names, _ = build_design(
            blocks, reference, temp, float(np.nanmean(temp)), spec, use_night,
            exclude_filters=exclude, use_drift=use_drift,
            split_temp=split_temp,
        )
        X, names, _ = _drop_aliased_columns(X, names)
        _, _, resid, _, _ = _wls(X, y)
        rss = float(resid @ resid)
        p = np.linalg.matrix_rank(X)
        # Small-sample corrected AIC; the grid search itself costs a parameter
        # when tau > 0, which is charged below.
        k = p + (1 if tau > 0 else 0)
        aic = n * np.log(max(rss / n, 1e-12)) + 2 * k
        if n - k - 1 > 0:
            aic += (2 * k * (k + 1)) / (n - k - 1)
        trace.append((float(tau), float(aic)))
        if best is None or aic < best[1]:
            best = (float(tau), float(aic))
    if best is None:
        return None, trace
    return (best[0] if best[0] > 0 else None), trace


def thermal_lag_report(
    blocks: pd.DataFrame,
    raw_temp: np.ndarray,
    reference: str,
    spec: ModelSpec,
    use_night: bool = False,
    grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Residual variance versus lag, with and without the parameter penalty.

    Reported separately because the two can disagree: a lag often reduces the
    residuals appreciably while still losing on AICc when there are only a
    dozen blocks to pay for the extra parameter.
    """
    if grid is None:
        grid = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0]
    y = pd.to_numeric(blocks["focus_pos"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for tau in grid:
        temp = apply_thermal_lag(blocks["t_mid"], raw_temp, tau) if tau else raw_temp
        if np.nanstd(temp) < 1e-9:
            continue
        X, names, _ = build_design(
            blocks, reference, temp, float(np.nanmean(temp)), spec, use_night
        )
        _, _, resid, _, _ = _wls(X, y)
        rows.append(
            {
                "tau_minutes": float(tau),
                "rss": float(resid @ resid),
                "resid_rms": float(np.sqrt(np.mean(resid**2))),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"trace": df}
    best_rss = float(df.loc[df["rss"].idxmin(), "tau_minutes"])
    penalised, trace = _profile_thermal_lag(
        blocks, raw_temp, reference, spec, use_night, grid
    )
    tr = pd.DataFrame(trace, columns=["tau_minutes", "aicc"])
    df = df.merge(tr, on="tau_minutes", how="left")
    return {
        "trace": df,
        "best_tau_by_rss": best_rss,
        "best_tau_by_aicc": penalised,
        "rss_at_zero": float(df.loc[df["tau_minutes"] == 0.0, "rss"].iloc[0])
        if (df["tau_minutes"] == 0.0).any() else float("nan"),
        "rss_best": float(df["rss"].min()),
    }


def fit_offsets(blocks: pd.DataFrame, spec: ModelSpec | None = None) -> FitResult:
    """Fit Model A and return offsets, temperature coefficient and diagnostics."""
    spec = spec or ModelSpec()
    blocks = blocks.copy().reset_index(drop=True)
    notes: list[str] = []

    if blocks.empty:
        raise ValueError("no focus blocks to fit")

    temp_col, temp_reasons = (
        (spec.temp_column, {}) if (spec.temp_column and spec.temp_column != "auto")
        else assess_temperature_sources(blocks)
    )
    if temp_col is not None and temp_col not in blocks:
        temp_col = None
    have_temp = temp_col is not None
    if not have_temp:
        # Offsets are differences between filters and remain perfectly well
        # posed without temperature; only the coefficient is lost. Fitting one
        # anyway against a dead or regulated sensor produces a confident number
        # from noise, which is worse than reporting none.
        notes.append(
            "No usable temperature source, so no temperature coefficient is "
            "estimated and the offsets are fitted without one. "
            + "; ".join(f"{k}: {v}" for k, v in temp_reasons.items())
            + "."
        )
        temp_col = "ambient_temp" if "ambient_temp" in blocks else "ccd_temp"

    src_col = temp_col
    if spec.temp_at_block_start and f"{temp_col}_first" in blocks:
        src_col = f"{temp_col}_first"
    raw_temp = pd.to_numeric(blocks[src_col], errors="coerce").to_numpy(dtype=float)
    if not have_temp:
        raw_temp = np.zeros(len(blocks), dtype=float)
    if not np.isfinite(raw_temp).all():
        fill = pd.to_numeric(blocks[temp_col], errors="coerce").to_numpy(dtype=float)
        raw_temp = np.where(np.isfinite(raw_temp), raw_temp, fill)

    y = pd.to_numeric(blocks["focus_pos"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(y).any() and float(np.nanmax(y) - np.nanmin(y)) == 0.0:
        notes.append(
            "The focuser position is identical in every block, so every offset "
            "is exactly zero by construction. Either autofocus never moved the "
            "focuser, or the position recorded in the headers is not being "
            "updated."
        )
    good = np.isfinite(y) & np.isfinite(raw_temp)
    if good.sum() < blocks.shape[0]:
        notes.append(f"Dropped {int((~good).sum())} block(s) lacking position or temperature.")
        blocks = blocks.loc[good].reset_index(drop=True)
        y, raw_temp = y[good], raw_temp[good]

    n_nights = int(blocks["session"].nunique())
    drift = DriftTest()
    mode = str(spec.night_effects).lower()
    if mode == "auto":
        drift = zero_point_drift_test(blocks, spec)
        use_night = drift.verdict == "jumps"
        if drift.tested and not use_night:
            notes.append(
                "Focuser positions are treated as comparable across nights. The "
                "zero-point test found "
                + (
                    f"a smooth drift of {drift.drift_per_day:+.3f} steps/day "
                    f"({drift.drift_total_steps:+.1f} steps over "
                    f"{drift.span_days:.0f} days) but no night-to-night jumps "
                    f"(detrended between-night scatter {drift.between_sd_detrended:.2f} "
                    f"steps against {drift.within_sd:.2f} within a night)."
                    if drift.verdict == "drift" else
                    f"neither a trend nor jumps (between-night scatter "
                    f"{drift.between_sd_detrended:.2f} steps against "
                    f"{drift.within_sd:.2f} within a night)."
                )
            )
        elif not drift.tested and n_nights > 1:
            notes.append(
                f"Focuser positions are assumed comparable across nights: {drift.reason}. "
                "This assumes the imaging train was not disturbed between sessions."
            )
    else:
        use_night = mode in ("true", "yes", "on", "1")
    use_night = bool(use_night) and n_nights > 1
    if use_night:
        notes.append(
            f"Per-night intercepts included ({n_nights} nights): offsets are "
            "identified from within-night filter comparisons, which absorbs "
            "focuser zero-point jumps between nights - at the cost of any offset "
            "that relies on comparing nights."
        )

    drift_mode = spec.drift_term
    if isinstance(drift_mode, str) and drift_mode.lower() == "auto":
        use_drift = bool(
            (not use_night) and drift.tested and drift.verdict == "drift"
        )
    else:
        use_drift = bool(drift_mode) and n_nights > 1
    if use_drift:
        notes.append(
            "A linear drift with date is included, which absorbs slow settling "
            "with one parameter instead of a free level per night, so every "
            "filter offset stays measurable."
        )

    reference = pick_reference(blocks, spec)

    # thermal lag - profiled on whichever variation the coefficient will use
    mode_tf_pre = str(spec.temp_from).lower()
    # The within-night regressor is temperature minus its own night's mean. If
    # temperature is logged once per night, or is coarsely quantised, that is
    # identically zero (or numerical dust) and a slope fitted on it is
    # meaningless - one archive shape produced a coefficient wrong by five
    # orders of magnitude because a relaxation artefact of ~1e-6 C survived as
    # a "regressor". So the split is only used when there is real variation
    # inside nights to support it.
    within_span = _within_night_span(blocks, raw_temp)
    within_ok = bool(np.isfinite(within_span) and within_span >= spec.min_within_night_range_c)
    within_planned = (
        n_nights > 1
        and not use_night
        and mode_tf_pre in ("auto", "within_night")
        and within_ok
    )
    if (
        n_nights > 1 and not use_night
        and mode_tf_pre in ("auto", "within_night") and not within_ok
    ):
        notes.append(
            f"Temperature varies by only {within_span:.2f} C within nights, too "
            "little to estimate a separate within-night response, so a single "
            "temperature slope is fitted across all the data. It therefore "
            "mixes within-night and between-night variation."
        )
    lag: float | None = None
    lag_trace: list[tuple[float, float]] = []
    within_lag_table = pd.DataFrame()
    if not have_temp:
        lag = None
    elif isinstance(spec.thermal_lag_minutes, str) and spec.thermal_lag_minutes == "fit":
        if within_planned:
            lag, within_lag_table = profile_thermal_lag_within_night(
                blocks, raw_temp, y
            )
            if not within_lag_table.empty:
                lag_trace = list(
                    zip(within_lag_table["tau_minutes"], within_lag_table["aicc"])
                )
        else:
            lag, lag_trace = _profile_thermal_lag(
                blocks, raw_temp, reference, spec, use_night
            )
    elif isinstance(spec.thermal_lag_minutes, (int, float)) and spec.thermal_lag_minutes:
        lag = float(spec.thermal_lag_minutes)
    temp = apply_thermal_lag(blocks["t_mid"], raw_temp, lag) if lag else raw_temp

    temp_ref = float(np.nanmean(temp))

    # --- identifiability, decided before anything is fitted ----------------
    ident = identifiability(blocks, use_night, reference)
    not_identified = list(ident.get("not_identified", []))
    if not_identified:
        notes.append(
            "These filters share no night with the reference filter, so with "
            "per-night intercepts their offsets are absorbed entirely by those "
            "nights' zero points and cannot be estimated: "
            + ", ".join(not_identified)
            + ". Their frames still inform the temperature coefficient."
        )
    # With per-night intercepts the night-mean temperature is already absorbed
    # by them, so the split is only meaningful (and only identifiable) without.
    split_temp = within_planned
    X, names, others = build_design(
        blocks, reference, temp, temp_ref, spec, use_night,
        exclude_filters=not_identified, use_drift=use_drift,
        split_temp=split_temp and have_temp, include_temp=have_temp,
    )
    # Safety net for any residual exact collinearity (e.g. a night on which
    # only one filter was ever used together with an altitude term).
    X, names, aliased = _drop_aliased_columns(X, names)
    if aliased:
        notes.append(
            "Dropped exactly-collinear design columns rather than letting the "
            "pseudo-inverse invent a minimum-norm answer for them: "
            + ", ".join(aliased)
            + "."
        )
        extra = {nm[len("filter["):-1] for nm in aliased if nm.startswith("filter[")}
        not_identified = sorted(set(not_identified) | extra)
        others = [f for f in others if f not in not_identified]

    n, p = X.shape
    if n <= p:
        raise ValueError(
            f"only {n} focus blocks for {p} parameters - cannot identify the model. "
            "Collect more blocks, or disable night/altitude terms."
        )

    method = spec.method
    if method == "auto":
        method = "robust" if n >= 8 else "ols"

    if method == "robust":
        beta, cov, resid, s2, dof, w, scale = _huber(X, y)
    else:
        beta, cov, resid, s2, dof = _wls(X, y)
        w = np.ones(n)
        scale = float(np.sqrt(s2))

    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    idx = {nm: i for i, nm in enumerate(names)}

    # Only bootstrap when the result will actually be preferred. Below the
    # block count where it is trusted it was being computed and then thrown
    # away, costing seconds per run on every archive measured.
    boot: dict[str, np.ndarray] = {}
    if spec.bootstrap and n >= 25:
        boot = _bootstrap(
            blocks, y, temp, temp_ref, reference, spec, use_night, method, names,
            exclude=not_identified, use_drift=use_drift, split_temp=split_temp,
        )

    alpha = 1.0 - spec.confidence
    res = FitResult(
        reference=reference,
        filters=sorted(blocks["filter"].dropna().unique().tolist()),
        temp_coeff=float(beta[idx["temp"]]) if "temp" in idx else float("nan"),
        temp_coeff_se=float(se[idx["temp"]]) if "temp" in idx else float("nan"),
        intercept=float(beta[idx["intercept"]]),
        temp_ref=temp_ref,
        temp_column=src_col if have_temp else "",
        method=method,
        n_blocks=n,
        n_params=int(np.linalg.matrix_rank(X)),
        resid=resid,
        resid_rms=float(np.sqrt(np.mean(resid**2))),
        resid_scale=float(scale),
        weights=w,
        design_columns=list(names),
        blocks=blocks,
        thermal_lag_minutes=lag,
        notes=notes,
    )
    ss_tot = float(((y - y.mean()) ** 2).sum())
    res.r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")

    # With a handful of blocks a percentile bootstrap is itself unstable (it
    # can resample away an entire filter), so the Student-t interval is the
    # headline number and the bootstrap is carried alongside as a check.
    # A percentile bootstrap is only preferred when it is actually informative.
    # When every identified offset rests on a single night, resampling nights
    # reproduces that night verbatim and the interval collapses to far below
    # the analytic standard error - which looks precise and is not.
    res.confidence = float(spec.confidence)
    res.ci_method = "bootstrap" if n >= 25 else "t"
    if res.ci_method == "bootstrap":
        ref_key = next(
            (f"filter[{f}]" for f in others if f"filter[{f}]" in boot), None
        )
        if ref_key is not None:
            spread = float(np.nanstd(np.asarray(boot[ref_key], dtype=float)))
            analytic = float(se[idx[ref_key]])
            if not np.isfinite(spread) or spread < 0.5 * analytic:
                res.ci_method = "t"
                notes.append(
                    "Bootstrap intervals were discarded as degenerate (resampled "
                    "spread far below the analytic standard error, which happens "
                    "when the offsets rest on very few nights); Student-t "
                    "intervals are reported instead."
                )
        elif not boot:
            res.ci_method = "t"
    res.offsets[reference] = 0.0
    res.offsets_se[reference] = 0.0
    res.offsets_ci[reference] = (0.0, 0.0)
    res.offsets_ci_boot[reference] = (0.0, 0.0)
    for f in others:
        key = f"filter[{f}]"
        res.offsets[f] = float(beta[idx[key]])
        res.offsets_se[f] = float(se[idx[key]])
        t_ci = _ci_t(beta[idx[key]], se[idx[key]], dof, alpha)
        b_ci = _ci_boot(boot.get(key), alpha)
        res.offsets_ci_boot[f] = b_ci
        res.offsets_ci[f] = b_ci if (res.ci_method == "bootstrap" and
                                     np.isfinite(b_ci[0])) else t_ci
    t_ci = _ci_t(res.temp_coeff, res.temp_coeff_se, dof, alpha)
    b_ci = _ci_boot(boot.get("temp"), alpha)
    res.temp_coeff_ci_boot = b_ci
    res.temp_coeff_ci = b_ci if (res.ci_method == "bootstrap" and
                                 np.isfinite(b_ci[0])) else t_ci
    res.temp_from = "within_night" if split_temp else "pooled"
    if "temp_between_nights" in idx:
        # The between-night response, reported for comparison. A large gap from
        # the within-night figure means something that differs between nights is
        # tracking temperature without being caused by it.
        res.temp_coeff_between = float(beta[idx["temp_between_nights"]])
        res.temp_coeff_between_se = float(se[idx["temp_between_nights"]])
        try:
            Xp, np_names, _ = build_design(
                blocks, reference, temp, temp_ref, spec, use_night,
                exclude_filters=not_identified, use_drift=use_drift,
                split_temp=False,
            )
            Xp, np_names, _ = _drop_aliased_columns(Xp, np_names)
            bp, cp, *_ = _wls(Xp, y)
            j = np_names.index("temp")
            res.temp_coeff_pooled = float(bp[j])
            res.temp_coeff_pooled_se = float(np.sqrt(max(cp[j, j], 0.0)))
        except Exception:
            pass

    res.temp_values = np.asarray(temp, dtype=float)
    res.drift_test = drift
    if use_drift and "drift_per_day" in idx:
        res.drift_per_day = float(beta[idx["drift_per_day"]])
        res.drift_per_day_se = float(se[idx["drift_per_day"]])
    res.not_identified = not_identified
    res.components = ident.get("components", [])
    if use_night:
        sessions = sorted(blocks["session"].unique().tolist())
        res.night_offsets = {int(sessions[0]): 0.0}
        for sess in sessions[1:]:
            key = f"night[{sess}]"
            res.night_offsets[int(sess)] = (
                float(beta[idx[key]]) if key in idx else 0.0
            )
    for f in not_identified:
        res.offsets[f] = float("nan")
        res.offsets_se[f] = float("nan")
        res.offsets_ci[f] = (float("nan"), float("nan"))
        res.offsets_ci_boot[f] = (float("nan"), float("nan"))
    if not_identified:
        # A number is still useful to the reader, provided it is labelled with
        # the assumption it rests on: no focuser zero-point shift between
        # nights. Fitted without night intercepts, which is what makes the
        # comparison possible at all.
        try:
            pooled_spec = ModelSpec(
                **{**spec.__dict__, "bootstrap": 0, "night_effects": "off",
                   "reference_filter": reference}
            )
            pooled = fit_offsets(blocks, pooled_spec)
            res.offsets_pooled = dict(pooled.offsets)
            res.offsets_pooled_se = dict(pooled.offsets_se)
        except Exception:
            pass

    res.diagnostics = design_diagnostics(
        blocks, X, names, temp, reference, by_night=use_night
    )
    res.diagnostics["identifiability"] = ident
    res.diagnostics["aliased_columns"] = aliased
    res.diagnostics["durbin_watson"] = durbin_watson(resid)
    res.diagnostics["cooks_distance"] = cooks_distance(X, resid, s2).tolist()
    res.diagnostics["thermal_lag_trace"] = lag_trace
    if not within_lag_table.empty:
        res.diagnostics["within_night_lag_table"] = within_lag_table
    res.diagnostics["bootstrap_n"] = int(spec.bootstrap)
    res.diagnostics["sigma"] = float(np.sqrt(s2))
    res.diagnostics["dof"] = int(dof)
    res.diagnostics["huber_weights"] = w.tolist()
    res.diagnostics["temp_source_column"] = src_col if have_temp else None
    res.diagnostics["temperature_sources"] = temp_reasons
    res.diagnostics["has_temperature"] = bool(have_temp)
    res.diagnostics["night_effects"] = bool(use_night)

    if spec.per_filter_slopes and have_temp:
        res.per_filter_slopes = _test_per_filter_slopes(
            blocks, y, temp, temp_ref, reference, spec, use_night, resid, X,
            exclude=not_identified,
        )
    return res


def _ci_t(estimate: float, se: float, dof: int, alpha: float) -> tuple[float, float]:
    """Student-t interval; the default, because block counts are usually small."""
    from scipy import stats

    if not np.isfinite(se) or se <= 0 or dof < 1:
        return (float("nan"), float("nan"))
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return (float(estimate - crit * se), float(estimate + crit * se))


def _ci_boot(samples, alpha: float) -> tuple[float, float]:
    """Percentile bootstrap interval, or NaNs when there are too few draws."""
    if samples is None:
        return (float("nan"), float("nan"))
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 100:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(arr, 100 * alpha / 2)),
        float(np.percentile(arr, 100 * (1 - alpha / 2))),
    )


def _bootstrap(
    blocks, y, temp, temp_ref, reference, spec, use_night, method, names,
    exclude: Sequence[str] = (), use_drift: bool = False,
    split_temp: bool = False,
) -> dict[str, np.ndarray]:
    """Resample blocks (clustered by night) and refit."""
    n_boot = int(spec.bootstrap)
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(spec.random_seed)
    out: dict[str, list[float]] = {nm: [] for nm in names}
    n = len(y)
    sessions = blocks["session"].to_numpy()
    uniq_sessions = np.unique(sessions)
    cluster = len(uniq_sessions) >= 4

    for _ in range(n_boot):
        if cluster:
            pick = rng.choice(uniq_sessions, size=len(uniq_sessions), replace=True)
            idx = np.concatenate([np.flatnonzero(sessions == s) for s in pick])
        else:
            idx = rng.integers(0, n, size=n)
        b = blocks.iloc[idx].reset_index(drop=True)
        if b["filter"].nunique() < blocks["filter"].nunique():
            continue
        # A resample that loses a night-filter link changes which offsets are
        # identifiable; including such draws is what produced absurdly wide
        # intervals (a 0.7-step standard error beside a 200-step interval).
        if use_night and filter_components(b) != filter_components(blocks):
            continue
        try:
            Xb, nb, _ = build_design(
                b, reference, temp[idx], temp_ref, spec, use_night,
                exclude_filters=exclude, use_drift=use_drift,
                split_temp=split_temp,
            )
            if Xb.shape[0] <= np.linalg.matrix_rank(Xb):
                continue
            if method == "robust":
                beta, *_ = _huber(Xb, y[idx])[:1]
            else:
                beta, *_ = _wls(Xb, y[idx])[:1]
        except (np.linalg.LinAlgError, ValueError):
            continue
        pos = {nm: i for i, nm in enumerate(nb)}
        for nm in names:
            if nm in pos:
                out[nm].append(float(beta[pos[nm]]))
    return {k: np.asarray(v) for k, v in out.items() if v}


def _test_per_filter_slopes(
    blocks, y, temp, temp_ref, reference, spec, use_night, _resid_unused, X0,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """Does each filter need its own temperature slope?

    Physically the focus shift with temperature is dominated by the tube and
    mirror, which are common to all filters, so a shared slope is expected.  A
    significant improvement here usually means something else is wrong (a
    confounded design, or autofocus behaving differently per filter).
    """
    filters = sorted(blocks["filter"].dropna().unique().tolist())
    others = [f for f in filters if f != reference and f not in set(exclude or ())]
    if len(others) < 1:
        return {"tested": False, "reason": "no comparison filter with an offset"}

    names = ["<base>"]
    extra = []
    for f in others:
        m = (blocks["filter"] == f).to_numpy(dtype=float)
        extra.append(m * (np.asarray(temp, dtype=float) - temp_ref))
        names.append(f"temp:filter[{f}]")
    X1 = np.column_stack([X0] + extra)
    n = len(y)
    if n <= np.linalg.matrix_rank(X1) + 1:
        return {
            "tested": False,
            "reason": f"need more blocks ({n}) to fit {X1.shape[1]} parameters",
        }

    # Both sides of the F-test must come from the same estimator, so the
    # restricted model is refitted by ordinary least squares here rather than
    # reusing the caller's (possibly Huber-weighted) residuals.
    _, _, r0, _, _ = _wls(X0, y)
    beta1, cov1, r1, _, _ = _wls(X1, y)
    rss0 = float(r0 @ r0)
    rss1 = float(r1 @ r1)
    p0 = np.linalg.matrix_rank(X0)
    p1 = np.linalg.matrix_rank(X1)
    df_num = max(p1 - p0, 1)
    df_den = max(n - p1, 1)
    F = ((rss0 - rss1) / df_num) / max(rss1 / df_den, 1e-12)
    from scipy import stats

    pval = float(1.0 - stats.f.cdf(F, df_num, df_den))
    slopes = {}
    for j, f in enumerate(others):
        i = X0.shape[1] + j
        slopes[f] = {
            "delta_k": float(beta1[i]),
            "se": float(np.sqrt(max(cov1[i, i], 0.0))),
        }
    return {
        "tested": True,
        "F": float(F),
        "df": (int(df_num), int(df_den)),
        "p_value": pval,
        "needed": bool(pval < 0.05),
        "delta_slopes": slopes,
    }


# --------------------------------------------------------------------------- #
# Model B - HFD excess
# --------------------------------------------------------------------------- #
def _natural_spline_basis(x: np.ndarray, n_knots: int) -> np.ndarray:
    """Truncated-power natural cubic spline basis (without intercept)."""
    x = np.asarray(x, dtype=float)
    if n_knots < 3:
        return x[:, None] if n_knots <= 1 else np.column_stack([x, x**2])
    qs = np.linspace(0, 100, n_knots)
    knots = np.unique(np.percentile(x[np.isfinite(x)], qs))
    if knots.size < 3:
        return x[:, None]
    k0, kK = knots[0], knots[-1]
    dK = max(kK - k0, 1e-9)

    def d(t):
        return (
            np.maximum(x - t, 0.0) ** 3 - np.maximum(x - kK, 0.0) ** 3
        ) / max(kK - t, 1e-9)

    cols = [x]
    for t in knots[1:-1]:
        cols.append(d(t) - d(k0) * 1.0)
    basis = np.column_stack(cols) / dK
    return basis


@dataclass
class HfdExcessResult:
    """Per-filter star-size excess after removing observing conditions."""

    metric: str
    backend: str
    reference: str
    excess_log: dict[str, float] = field(default_factory=dict)
    excess_log_se: dict[str, float] = field(default_factory=dict)
    excess_frac: dict[str, float] = field(default_factory=dict)
    excess_px: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    airmass_exponent: float = float("nan")
    airmass_exponent_se: float = float("nan")
    n_frames: int = 0
    spline_df: int = 0
    sigma: float = float("nan")
    r2: float = float("nan")
    baseline_hfd: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = []
        for f in sorted(self.excess_log):
            rows.append(
                {
                    "filter": f,
                    "excess_frac": self.excess_frac.get(f, np.nan),
                    "excess_px": self.excess_px.get(f, np.nan),
                    "se_frac": self.excess_log_se.get(f, np.nan),
                    "p_value": self.p_values.get(f, np.nan),
                    "is_reference": f == self.reference,
                }
            )
        return pd.DataFrame(rows)


def fit_hfd_excess(
    frames: pd.DataFrame,
    metric: str = "hfd_median_bright",
    backend: str = "internal",
    reference: str | None = None,
    spline_df: int | None = None,
    min_frames_per_filter: int = 3,
) -> HfdExcessResult:
    """Test whether any filter's stars are larger than conditions can explain.

    Fitted in logarithms, because seeing and airmass act multiplicatively on
    star size while an additive read would confound them with the overall level:

    ``log HFD = c + gamma*log(airmass) + spline(time) + eta_filter``

    The spline soaks up the night's seeing evolution.  ``eta_filter`` is then
    the fractional star-size excess attributable to the filter itself.

    A positive ``eta`` is *consistent with* that filter's autofocus landing off
    best focus, but is not proof of it: an intrinsically softer passband looks
    identical in near-focus data.  See the module docstring.
    """
    df = frames.copy()
    if "backend" in df.columns:
        df = df[df["backend"].astype(str) == backend]
    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} not present in the profile table")

    y_raw = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y_raw) & (y_raw > 0)
    if "qc_pass" in df.columns:
        ok &= df["qc_pass"].fillna(True).to_numpy(dtype=bool)
    df = df.loc[ok].reset_index(drop=True)
    y = np.log(pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float))

    counts = df["filter"].value_counts()
    keep = [f for f in counts.index if counts[f] >= min_frames_per_filter]
    notes: list[str] = []
    dropped = [f for f in counts.index if f not in keep]
    if dropped:
        notes.append(
            f"Excluded from the star-size test for want of frames: {', '.join(map(str, dropped))}."
        )
    df = df[df["filter"].isin(keep)].reset_index(drop=True)
    y = np.log(pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float))
    n = len(df)
    if n < 8 or df["filter"].nunique() < 2:
        return HfdExcessResult(
            metric=metric, backend=backend, reference=reference or "",
            n_frames=n,
            notes=notes + ["Too few usable frames to separate filter from conditions."],
        )

    filters = sorted(df["filter"].dropna().unique().tolist())
    ref = reference if reference in filters else filters[0]
    others = [f for f in filters if f != ref]

    t = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")
    elapsed = (t - t.min()).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    airmass = pd.to_numeric(df.get("airmass"), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(airmass).all():
        alt = pd.to_numeric(df.get("altitude"), errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fallback = 1.0 / np.cos(np.radians(90.0 - alt))
        airmass = np.where(np.isfinite(airmass), airmass, fallback)
    use_airmass = np.isfinite(airmass).all() and np.nanstd(np.log(airmass)) > 1e-3

    if spline_df is None:
        # Keep the smooth well short of the number of blocks so that it models
        # the night's seeing trend without absorbing the filter pattern.
        n_blocks = int(df["block"].nunique()) if "block" in df else n
        spline_df = int(np.clip(n_blocks // 3, 1, 5))

    cols = [np.ones(n)]
    names = ["intercept"]
    if use_airmass:
        cols.append(np.log(airmass))
        names.append("log_airmass")
    basis = _natural_spline_basis(elapsed, spline_df + 1)
    for j in range(basis.shape[1]):
        cols.append(basis[:, j])
        names.append(f"spline{j}")
    for f in others:
        cols.append((df["filter"] == f).to_numpy(dtype=float))
        names.append(f"filter[{f}]")
    X = np.column_stack(cols)
    if n <= X.shape[1] + 1:
        return HfdExcessResult(
            metric=metric, backend=backend, reference=ref, n_frames=n,
            notes=notes + ["Not enough frames for the conditions model."],
        )

    beta, cov, resid, s2, dof = _wls(X, y)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    idx = {nm: i for i, nm in enumerate(names)}
    from scipy import stats

    med = float(np.nanmedian(pd.to_numeric(df[metric], errors="coerce")))
    res = HfdExcessResult(
        metric=metric,
        backend=backend,
        reference=ref,
        n_frames=n,
        spline_df=int(basis.shape[1]),
        sigma=float(np.sqrt(s2)),
        baseline_hfd=med,
        notes=notes,
    )
    ss_tot = float(((y - y.mean()) ** 2).sum())
    res.r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")
    if use_airmass:
        res.airmass_exponent = float(beta[idx["log_airmass"]])
        res.airmass_exponent_se = float(se[idx["log_airmass"]])

    res.excess_log[ref] = 0.0
    res.excess_log_se[ref] = 0.0
    res.excess_frac[ref] = 0.0
    res.excess_px[ref] = 0.0
    res.p_values[ref] = float("nan")
    for f in others:
        key = f"filter[{f}]"
        b, s = float(beta[idx[key]]), float(se[idx[key]])
        res.excess_log[f] = b
        res.excess_log_se[f] = s
        res.excess_frac[f] = float(np.expm1(b))
        res.excess_px[f] = float(med * np.expm1(b))
        tstat = b / s if s > 0 else np.nan
        res.p_values[f] = (
            float(2 * (1 - stats.t.cdf(abs(tstat), dof))) if np.isfinite(tstat) else np.nan
        )
    return res


# --------------------------------------------------------------------------- #
# sensitivity analysis
# --------------------------------------------------------------------------- #
def sensitivity_analysis(
    blocks: pd.DataFrame, spec: ModelSpec | None = None
) -> pd.DataFrame:
    """Refit Model A under several defensible specifications.

    A number that moves when the specification changes is a number the data do
    not really pin down.  In practice the filter offsets are stable and the
    temperature coefficient is not, and the report should say so rather than
    quote one fit to three decimals.

    The ``+ elapsed time`` row is included deliberately as a *warning*: on a
    single night elapsed time and temperature are nearly collinear, so that fit
    will often look best while making the temperature coefficient meaningless
    for any other night.
    """
    spec = spec or ModelSpec()
    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline", {}),
        ("OLS (no robustness)", {"method": "ols"}),
        ("robust (Huber)", {"method": "robust"}),
        ("no thermal lag", {"thermal_lag_minutes": None}),
        ("thermal lag 30 min", {"thermal_lag_minutes": 30.0}),
        ("thermal lag 60 min", {"thermal_lag_minutes": 60.0}),
        ("block-mean temperature", {"temp_at_block_start": False}),
        ("+ altitude term", {"altitude_term": True}),
    ]
    rows: list[dict[str, Any]] = []
    for label, changes in variants:
        kwargs = {**spec.__dict__, **changes, "bootstrap": 0}
        try:
            fit = fit_offsets(blocks, ModelSpec(**kwargs))
        except Exception as exc:
            rows.append({"variant": label, "error": str(exc)})
            continue
        row: dict[str, Any] = {
            "variant": label,
            "reference": fit.reference,
            "k_steps_per_C": fit.temp_coeff,
            "k_se": fit.temp_coeff_se,
            "resid_rms": fit.resid_rms,
            "n_blocks": fit.n_blocks,
            "thermal_lag_min": fit.thermal_lag_minutes,
        }
        for f, v in fit.offsets.items():
            row[f"offset_{f}"] = v
        rows.append(row)

    # The confounded-on-purpose variant, reported separately.
    try:
        extra = _fit_with_elapsed(blocks, spec)
        rows.append(extra)
    except Exception:
        pass
    df = pd.DataFrame(rows)
    # Several variants collapse onto the baseline (auto already chose robust,
    # or no lag, or the altitude gate was not met). Counting them as agreement
    # would make the stability check congratulate itself, so they are marked
    # and excluded from the spread.
    if not df.empty:
        key = [c for c in df.columns if c.startswith("offset_")] + ["k_steps_per_C"]
        key = [c for c in key if c in df.columns]
        if key:
            df["duplicate_of_baseline"] = df[key].round(6).duplicated(keep="first")
            df.loc[df.index[0], "duplicate_of_baseline"] = False
    return df


def _fit_with_elapsed(blocks: pd.DataFrame, spec: ModelSpec) -> dict[str, Any]:
    """Model A plus a linear drift in elapsed time (diagnostic only)."""
    spec2 = ModelSpec(**{**spec.__dict__, "bootstrap": 0, "thermal_lag_minutes": None})
    b = blocks.reset_index(drop=True)
    temp_col = choose_temp_column(b, spec2)
    src = f"{temp_col}_first" if spec2.temp_at_block_start and f"{temp_col}_first" in b else temp_col
    temp = pd.to_numeric(b[src], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b["focus_pos"], errors="coerce").to_numpy(dtype=float)
    ref = pick_reference(b, spec2)
    X, names, others = build_design(b, ref, temp, float(np.nanmean(temp)), spec2, False)
    # Within-night elapsed time: the global figure spans the whole campaign,
    # where the gaps between nights dwarf the imaging and the "drift per hour"
    # would describe the weather rather than the telescope.
    col = (
        "elapsed_hours_in_session"
        if "elapsed_hours_in_session" in b.columns else "elapsed_hours"
    )
    el = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
    X = np.column_stack([X, el - np.nanmean(el)])
    names = names + ["elapsed_hours"]
    beta, cov, resid, s2, dof = _wls(X, y)
    idx = {nm: i for i, nm in enumerate(names)}
    row: dict[str, Any] = {
        "variant": "+ elapsed time (CONFOUNDED - diagnostic only)",
        "reference": ref,
        "k_steps_per_C": float(beta[idx["temp"]]),
        "k_se": float(np.sqrt(max(cov[idx["temp"], idx["temp"]], 0.0))),
        "resid_rms": float(np.sqrt(np.mean(resid**2))),
        "n_blocks": len(y),
        "drift_steps_per_hour": float(beta[idx["elapsed_hours"]]),
        "corr_temp_elapsed": float(np.corrcoef(temp, el)[0, 1]),
    }
    row[f"offset_{ref}"] = 0.0
    for f in others:
        row[f"offset_{f}"] = float(beta[idx[f"filter[{f}]"]])
    return row


# --------------------------------------------------------------------------- #
# within-block focus decay
# --------------------------------------------------------------------------- #
def within_block_decay(
    frames: pd.DataFrame,
    metric: str = "hfd_median_bright",
    temp_column: str = "ambient_temp",
    temp_coeff: float | None = None,
    slope_px_per_step: float | None = None,
) -> dict[str, Any]:
    """Test whether star size grows across the frames of a block.

    The usual working assumption is that the first sub after an autofocus run is
    the sharpest, because the focuser then sits still while the temperature keeps
    drifting.  That is true in principle, but the size of the effect is what
    matters, and it is easy to overestimate: the V-curve is *quadratic* at its
    minimum, so starting from perfect focus the first steps of drift cost almost
    nothing.

    Estimated with block fixed effects (each block demeaned), so the night's
    seeing evolution cannot leak into the slope.  When a temperature coefficient
    and a V-curve slope are supplied, the *predicted* decay is returned
    alongside, which is what turns "not significant" into the much more useful
    "not significant, and here is why it could not have been".
    """
    out: dict[str, Any] = {"metric": metric, "tested": False}
    if frames.empty or metric not in frames.columns:
        out["reason"] = f"metric {metric!r} unavailable"
        return out
    df = frames.copy()
    if "qc_pass" in df.columns:
        df = df[df["qc_pass"].fillna(True).to_numpy(dtype=bool)]
    df["_y"] = pd.to_numeric(df[metric], errors="coerce")
    df["_i"] = pd.to_numeric(df["index_in_block"], errors="coerce")
    df = df.dropna(subset=["_y", "_i", "block"])
    sizes = df.groupby("block")["_y"].transform("size")
    df = df[sizes >= 2]
    if len(df) < 6 or df["_i"].nunique() < 2:
        out["reason"] = "too few multi-frame blocks after quality control"
        return out

    y = df["_y"].to_numpy(dtype=float)
    x = df["_i"].to_numpy(dtype=float)
    blk = df["block"].to_numpy()
    yd = y - pd.Series(y).groupby(blk).transform("mean").to_numpy()
    xd = x - pd.Series(x).groupby(blk).transform("mean").to_numpy()
    denom = float(xd @ xd)
    if denom <= 0:
        out["reason"] = "no within-block variation in frame index"
        return out
    slope = float((xd @ yd) / denom)
    resid = yd - slope * xd
    n_blocks = int(df["block"].nunique())
    dof = max(len(y) - n_blocks - 1, 1)
    s2 = float(resid @ resid) / dof
    se = float(np.sqrt(s2 / denom))
    from scipy import stats

    tstat = slope / se if se > 0 else np.nan
    out.update(
        {
            "tested": True,
            "slope_px_per_frame": slope,
            "se": se,
            "t": float(tstat) if np.isfinite(tstat) else None,
            "p_value": float(2 * (1 - stats.t.cdf(abs(tstat), dof)))
            if np.isfinite(tstat) else None,
            "n_frames": int(len(y)),
            "n_blocks": n_blocks,
            "dof": dof,
        }
    )
    out["significant"] = bool(
        out["p_value"] is not None and out["p_value"] < 0.05 and slope > 0
    )

    # Predicted decay, for comparison.
    try:
        t = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")
        temps = pd.to_numeric(df.get(temp_column), errors="coerce")
        per_block = []
        for _, g in df.assign(_t=t, _T=temps).groupby("block"):
            g = g.sort_values("_i")
            if len(g) < 2 or g["_T"].isna().all():
                continue
            dT = float(g["_T"].iloc[-1] - g["_T"].iloc[0]) / max(len(g) - 1, 1)
            per_block.append(dT)
        if per_block:
            out["mean_dT_per_frame"] = float(np.mean(per_block))
        if (
            temp_coeff is not None
            and slope_px_per_step is not None
            and out.get("mean_dT_per_frame") is not None
            and np.isfinite(slope_px_per_step)
        ):
            h0 = float(np.nanmedian(y))
            dT = out["mean_dT_per_frame"]
            steps = -float(temp_coeff) * dT  # focus error accrued per frame
            blur = float(slope_px_per_step) * steps
            # Quadratic bottom: size change from perfect focus after one frame.
            out["predicted_defocus_steps_per_frame"] = steps
            out["predicted_px_per_frame"] = float(
                np.sqrt(h0**2 + blur**2) - h0
            )
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# identifiability
# --------------------------------------------------------------------------- #
def filter_components(blocks: pd.DataFrame) -> list[list[str]]:
    """Group filters into sets whose offsets are mutually identifiable.

    With per-night intercepts in the model, a filter's offset is only measurable
    *relative to another filter observed on the same night*.  Two filters that
    never share a night are connected only if a chain of other filters links
    them.  Formally this is the connected-component structure of the bipartite
    night-filter graph, and it is the same identification condition that governs
    any two-way fixed-effects model.

    Filters in different components cannot be compared at all once night
    effects are included: the difference between them is perfectly absorbed by
    the night intercepts.  Reporting a number for such a pair - as a naive
    least-squares solve happily will, via the pseudo-inverse - is meaningless.
    """
    if blocks.empty or "filter" not in blocks or "session" not in blocks:
        return []
    pairs = blocks[["session", "filter"]].dropna().drop_duplicates()
    filters = sorted(pairs["filter"].unique().tolist())
    parent = {f: f for f in filters}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, g in pairs.groupby("session"):
        fs = sorted(g["filter"].unique().tolist())
        for other in fs[1:]:
            union(fs[0], other)

    groups: dict[str, list[str]] = {}
    for f in filters:
        groups.setdefault(find(f), []).append(f)
    out = [sorted(v) for v in groups.values()]
    # Largest component first, so the reference is chosen from the best-linked set.
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


def identifiability(
    blocks: pd.DataFrame, use_night: bool, reference: str
) -> dict[str, Any]:
    """Work out which filter offsets the design can actually support."""
    comps = filter_components(blocks)
    info: dict[str, Any] = {
        "components": comps,
        "night_effects": bool(use_night),
        "reference": reference,
    }
    filters = sorted(blocks["filter"].dropna().unique().tolist())
    if not use_night or len(comps) <= 1:
        info["identified"] = filters
        info["not_identified"] = []
        info["reference_component"] = comps[0] if comps else filters
        return info
    info["n_informative_nights"] = int(
        blocks.groupby("session")["filter"].nunique().gt(1).sum()
    )
    ref_comp = next((c for c in comps if reference in c), comps[0])
    info["reference_component"] = ref_comp
    info["identified"] = [f for f in filters if f in ref_comp]
    info["not_identified"] = [f for f in filters if f not in ref_comp]
    # Nights that would tie an orphaned component to the reference component.
    info["singleton_filters"] = [c[0] for c in comps if len(c) == 1]
    return info


def _drop_aliased_columns(
    X: np.ndarray, names: Sequence[str], protect: Sequence[str] = ("intercept", "temp")
) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove columns that are exact linear combinations of earlier ones.

    Without this, ``lstsq``/``pinv`` silently return a minimum-norm solution for
    a rank-deficient design: every coefficient comes back finite and plausible
    looking, the standard errors look small, and the numbers mean nothing.  The
    columns are processed in order with a QR-style rank check so that protected
    regressors and earlier filters survive and the redundant ones are named.
    """
    names = list(names)
    if not np.isfinite(X).all():
        bad = [names[j] for j in range(X.shape[1]) if not np.isfinite(X[:, j]).all()]
        raise ValueError(
            "design matrix contains non-finite values in: " + ", ".join(bad)
        )
    keep: list[int] = []
    dropped: list[str] = []
    tol = 1e-8
    norms = np.linalg.norm(X, axis=0)
    biggest = float(norms.max()) if norms.size else 1.0
    protected = set(protect)
    for j, name in enumerate(names):
        # A column that is numerical dust next to the others is not a
        # regressor, however linearly independent it looks once each column has
        # been normalised to unit length.
        if name not in protected and biggest > 0 and norms[j] < 1e-10 * biggest:
            dropped.append(name)
            continue
        trial = X[:, keep + [j]]
        scale = np.linalg.norm(trial, axis=0)
        scale[scale == 0] = 1.0
        r = np.linalg.matrix_rank(trial / scale, tol=tol)
        if r == len(keep) + 1:
            keep.append(j)
        else:
            dropped.append(name)
    return X[:, keep], [names[i] for i in keep], dropped


# --------------------------------------------------------------------------- #
# focuser zero-point stability
# --------------------------------------------------------------------------- #
@dataclass
class DriftTest:
    """Whether the focuser's zero point can be trusted across nights."""

    tested: bool = False
    reason: str = ""
    verdict: str = "unknown"  # "stable" | "drift" | "jumps"
    n_nights: int = 0
    anchor_filters: list[str] = field(default_factory=list)
    night_means: dict[int, float] = field(default_factory=dict)
    night_days: dict[int, float] = field(default_factory=dict)
    drift_per_day: float = float("nan")
    drift_se: float = float("nan")
    drift_p: float = float("nan")
    drift_total_steps: float = float("nan")
    span_days: float = float("nan")
    between_sd: float = float("nan")
    between_sd_detrended: float = float("nan")
    within_sd: float = float("nan")
    n_nights_used_for_within: int = 0
    #: Scatter a night mean would show even with a perfectly stable zero point.
    expected_night_sd: float = float("nan")
    #: Estimated zero-point movement between nights, in steps (variance component).
    night_sd: float = float("nan")
    #: Whether a between-night temperature term was affordable in the test.
    used_between_night_temp: bool = False
    jump_ratio: float = float("nan")
    anova_F: float = float("nan")
    anova_p: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def zero_point_drift_test(
    blocks: pd.DataFrame,
    spec: ModelSpec | None = None,
    min_nights_per_anchor: int = 2,
    min_nights_for_between_temp: int = 4,
    jump_threshold: float = 0.5,
    jump_min_steps: float = 1.0,
    trend_alpha: float = 0.05,
) -> DriftTest:
    """Ask whether focuser positions are comparable between nights.

    If the imaging train is never disturbed, the focuser's absolute encoder
    makes positions directly comparable across nights, and the *between*-night
    information is exactly what lets a filter used alone on its own nights be
    tied to the rest.  Throwing it away with per-night intercepts is safe but
    expensive: it can leave a filter with no measurable offset at all.

    So the assumption is tested rather than assumed either way.  Residuals from
    a fit with no night terms are averaged per night and decomposed into:

    * a **smooth trend** with date - slow mechanical settling, or a seasonal
      effect the temperature model misses.  One parameter absorbs it and no
      identifiability is lost.
    * **random jumps** between nights - what re-seating a camera, changing
      spacers or a focuser losing its home position looks like.  Nothing short
      of a free level per night handles this, and that is what costs offsets.

    The jump component is measured *after* removing the trend and compared with
    the within-night scatter, which is the autofocus repeatability floor.  If
    the jumps are no larger than that floor there is nothing for per-night
    intercepts to absorb.

    Nights contributing an offset estimate for a filter seen on only that night
    would confound the filter's offset with the night's level, so the per-night
    means are taken from *anchor* filters observed on several nights.
    """
    spec = spec or ModelSpec()
    out = DriftTest()
    if blocks.empty or "session" not in blocks.columns:
        out.reason = "no session information"
        return out

    nights = sorted(blocks["session"].unique().tolist())
    out.n_nights = len(nights)
    if out.n_nights < 3:
        out.reason = (
            f"only {out.n_nights} night(s); at least 3 are needed to separate a "
            "trend from night-to-night jumps"
        )
        return out

    # Anchor filters: observed on enough nights that their own offset cannot be
    # mistaken for a night level.
    per_filter_nights = blocks.groupby("filter")["session"].nunique()
    anchors = sorted(
        per_filter_nights[per_filter_nights >= min_nights_per_anchor].index.tolist()
    )
    if not anchors:
        out.reason = (
            "no filter appears on at least "
            f"{min_nights_per_anchor} nights, so a night's level cannot be told "
            "apart from that filter's offset"
        )
        return out
    out.anchor_filters = anchors

    sub = blocks[blocks["filter"].isin(anchors)].copy().reset_index(drop=True)
    if sub["session"].nunique() < 3:
        out.reason = "anchor filters span fewer than 3 nights"
        return out

    temp_col = choose_temp_column(sub, spec)
    if temp_col is None:
        out.reason = "no usable temperature column"
        return out
    src = (
        f"{temp_col}_first"
        if spec.temp_at_block_start and f"{temp_col}_first" in sub else temp_col
    )
    temp = pd.to_numeric(sub[src], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub["focus_pos"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(temp) & np.isfinite(y)
    sub, temp, y = sub[good].reset_index(drop=True), temp[good], y[good]

    # Base model: a level per anchor filter, and temperature split into its
    # within-night and between-night parts, but deliberately no night terms -
    # the night structure is what we are trying to see.
    #
    # Splitting temperature matters enormously here. With a single slope, the
    # fit compromises between two genuinely different responses, and whatever
    # it cannot explain lands in the per-night residuals and reads as a jumping
    # zero point. On one real archive that mistake turned a temperature effect
    # correlated with night (r = 0.91) into an apparent 12-step zero-point
    # wander, and then averaged it away again into a "stable" verdict.
    centred = temp - np.nanmean(temp)
    night_mean = (
        pd.Series(centred).groupby(sub["session"].to_numpy())
        .transform("mean").to_numpy(dtype=float)
    )
    cols = [np.ones(len(y)), centred - night_mean]
    # The between-night temperature term is only included when enough nights
    # support it. With a handful of nights it has almost as many degrees of
    # freedom as there are night means to explain, so it absorbs genuine
    # zero-point movement and the test reports "stable" for a rig that is not.
    n_test_nights = int(sub["session"].nunique())
    used_between = (
        np.nanstd(night_mean) > 1e-9
        and n_test_nights >= min_nights_for_between_temp
    )
    if used_between:
        cols.append(night_mean - np.nanmean(night_mean))
    elif np.nanstd(night_mean) > 1e-9:
        out.reason = (
            f"only {n_test_nights} nights, too few to separate a between-night "
            "temperature response from zero-point movement; the test therefore "
            "treats any temperature-correlated night structure as movement"
        )
    out.used_between_night_temp = bool(used_between)
    for f in anchors[1:]:
        cols.append((sub["filter"] == f).to_numpy(dtype=float))
    X = np.column_stack(cols)
    if len(y) <= X.shape[1] + 2:
        out.reason = "too few blocks for the drift test"
        return out
    _, _, resid, _, _ = _wls(X, y)

    t = pd.to_datetime(sub["t_mid"], utc=True, errors="coerce")
    days = (t - t.min()).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    sess = sub["session"].to_numpy()

    groups, means, mean_days = [], [], []
    for n in sorted(set(sess)):
        m = sess == n
        if m.sum() < 1:
            continue
        groups.append(resid[m])
        means.append(float(resid[m].mean()))
        mean_days.append(float(days[m].mean()))
        out.night_means[int(n)] = float(resid[m].mean())
        out.night_days[int(n)] = float(days[m].mean())
    means_a = np.asarray(means, dtype=float)
    days_a = np.asarray(mean_days, dtype=float)
    out.span_days = float(np.ptp(days_a)) if days_a.size else float("nan")
    out.between_sd = float(means_a.std(ddof=1)) if means_a.size > 1 else float("nan")

    # Pooled within-night scatter, weighted by degrees of freedom. An unweighted
    # mean of per-night variances lets a night with two blocks dominate, which
    # inflates the floor and hides real jumps behind it.
    num = sum(float(g.var(ddof=1)) * (g.size - 1) for g in groups if g.size >= 3)
    den = sum((g.size - 1) for g in groups if g.size >= 3)
    if den <= 0:  # fall back when every night is tiny
        num = sum(float(g.var(ddof=1)) * (g.size - 1) for g in groups if g.size > 1)
        den = sum((g.size - 1) for g in groups if g.size > 1)
    out.within_sd = float(np.sqrt(num / den)) if den > 0 else float("nan")
    out.n_nights_used_for_within = int(
        sum(1 for g in groups if g.size >= 3) or sum(1 for g in groups if g.size > 1)
    )

    from scipy import stats

    usable = [g for g in groups if g.size > 1]
    if len(usable) >= 2:
        try:
            F, p = stats.f_oneway(*usable)
            out.anova_F, out.anova_p = float(F), float(p)
        except Exception:
            pass

    # Smooth trend with date, and - when the design could not afford a
    # between-night temperature term - the night-mean temperature as well. Doing
    # that here costs a single parameter at the night level rather than one per
    # block, so it stays affordable with only a handful of nights. Without it,
    # ordinary night-to-night temperature variation is indistinguishable from a
    # wandering zero point and the test condemns a perfectly stable rig.
    night_temp = np.array(
        [float(np.nanmean(centred[sess == n])) for n in sorted(set(sess))],
        dtype=float,
    )
    if means_a.size >= 3 and np.ptp(days_a) > 0:
        lr = stats.linregress(days_a, means_a)
        out.drift_per_day = float(lr.slope)
        out.drift_se = float(lr.stderr)
        out.drift_p = float(lr.pvalue)
        out.drift_total_steps = float(lr.slope * np.ptp(days_a))

        cols_n = [np.ones(means_a.size), days_a - days_a.mean()]
        if not used_between and np.nanstd(night_temp) > 1e-9:
            cols_n.append(night_temp - np.nanmean(night_temp))
        Xn = np.column_stack(cols_n)
        if means_a.size > Xn.shape[1]:
            coef, *_ = np.linalg.lstsq(Xn, means_a, rcond=None)
            detrended = means_a - Xn @ coef
            dof_n = means_a.size - Xn.shape[1]
            # Scale up so the spread is comparable with an unfitted SD.
            out.between_sd_detrended = float(
                np.sqrt(float(detrended @ detrended) / max(dof_n, 1))
            )
        else:
            detrended = means_a - (lr.intercept + lr.slope * days_a)
            out.between_sd_detrended = float(detrended.std(ddof=1))
    else:
        out.between_sd_detrended = out.between_sd

    out.tested = True

    # A night's mean residual already scatters by within_sd / sqrt(n) even when
    # the zero point never moves, so comparing the spread of night means
    # directly against the block-to-block scatter is not a fair test - it hides
    # real jumps on archives with many blocks per night and invents them on
    # archives with few. What is wanted is the variance *component*: how much
    # night-to-night spread remains once the expected sampling scatter of a
    # mean is removed.
    sizes = np.array([g.size for g in groups if g.size > 0], dtype=float)
    if np.isfinite(out.within_sd) and out.within_sd > 0 and sizes.size:
        expected_var = float(out.within_sd**2 * np.mean(1.0 / sizes))
        out.expected_night_sd = float(np.sqrt(expected_var))
        excess = float(out.between_sd_detrended**2 - expected_var)
        out.night_sd = float(np.sqrt(excess)) if excess > 0 else 0.0
        out.jump_ratio = float(out.night_sd / out.within_sd)

    has_trend = bool(
        np.isfinite(out.drift_p) and out.drift_p < trend_alpha
        and abs(out.drift_total_steps) > max(out.within_sd, 1.0)
    )
    # Jumps must be statistically visible, large relative to autofocus
    # repeatability, *and* large in absolute terms. The absolute floor matters:
    # a focuser moves in whole steps, so zero-point movement below about one
    # step cannot be acted on however significant it is, and on very clean data
    # a purely relative threshold would flag it.
    has_jumps = bool(
        np.isfinite(out.night_sd)
        and out.night_sd > max(jump_threshold * out.within_sd, jump_min_steps)
        and (not np.isfinite(out.anova_p) or out.anova_p < 0.05)
    )
    out.verdict = "jumps" if has_jumps else ("drift" if has_trend else "stable")
    return out
