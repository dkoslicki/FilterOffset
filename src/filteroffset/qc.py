"""Frame and block quality control.

The filter offsets are only as good as the frames they came from, so every
frame is screened before it reaches a model and every rejection is recorded
with a reason.  Checks are deliberately *relative* wherever possible - compared
against the same filter on the same night - because absolute thresholds on star
counts or background level are meaningless across different filters, targets
and sky conditions.

Severities
----------
``fail``
    The frame should not inform the fit.
``warn``
    Usable, but worth knowing about; reported, not excluded.
``info``
    Context for the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QcParams:
    """Thresholds for the quality checks."""

    #: Fail a frame whose star count is below this fraction of the median for
    #: the same filter and night (transparency loss, clouds, dome, dew).
    star_count_frac: float = 0.55
    #: Warn below this fraction.
    star_count_warn_frac: float = 0.75
    #: Absolute floor on usable stars.
    min_stars: int = 25
    #: Fail above this median eccentricity (trailing from guiding or wind).
    ecc_fail: float = 0.60
    ecc_warn: float = 0.50
    #: Fail a frame whose star size exceeds the same-filter/night median by
    #: this factor (a robust outlier test, not an absolute size limit).
    hfd_fail_ratio: float = 1.35
    hfd_warn_ratio: float = 1.18
    #: Robust z-score beyond which a frame's background is anomalous.
    background_z: float = 5.0
    #: Fractional disagreement between backends that triggers a warning.
    backend_disagree_frac: float = 0.25
    #: Bright-quartile HFD rise indicating saturation is inflating sizes.
    saturation_rise_px: float = 0.35
    #: Minimum frames in a block for its measurements to be usable.
    min_frames_per_block: int = 1


def _num(value) -> float | None:
    """Float or None - never collapsing a legitimate zero into a missing value."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _robust_z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(v)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(v)
    return (v - med) / scale


def check_frames(
    frames: pd.DataFrame,
    profiles: pd.DataFrame,
    params: QcParams | None = None,
    primary_backend: str = "internal",
    metric: str = "hfd_median_bright",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen frames and return ``(frames_with_qc, flags)``.

    ``frames_with_qc`` gains ``qc_pass``, ``qc_severity`` and ``qc_reasons``
    plus the merged measurements from *primary_backend*.  ``flags`` is a long
    table with one row per (frame, check) so the report can enumerate exactly
    why anything was dropped.
    """
    params = params or QcParams()
    flags: list[dict[str, Any]] = []

    df = frames.copy().reset_index(drop=True)
    if profiles is None or profiles.empty:
        df["qc_pass"] = True
        df["qc_severity"] = "info"
        df["qc_reasons"] = ""
        return df, pd.DataFrame(columns=["path", "check", "severity", "detail"])

    prim = profiles[profiles["backend"].astype(str) == primary_backend].copy()
    if prim.empty:
        prim = profiles.copy()
    keep_cols = [
        c for c in prim.columns
        if c not in ("backend",) and (c == "path" or c not in df.columns)
    ]
    df = df.merge(prim[keep_cols], on="path", how="left")

    # Measurement failures from the backend itself.
    if "ok" in df.columns:
        for row in df[df["ok"].fillna(True) == False].itertuples():  # noqa: E712
            flags.append(
                {
                    "path": row.path,
                    "check": "measurement_failed",
                    "severity": "fail",
                    "detail": str(getattr(row, "error", "backend reported failure")),
                }
            )

    grp_cols = [c for c in ("session", "filter") if c in df.columns]
    if not grp_cols:
        grp_cols = ["filter"] if "filter" in df.columns else []

    def _grouped_median(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        vals = pd.to_numeric(df[col], errors="coerce")
        if not grp_cols:
            return pd.Series(vals.median(), index=df.index)
        return vals.groupby([df[c] for c in grp_cols]).transform("median")

    # --- star counts: transparency ------------------------------------------
    if "n_stars" in df.columns:
        n = pd.to_numeric(df["n_stars"], errors="coerce")
        ref = _grouped_median("n_stars")
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = n / ref
        for i, row in df.iterrows():
            f = float(frac.iloc[i]) if np.isfinite(frac.iloc[i]) else np.nan
            if np.isfinite(n.iloc[i]) and n.iloc[i] < params.min_stars:
                flags.append({
                    "path": row["path"], "check": "too_few_stars", "severity": "fail",
                    "detail": f"{int(n.iloc[i])} stars (absolute floor {params.min_stars})",
                })
            elif np.isfinite(f) and f < params.star_count_frac:
                flags.append({
                    "path": row["path"], "check": "star_count_low", "severity": "fail",
                    "detail": (
                        f"{int(n.iloc[i])} stars = {100*f:.0f}% of the "
                        f"{row.get('filter')} median this night - likely cloud "
                        "or transparency loss"
                    ),
                })
            elif np.isfinite(f) and f < params.star_count_warn_frac:
                flags.append({
                    "path": row["path"], "check": "star_count_reduced",
                    "severity": "warn",
                    "detail": f"{int(n.iloc[i])} stars = {100*f:.0f}% of the nightly median",
                })

    # --- eccentricity: guiding / tracking / wind ----------------------------
    if "ecc_median" in df.columns and pd.to_numeric(df["ecc_median"], errors="coerce").notna().any():
        ecc = pd.to_numeric(df["ecc_median"], errors="coerce")
        base = float(np.nanmedian(ecc))
        for i, row in df.iterrows():
            e = float(ecc.iloc[i]) if np.isfinite(ecc.iloc[i]) else np.nan
            if not np.isfinite(e):
                continue
            theta = pd.to_numeric(
                pd.Series([row.get("theta_concentration", np.nan)]), errors="coerce"
            ).iloc[0]
            direction = (
                " with a common elongation direction (tracking/guiding, not seeing)"
                if np.isfinite(theta) and theta > 0.5 else ""
            )
            if e > params.ecc_fail:
                flags.append({
                    "path": row["path"], "check": "elongated_stars", "severity": "fail",
                    "detail": f"median eccentricity {e:.2f} (night median {base:.2f}){direction}",
                })
            elif e > params.ecc_warn:
                flags.append({
                    "path": row["path"], "check": "elongated_stars", "severity": "warn",
                    "detail": f"median eccentricity {e:.2f} (night median {base:.2f}){direction}",
                })

    # --- star size outliers within filter/night -----------------------------
    if metric in df.columns:
        h = pd.to_numeric(df[metric], errors="coerce")
        ref = _grouped_median(metric)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = h / ref
        for i, row in df.iterrows():
            r = float(ratio.iloc[i]) if np.isfinite(ratio.iloc[i]) else np.nan
            if not np.isfinite(r):
                continue
            if r > params.hfd_fail_ratio:
                flags.append({
                    "path": row["path"], "check": "star_size_outlier", "severity": "fail",
                    "detail": (
                        f"{metric} {h.iloc[i]:.2f} is x{r:.2f} the "
                        f"{row.get('filter')} median this night"
                    ),
                })
            elif r > params.hfd_warn_ratio:
                flags.append({
                    "path": row["path"], "check": "star_size_elevated", "severity": "warn",
                    "detail": f"{metric} {h.iloc[i]:.2f} is x{r:.2f} the nightly median",
                })

    # --- background anomalies: moon, cloud, gradient ------------------------
    if "background" in df.columns and pd.to_numeric(df["background"], errors="coerce").notna().any():
        for _keys, g in (
            df.groupby([df[c] for c in grp_cols]) if grp_cols else [((), df)]
        ):
            z = _robust_z(pd.to_numeric(g["background"], errors="coerce").to_numpy())
            for (_, row), zz in zip(g.iterrows(), z):
                if np.isfinite(zz) and abs(zz) > params.background_z:
                    flags.append({
                        "path": row["path"], "check": "background_anomaly",
                        "severity": "warn",
                        "detail": f"sky level {row['background']:.0f} ADU, robust z={zz:+.1f}",
                    })

    # --- saturation inflating star sizes ------------------------------------
    if "hfd_saturation_rise" in df.columns:
        rise = pd.to_numeric(df["hfd_saturation_rise"], errors="coerce")
        for i, row in df.iterrows():
            if np.isfinite(rise.iloc[i]) and rise.iloc[i] > params.saturation_rise_px:
                flags.append({
                    "path": row["path"], "check": "saturation_inflating_hfd",
                    "severity": "warn",
                    "detail": (
                        f"brightest-quartile star size exceeds the next quartile by "
                        f"{rise.iloc[i]:.2f} px - saturated cores are inflating HFD"
                    ),
                })

    # --- backend disagreement ------------------------------------------------
    flags.extend(_backend_disagreement(profiles, params, metric))

    flag_df = pd.DataFrame(flags, columns=["path", "check", "severity", "detail"])
    df = _apply_flags(df, flag_df)
    return df, flag_df


def _backend_disagreement(
    profiles: pd.DataFrame, params: QcParams, metric: str
) -> list[dict[str, Any]]:
    """Compare backends on the *same* frame.

    Absolute star sizes are not expected to agree - half-flux diameter depends
    on the measurement aperture, which the two implementations choose
    differently - so the comparison is made after removing each backend's own
    median scale.  What must agree is the frame-to-frame *pattern*.
    """
    out: list[dict[str, Any]] = []
    if "backend" not in profiles.columns or metric not in profiles.columns:
        return out
    backends = sorted(profiles["backend"].dropna().unique().tolist())
    if len(backends) < 2:
        return out
    wide = profiles.pivot_table(index="path", columns="backend", values=metric)
    if wide.shape[1] < 2:
        return out
    scaled = wide / wide.median(axis=0)
    a, b = backends[0], backends[1]
    if a not in scaled or b not in scaled:
        return out
    rel = (scaled[a] - scaled[b]).abs()
    for path, val in rel.items():
        if np.isfinite(val) and val > params.backend_disagree_frac:
            out.append({
                "path": path, "check": "backend_disagreement", "severity": "warn",
                "detail": (
                    f"{a} and {b} disagree by {100*val:.0f}% on scale-normalised "
                    f"{metric}; treat this frame's size measurement with caution"
                ),
            })
    return out


def _apply_flags(df: pd.DataFrame, flag_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if flag_df.empty:
        df["qc_pass"] = True
        df["qc_severity"] = "ok"
        df["qc_reasons"] = ""
        return df
    fails = set(flag_df.loc[flag_df["severity"] == "fail", "path"])
    warns = set(flag_df.loc[flag_df["severity"] == "warn", "path"])
    reasons = (
        flag_df.assign(text=flag_df["check"] + ": " + flag_df["detail"].astype(str))
        .groupby("path")["text"]
        .apply(lambda s: " | ".join(s))
    )
    df["qc_pass"] = ~df["path"].isin(fails)
    df["qc_severity"] = np.where(
        df["path"].isin(fails), "fail",
        np.where(df["path"].isin(warns), "warn", "ok"),
    )
    df["qc_reasons"] = df["path"].map(reasons).fillna("")
    return df


def check_blocks(
    blocks: pd.DataFrame,
    frames_qc: pd.DataFrame,
    params: QcParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen focus blocks and return ``(blocks_with_qc, flags)``."""
    params = params or QcParams()
    flags: list[dict[str, Any]] = []
    bl = blocks.copy().reset_index(drop=True)

    if not frames_qc.empty and "qc_pass" in frames_qc.columns:
        counts = (
            frames_qc.groupby("block")["qc_pass"]
            .agg(n_frames_qc="size", n_pass=lambda s: int(s.sum()))
            .reset_index()
        )
        bl = bl.merge(counts, on="block", how="left")
    else:
        bl["n_pass"] = bl.get("n_frames")

    for row in bl.itertuples():
        b = int(row.block)
        if getattr(row, "af_likely", None) is False:
            flags.append({
                "block": b, "check": "autofocus_not_detected", "severity": "fail",
                "detail": (
                    f"only {getattr(row, 'dead_time_before_s', float('nan')):.0f}s of dead "
                    "time before this block - too little for an autofocus run, so its "
                    "position may be inherited rather than freshly measured"
                ),
            })
        if getattr(row, "temp_comp_active", False):
            flags.append({
                "block": b, "check": "temperature_compensation_active", "severity": "fail",
                "detail": (
                    "the focuser moved during this filter run, so the recorded position "
                    "is not a single autofocus result"
                ),
            })
        # NB: `float(n_pass or np.nan)` used to sit here, which turned the one
        # value the check exists to catch - zero - into NaN, so a block whose
        # every frame had been rejected still entered the fit at full weight.
        n_pass = _num(getattr(row, "n_pass", None))
        if n_pass is not None and n_pass < params.min_frames_per_block:
            flags.append({
                "block": b, "check": "no_usable_frames", "severity": "fail",
                "detail": "every frame in this block failed quality control",
            })
        rng = _num(getattr(row, "ambient_temp_range", None))
        if rng is not None and rng > 1.0:
            flags.append({
                "block": b, "check": "temperature_drifted_within_block",
                "severity": "warn",
                "detail": (
                    f"temperature moved {float(rng):.2f} C during the block, so a single "
                    "block temperature is a coarse summary"
                ),
            })

    flag_df = pd.DataFrame(flags, columns=["block", "check", "severity", "detail"])
    if flag_df.empty:
        bl["qc_pass"] = True
        bl["qc_reasons"] = ""
    else:
        fails = set(flag_df.loc[flag_df["severity"] == "fail", "block"])
        reasons = (
            flag_df.assign(text=flag_df["check"] + ": " + flag_df["detail"].astype(str))
            .groupby("block")["text"].apply(lambda s: " | ".join(s))
        )
        bl["qc_pass"] = ~bl["block"].isin(fails)
        bl["qc_reasons"] = bl["block"].map(reasons).fillna("")
    return bl, flag_df
