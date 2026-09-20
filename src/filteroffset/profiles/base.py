"""Shared star-table schema and summary statistics for all backends."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Columns every backend must provide.  ``a``/``b``/``theta``/``fwhm`` are
#: optional and filled with NaN when a backend cannot measure them.
STAR_COLUMNS = ("x", "y", "hfd", "snr", "flux", "peak", "a", "b", "theta", "fwhm")


class ProfileBackendError(RuntimeError):
    """Raised when a measurement backend cannot run or fails on a frame."""


@dataclass
class ProfileParams:
    """Knobs shared by both backends.

    The defaults are chosen so the two backends see a comparable star sample.

    The ``bright_*_pct`` pair is the important one for filter-offset work.  HFD
    measured on faint stars is biased upward by sky background, so a filter
    sitting over a brighter sky (light pollution, the moon, a wide passband)
    will *look* softer even when perfectly focused.  Statistics suffixed
    ``_bright`` therefore use a *flux percentile band* within each frame rather
    than an absolute cut, which makes the comparison between filters fair and,
    critically, makes it identical across backends whose SNR scales differ.
    ASTAP's reported SNR is not on the same scale as a sep flux/flux-error
    ratio, so any absolute SNR threshold would silently mean different things.

    The upper bound excludes the very brightest stars, whose cores saturate and
    whose HFD is consequently inflated.
    """

    snr_min: float = 10.0
    #: Flux-percentile band (within each frame) defining the "bright" sample.
    bright_lo_pct: float = 70.0
    bright_hi_pct: float = 97.0
    #: Require at least this many stars in the band, else fall back to all.
    bright_min_stars: int = 20
    #: Reject stars whose peak exceeds this fraction of full well.
    saturation_frac: float = 0.90
    #: Assumed full-well ADU when the header does not say.
    full_well_adu: float = 65535.0
    #: Only use stars inside this fraction of the half-diagonal.  Field
    #: curvature, tilt and coma all inflate corner HFD, none of which is
    #: filter-dependent focus, so the centre is the cleanest probe.
    center_radius_frac: float = 0.40
    #: Side length of the centred square crop the internal backend reads.
    #: ``None`` reads the whole frame (slow and memory hungry on a QHY600).
    crop: int | None = 3000
    #: Minimum number of stars for a frame's measurement to be trusted.
    min_stars: int = 25
    #: sep/photutils detection threshold in units of background sigma.
    detect_sigma: float = 5.0
    #: Minimum connected pixels above threshold for a detection.
    min_area: int = 5
    #: FWHM (px) of the matched filter applied before thresholding.
    filter_fwhm: float = 3.0
    #: Discard stars larger than this many pixels (satellites, galaxies).
    hfd_max: float = 40.0
    #: Reject detections smaller than this fraction of the frame's own stellar
    #: scale.  Hot pixels and cosmic rays are bright enough to survive any SNR
    #: cut but are only ~1 px across, so the discriminator has to be
    #: morphological.  The scale is measured from the frame's brightest
    #: detections, which are reliably real stars, making this self-calibrating
    #: across very different exposures, filters and sky levels.
    min_hfd_frac_of_psf: float = 0.55
    #: Absolute floor in pixels, as a backstop when the scale itself is suspect.
    hfd_min_abs: float = 1.0
    #: Flux percentile defining the "reliably real stars" used for the scale.
    psf_scale_pct: float = 90.0
    #: Backends the *relative* floor applies to.  It is correct only where the
    #: measured size is essentially independent of brightness for real stars,
    #: which holds for the adaptive-aperture internal extractor and not for
    #: ASTAP, whose HFD rises steeply with flux by construction.  Applying it to
    #: ASTAP was measured to discard ~15% of genuine faint stars from a clean
    #: frame.  ASTAP also needs it less: it validates stellar profiles during
    #: detection, so hot pixels and cosmic rays rarely reach its output, whereas
    #: the internal extractor is a raw thresholder.  The absolute floor
    #: (`hfd_min_abs`) still applies everywhere.
    relative_floor_backends: tuple[str, ...] = ("internal",)

    def key(self) -> str:
        """A short deterministic string identifying these parameters."""
        import hashlib
        import json

        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass
class FrameProfile:
    """Summary of one frame's star field, as produced by one backend."""

    path: str
    backend: str
    ok: bool = True
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "path": self.path,
            "backend": self.backend,
            "ok": bool(self.ok),
            "error": self.error,
        }
        row.update(self.stats)
        return row


def _q(values: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(values, q)) if values.size else float("nan")


def _robust_sigma(values: np.ndarray) -> float:
    """Median absolute deviation scaled to a Gaussian sigma."""
    if values.size < 3:
        return float("nan")
    med = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - med)))


def summarise_stars(
    stars: pd.DataFrame,
    params: ProfileParams,
    naxis1: float | None = None,
    naxis2: float | None = None,
    background: float | None = None,
    background_rms: float | None = None,
    full_well: float | None = None,
    backend: str = "internal",
) -> dict[str, Any]:
    """Reduce a per-star table to the per-frame statistics the models use.

    This is deliberately backend-agnostic: both ASTAP and the internal
    extractor are summarised by this exact code so that any difference in the
    reported numbers comes from detection/measurement, not from statistics.
    """
    out: dict[str, Any] = {}
    df = stars.copy()
    for col in STAR_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    out["n_detected"] = int(len(df))
    out["background"] = float(background) if background is not None else float("nan")
    out["background_rms"] = (
        float(background_rms) if background_rms is not None else float("nan")
    )

    # --- rejection cascade, reported so the report can explain the sample ----
    well = float(full_well if full_well else params.full_well_adu)
    n0 = len(df)
    df = df[np.isfinite(df["hfd"]) & (df["hfd"] > 0) & (df["hfd"] < params.hfd_max)]
    out["n_rej_hfd"] = n0 - len(df)

    # --- reject detections too compact to be stars --------------------------
    # Estimated from the frame itself: the brightest detections are real stars,
    # so their median half-flux diameter sets the scale that anything stellar
    # must approach.  On a long narrowband sub over a dark sky this removes the
    # hot-pixel and cosmic-ray population, which is otherwise numerous enough to
    # dominate the median and drag it below one pixel.
    psf_scale = float("nan")
    n0 = len(df)
    if len(df) >= 20:
        fl = df["flux"].to_numpy(float)
        finite = np.isfinite(fl)
        if finite.sum() >= 20:
            bright_cut = np.nanpercentile(fl[finite], params.psf_scale_pct)
            bright_hfd = df.loc[df["flux"] >= bright_cut, "hfd"].to_numpy(float)
            if bright_hfd.size >= 5:
                psf_scale = float(np.nanmedian(bright_hfd))
    floor = params.hfd_min_abs
    use_relative = str(backend).lower() in {
        b.lower() for b in params.relative_floor_backends
    }
    if use_relative and np.isfinite(psf_scale) and psf_scale > 0:
        floor = max(floor, params.min_hfd_frac_of_psf * psf_scale)
    df = df[df["hfd"] >= floor]
    out["psf_scale_px"] = psf_scale
    out["hfd_floor_px"] = float(floor)
    out["n_rej_toosmall"] = n0 - len(df)
    out["frac_rej_toosmall"] = float((n0 - len(df)) / n0) if n0 else 0.0

    n0 = len(df)
    if df["peak"].notna().any():
        df = df[~(df["peak"] > params.saturation_frac * well)]
    out["n_rej_saturated"] = n0 - len(df)

    n0 = len(df)
    df = df[~(df["snr"] < params.snr_min)]
    out["n_rej_lowsnr"] = n0 - len(df)

    out["n_stars"] = int(len(df))
    if df.empty:
        for k in (
            "hfd_median", "hfd_mad", "hfd_p25", "hfd_p75", "hfd_mean",
            "hfd_median_bright", "n_bright", "hfd_median_center",
            "n_center", "hfd_median_center_bright", "n_center_bright",
            "ecc_median", "fwhm_median", "snr_median", "flux_median",
            "hfd_faint_minus_bright", "hfd_se", "hfd_bright_se",
            "hfd_saturation_rise", "hfd_fluxq1", "hfd_fluxq2",
            "hfd_fluxq3", "hfd_fluxq4", "psf_scale_px", "hfd_floor_px",
            "frac_rej_toosmall",
        ):
            out[k] = float("nan")
        out["n_bright"] = 0
        out["n_center"] = 0
        out["n_center_bright"] = 0
        out["n_rej_toosmall"] = out.get("n_rej_toosmall", 0)
        return out

    hfd = df["hfd"].to_numpy(float)
    snr = df["snr"].to_numpy(float)
    out["hfd_median"] = float(np.nanmedian(hfd))
    out["hfd_mean"] = float(np.nanmean(hfd))
    out["hfd_mad"] = _robust_sigma(hfd)
    out["hfd_p25"] = _q(hfd, 25)
    out["hfd_p75"] = _q(hfd, 75)
    out["snr_median"] = float(np.nanmedian(snr))
    out["flux_median"] = float(np.nanmedian(df["flux"].to_numpy(float)))
    # Standard error of the median, for weighting frames in the V-curve fit.
    sig = out["hfd_mad"]
    out["hfd_se"] = (
        float(1.2533 * sig / math.sqrt(len(hfd))) if np.isfinite(sig) and len(hfd) else float("nan")
    )

    # --- matched bright sample, defined by flux rank within this frame ------
    flux = df["flux"].to_numpy(float)
    finite_flux = np.isfinite(flux)
    if finite_flux.sum() >= params.bright_min_stars:
        lo = np.nanpercentile(flux[finite_flux], params.bright_lo_pct)
        hi = np.nanpercentile(flux[finite_flux], params.bright_hi_pct)
        band = df[(df["flux"] >= lo) & (df["flux"] <= hi)]
        if len(band) < params.bright_min_stars:
            band = df[df["flux"] >= lo]
    else:
        band = df
    out["n_bright"] = int(len(band))
    out["hfd_median_bright"] = (
        float(np.nanmedian(band["hfd"].to_numpy(float))) if len(band) else float("nan")
    )
    bs = _robust_sigma(band["hfd"].to_numpy(float))
    out["hfd_bright_se"] = (
        float(1.2533 * bs / math.sqrt(len(band)))
        if np.isfinite(bs) and len(band) else float("nan")
    )

    # --- HFD versus flux quartile -------------------------------------------
    # This is the diagnostic that separates a genuinely defocused filter from
    # one that merely sits over a brighter sky.  Defocus inflates HFD for every
    # star equally, so the profile across quartiles stays flat.  Background /
    # SNR bias inflates only the faint end, giving a declining profile.  A
    # rising top quartile instead means saturation.
    if finite_flux.sum() >= 20:
        edges = np.nanpercentile(flux[finite_flux], [0, 25, 50, 75, 100])
        qmeds: list[float] = []
        for qi in range(4):
            m = (flux >= edges[qi]) & (
                flux <= edges[qi + 1] if qi == 3 else flux < edges[qi + 1]
            )
            qmeds.append(
                float(np.nanmedian(hfd[m])) if m.sum() >= 5 else float("nan")
            )
        for qi, val in enumerate(qmeds, start=1):
            out[f"hfd_fluxq{qi}"] = val
        out["hfd_faint_minus_bright"] = qmeds[0] - qmeds[3]
        out["hfd_saturation_rise"] = qmeds[3] - qmeds[2]
    else:
        for qi in range(1, 5):
            out[f"hfd_fluxq{qi}"] = float("nan")
        out["hfd_faint_minus_bright"] = float("nan")
        out["hfd_saturation_rise"] = float("nan")

    # --- field-position resolved statistics ---------------------------------
    if naxis1 and naxis2:
        cx, cy = float(naxis1) / 2.0, float(naxis2) / 2.0
        half_diag = math.hypot(cx, cy)
        r = np.hypot(df["x"].to_numpy(float) - cx, df["y"].to_numpy(float) - cy)
        rn = r / half_diag
        out["r_norm_median"] = float(np.nanmedian(rn))
        center = df[rn <= params.center_radius_frac]
        out["n_center"] = int(len(center))
        out["hfd_median_center"] = (
            float(np.nanmedian(center["hfd"].to_numpy(float))) if len(center) else float("nan")
        )
        cb = center[center["flux"] >= np.nanpercentile(
            flux[finite_flux], params.bright_lo_pct
        )] if finite_flux.sum() >= params.bright_min_stars else center
        out["n_center_bright"] = int(len(cb))
        out["hfd_median_center_bright"] = (
            float(np.nanmedian(cb["hfd"].to_numpy(float))) if len(cb) else float("nan")
        )
        outer = df[rn > 0.75]
        out["hfd_median_outer"] = (
            float(np.nanmedian(outer["hfd"].to_numpy(float))) if len(outer) >= 5 else float("nan")
        )
        # Corner-minus-centre HFD flags tilt / backfocus, which is a confounder
        # for any focus measurement taken off-axis.
        out["hfd_outer_minus_center"] = out["hfd_median_outer"] - out["hfd_median_center"]
        # Crude tilt metric: HFD imbalance between opposite field halves.
        x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
        h = df["hfd"].to_numpy(float)
        def _half(mask):
            return float(np.nanmedian(h[mask])) if mask.sum() >= 5 else float("nan")
        out["hfd_tilt_x"] = _half(x > cx) - _half(x <= cx)
        out["hfd_tilt_y"] = _half(y > cy) - _half(y <= cy)
    else:
        for k in (
            "r_norm_median", "n_center", "hfd_median_center",
            "n_center_bright", "hfd_median_center_bright", "hfd_median_outer",
            "hfd_outer_minus_center", "hfd_tilt_x", "hfd_tilt_y",
        ):
            out[k] = float("nan")
        out["n_center"] = 0
        out["n_center_bright"] = 0

    # --- shape (guiding / tracking) -----------------------------------------
    # Second moments of faint sources are dominated by pixel noise, which
    # biases eccentricity high, so shape is measured on the bright band only.
    shape_src = band if len(band) >= params.bright_min_stars else df
    a, b = shape_src["a"].to_numpy(float), shape_src["b"].to_numpy(float)
    good = np.isfinite(a) & np.isfinite(b) & (a > 0)
    if good.sum() >= 5:
        ratio = np.clip(b[good] / a[good], 0.0, 1.0)
        out["ecc_median"] = float(np.nanmedian(np.sqrt(1.0 - ratio**2)))
        out["elong_median"] = float(np.nanmedian(a[good] / np.maximum(b[good], 1e-6)))
        th = shape_src["theta"].to_numpy(float)[good]
        # Resultant length of the doubled position angles: near 1 means the
        # elongation shares a common direction (tracking), near 0 means random.
        out["theta_concentration"] = float(
            np.hypot(np.nanmean(np.cos(2 * th)), np.nanmean(np.sin(2 * th)))
        )
    else:
        out["ecc_median"] = float("nan")
        out["elong_median"] = float("nan")
        out["theta_concentration"] = float("nan")

    fw = shape_src["fwhm"].to_numpy(float)
    out["fwhm_median"] = float(np.nanmedian(fw)) if np.isfinite(fw).sum() >= 5 else float("nan")
    fwa = df["fwhm"].to_numpy(float)
    out["fwhm_median_all"] = (
        float(np.nanmedian(fwa)) if np.isfinite(fwa).sum() >= 5 else float("nan")
    )
    return out
