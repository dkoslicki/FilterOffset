"""Independent, in-process star profile measurement.

This backend exists to check ASTAP rather than to replace it.  It measures the
half-flux diameter directly from a curve of growth (:func:`sep.flux_radius`),
which is the same physical definition ASTAP uses but a completely separate
implementation, and it additionally provides the second-moment shape
parameters ASTAP's CSV does not export (needed to tell defocus apart from
trailing caused by guiding or wind).

By default only a centred crop is read.  That keeps peak memory to a few
hundred MB per worker on a 61-megapixel frame, and it is also the scientifically
cleaner choice: field curvature, tilt and coma inflate corner HFD without being
filter-dependent focus.  Pass ``crop=None`` for whole-frame measurement when
tilt diagnostics are wanted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import FrameProfile, ProfileParams, summarise_stars

_PIXSTACK_SET = False


def _gaussian_kernel(fwhm: float = 3.0, size: int = 7) -> np.ndarray:
    """Matched filter for detection, as SExtractor uses by default.

    Convolving before thresholding suppresses single-pixel noise spikes and
    materially reduces spurious detections without shifting the measured sizes,
    which are always computed on the unconvolved image.
    """
    sigma = max(fwhm, 0.5) / 2.3548
    radius = (int(size) - 1) // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _load_region(path: str, crop: int | None) -> tuple[np.ndarray, int, int, int, int]:
    """Return (image, x_offset, y_offset, naxis1, naxis2).

    Uses ``hdu.section`` so that a crop reads only the needed rows from disk
    instead of materialising a 123 MB frame.
    """
    from astropy.io import fits

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ``do_not_scale_image_data`` keeps the on-disk integer dtype so that a
        # memory map is possible at all (astropy refuses to memmap when
        # BZERO/BSCALE are present); the scaling is applied by hand below, on
        # the crop only, which also keeps peak memory down.
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
            hdu = next(
                (h for h in hdul if h.header.get("NAXIS", 0) >= 2),
                hdul[0],
            )
            n1 = int(hdu.header["NAXIS1"])
            n2 = int(hdu.header["NAXIS2"])
            bzero = float(hdu.header.get("BZERO", 0.0))
            bscale = float(hdu.header.get("BSCALE", 1.0))
            if crop is None or crop <= 0 or (crop >= n1 and crop >= n2):
                raw = hdu.section[:, :]
                x0 = y0 = 0
            else:
                half = int(crop) // 2
                cx, cy = n1 // 2, n2 // 2
                x0, x1 = max(0, cx - half), min(n1, cx + half)
                y0, y1 = max(0, cy - half), min(n2, cy + half)
                raw = hdu.section[y0:y1, x0:x1]
            data = np.asarray(raw, dtype=np.float32)
            if bscale != 1.0:
                data *= bscale
            if bzero != 0.0:
                data += bzero
            return data, x0, y0, n1, n2


def _extract_sep(
    data: np.ndarray, params: ProfileParams
) -> tuple[pd.DataFrame, float, float]:
    import sep

    global _PIXSTACK_SET
    if not _PIXSTACK_SET:
        # Dense Milky Way fields with big defocused stars overflow the default.
        try:
            sep.set_extract_pixstack(10_000_000)
            sep.set_sub_object_limit(2048)
        except Exception:
            pass
        _PIXSTACK_SET = True

    img = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(img, bw=64, bh=64, fw=3, fh=3)
    back_level = float(bkg.globalback)
    back_rms = float(bkg.globalrms)
    img_sub = img - bkg.back()
    del img

    objs = sep.extract(
        img_sub,
        thresh=params.detect_sigma,
        err=back_rms,
        minarea=params.min_area,
        filter_kernel=_gaussian_kernel(params.filter_fwhm),
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        clean_param=1.0,
    )
    if len(objs) == 0:
        return pd.DataFrame(), back_level, back_rms

    x = objs["x"].astype(float)
    y = objs["y"].astype(float)
    a = objs["a"].astype(float)
    b = objs["b"].astype(float)
    theta = objs["theta"].astype(float)
    peak = objs["peak"].astype(float) + back_level

    # Flux in a generous circular aperture, then the half-flux radius inside it.
    kron_r = 3.0 * np.maximum(a, 1.0)
    rmax = np.clip(kron_r, 4.0, 60.0)
    flux, fluxerr, _ = sep.sum_circle(img_sub, x, y, rmax, err=back_rms, subpix=5)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(fluxerr > 0, flux / fluxerr, np.nan)

    r_half, flag = sep.flux_radius(
        img_sub, x, y, rmax, 0.5, normflux=flux, subpix=5
    )
    hfd = 2.0 * np.asarray(r_half, dtype=float)
    hfd[np.asarray(flag, dtype=int) != 0] = np.nan

    # Second-moment FWHM, for comparison with the HFD.
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(np.maximum(a * b, 0.0))
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

    df = pd.DataFrame(
        {
            "x": x, "y": y, "hfd": hfd, "snr": snr, "flux": flux,
            "peak": peak, "a": a, "b": b, "theta": theta, "fwhm": fwhm,
        }
    )
    return df, back_level, back_rms


def _extract_photutils(
    data: np.ndarray, params: ProfileParams
) -> tuple[pd.DataFrame, float, float]:
    """Fallback extractor when :mod:`sep` is unavailable."""
    from astropy.stats import sigma_clipped_stats
    from photutils.aperture import CircularAperture, aperture_photometry
    from photutils.detection import DAOStarFinder

    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    finder = DAOStarFinder(fwhm=4.0, threshold=params.detect_sigma * float(std))
    tbl = finder(data - median)
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(), float(median), float(std)

    x = np.asarray(tbl["xcentroid"], dtype=float)
    y = np.asarray(tbl["ycentroid"], dtype=float)
    sub = data - median

    # Curve of growth: find the radius enclosing half the flux at r=rmax.
    radii = np.arange(1.0, 25.0, 0.5)
    pos = np.column_stack([x, y])
    curves = np.empty((len(x), radii.size), dtype=float)
    for j, r in enumerate(radii):
        phot = aperture_photometry(sub, CircularAperture(pos, r=r))
        curves[:, j] = np.asarray(phot["aperture_sum"], dtype=float)
    total = curves[:, -1]
    hfd = np.full(len(x), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = curves / total[:, None]
    for i in range(len(x)):
        if not np.isfinite(total[i]) or total[i] <= 0:
            continue
        f = frac[i]
        k = int(np.searchsorted(f, 0.5))
        if k <= 0 or k >= radii.size:
            continue
        f0, f1 = f[k - 1], f[k]
        r0, r1 = radii[k - 1], radii[k]
        hfd[i] = 2.0 * (r0 + (0.5 - f0) * (r1 - r0) / max(f1 - f0, 1e-9))

    npix = np.pi * (radii[-1] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = total / np.sqrt(np.maximum(total, 0) + npix * float(std) ** 2)
    df = pd.DataFrame(
        {
            "x": x, "y": y, "hfd": hfd, "snr": snr, "flux": total,
            "peak": np.asarray(tbl["peak"], dtype=float) + float(median),
            "a": np.nan, "b": np.nan, "theta": np.nan,
            "fwhm": np.asarray(tbl.get("fwhm", np.full(len(x), np.nan)), dtype=float),
        }
    )
    return df, float(median), float(std)


def measure_frame(
    path: str,
    params: ProfileParams | None = None,
    naxis1: float | None = None,
    naxis2: float | None = None,
    full_well: float | None = None,
    timeout: float = 300.0,
    workdir: str | None = None,
) -> FrameProfile:
    """Measure one frame's star profiles without leaving the process."""
    params = params or ProfileParams()
    try:
        data, x0, y0, n1, n2 = _load_region(path, params.crop)
    except Exception as exc:
        return FrameProfile(path, "internal", ok=False, error=f"read failed: {exc}")

    try:
        try:
            import sep  # noqa: F401

            stars, back, rms = _extract_sep(data, params)
            engine = "sep"
        except ImportError:
            stars, back, rms = _extract_photutils(data, params)
            engine = "photutils"
    except Exception as exc:
        return FrameProfile(path, "internal", ok=False, error=f"extract failed: {exc}")
    finally:
        del data

    if stars.empty:
        return FrameProfile(path, "internal", ok=False, error="no sources detected")

    # Put coordinates back in full-frame pixels so radial statistics are right.
    stars["x"] = stars["x"] + x0
    stars["y"] = stars["y"] + y0

    stats = summarise_stars(
        stars,
        params,
        naxis1=naxis1 or n1,
        naxis2=naxis2 or n2,
        background=back,
        background_rms=rms,
        full_well=full_well,
        backend="internal",
    )
    stats["internal_engine"] = engine
    stats["internal_cropped"] = bool(params.crop and (params.crop < min(n1, n2)))
    ok = stats.get("n_stars", 0) >= params.min_stars
    err = None if ok else f"only {stats.get('n_stars', 0)} usable stars"
    return FrameProfile(path, "internal", ok=ok, error=err, stats=stats)
