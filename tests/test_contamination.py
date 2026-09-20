"""Rejection of detections too compact to be stars."""

import numpy as np
import pandas as pd
import pytest

from filteroffset.profiles import ProfileParams, summarise_stars


def field(n_stars=400, hfd=2.8, n_hot=0, hot_hfd=0.85, seed=0):
    """A star field, optionally contaminated with hot pixels / cosmic rays.

    The contaminants are deliberately *bright*: that is what makes them immune
    to signal-to-noise cuts and why the discriminator has to be morphological.
    """
    rng = np.random.default_rng(seed)
    real = pd.DataFrame(
        {
            "x": rng.uniform(0, 9600, n_stars), "y": rng.uniform(0, 6422, n_stars),
            "hfd": rng.normal(hfd, 0.12, n_stars),
            "snr": rng.uniform(20, 600, n_stars),
            "flux": rng.lognormal(10, 1.1, n_stars),
            "peak": rng.uniform(1000, 40000, n_stars),
            "a": rng.normal(1.2, 0.1, n_stars), "b": rng.normal(1.15, 0.1, n_stars),
            "theta": rng.uniform(-1.5, 1.5, n_stars),
            "fwhm": rng.normal(hfd * 0.95, 0.12, n_stars),
        }
    )
    if not n_hot:
        return real
    hot = pd.DataFrame(
        {
            "x": rng.uniform(0, 9600, n_hot), "y": rng.uniform(0, 6422, n_hot),
            "hfd": rng.normal(hot_hfd, 0.05, n_hot),
            "snr": rng.uniform(30, 300, n_hot),
            "flux": rng.lognormal(9.0, 0.6, n_hot),
            "peak": rng.uniform(2000, 30000, n_hot),
            "a": rng.normal(0.65, 0.05, n_hot), "b": rng.normal(0.6, 0.05, n_hot),
            "theta": rng.uniform(-1.5, 1.5, n_hot),
            "fwhm": rng.normal(hot_hfd, 0.05, n_hot),
        }
    )
    return pd.concat([real, hot], ignore_index=True)


def test_clean_field_is_left_alone():
    s = summarise_stars(field(), ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["frac_rej_toosmall"] < 0.05
    assert s["hfd_median"] == pytest.approx(2.8, abs=0.08)


def test_hot_pixels_are_removed_and_the_median_recovers():
    """Contaminants outnumbering the stars must not capture the median."""
    contaminated = field(n_stars=400, n_hot=900, seed=1)
    raw = float(np.median(contaminated["hfd"]))
    assert raw < 1.5, "fixture should be dominated by the contaminants"
    s = summarise_stars(contaminated, ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["frac_rej_toosmall"] > 0.5
    assert s["hfd_median"] == pytest.approx(2.8, abs=0.1)
    assert s["n_stars"] == pytest.approx(400, rel=0.1)


def test_saturation_metric_is_not_corrupted_by_contaminants():
    s = summarise_stars(field(n_stars=400, n_hot=900, seed=2), ProfileParams(),
                        naxis1=9600, naxis2=6422)
    assert abs(s["hfd_saturation_rise"]) < 0.25


def test_defocused_frame_is_not_mistaken_for_contamination():
    """A genuinely soft frame must survive: the floor scales with the frame."""
    s = summarise_stars(field(hfd=6.5), ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["frac_rej_toosmall"] < 0.05
    assert s["hfd_median"] == pytest.approx(6.5, abs=0.15)


def test_relative_floor_is_not_applied_to_astap():
    """ASTAP's HFD rises with flux by construction, so faint stars read small.

    Applying the relative floor there was measured to discard ~15% of genuine
    stars from a clean frame, so the floor is restricted to backends whose size
    measurement is flux-independent.
    """
    rng = np.random.default_rng(5)
    n = 600
    flux = rng.lognormal(10, 1.3, n)
    # HFD rising steeply with flux, as ASTAP measures it: on real data the
    # faintest flux quartile reads ~3.1 px against ~5.3 px for the brightest.
    hfd = 2.6 + 5.0 * (np.log10(flux) - np.log10(flux).min()) / np.ptp(np.log10(flux))
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 9600, n), "y": rng.uniform(0, 6422, n),
            "hfd": hfd, "snr": rng.uniform(50, 800, n), "flux": flux,
            "peak": rng.uniform(1000, 40000, n), "a": np.nan, "b": np.nan,
            "theta": np.nan, "fwhm": np.nan,
        }
    )
    astap = summarise_stars(df, ProfileParams(), naxis1=9600, naxis2=6422,
                            backend="astap")
    internal = summarise_stars(df, ProfileParams(), naxis1=9600, naxis2=6422,
                               backend="internal")
    assert astap["frac_rej_toosmall"] == 0.0
    assert internal["frac_rej_toosmall"] > 0.0


def test_absolute_floor_still_applies_to_every_backend():
    df = field(n_stars=200, n_hot=200, hot_hfd=0.4, seed=7)
    s = summarise_stars(df, ProfileParams(), naxis1=9600, naxis2=6422,
                        backend="astap")
    assert s["n_rej_toosmall"] >= 190, "sub-pixel detections are never stars"


def test_psf_scale_is_reported():
    s = summarise_stars(field(hfd=3.4), ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["psf_scale_px"] == pytest.approx(3.4, abs=0.2)
    assert s["hfd_floor_px"] < s["psf_scale_px"]
