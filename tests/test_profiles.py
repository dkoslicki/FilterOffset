"""Shared star-table summarisation, used identically by both backends."""

import numpy as np
import pandas as pd
import pytest

from filteroffset.profiles import ProfileParams, summarise_stars
from filteroffset.profiles.astap import read_astap_csv


def star_table(n=400, hfd=3.0, seed=0, flux_scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 9600, n),
            "y": rng.uniform(0, 6422, n),
            "hfd": rng.normal(hfd, 0.2, n),
            "snr": rng.uniform(20, 500, n),
            "flux": rng.lognormal(10, 1.2, n) * flux_scale,
            "peak": rng.uniform(1000, 40000, n),
            "a": rng.normal(1.2, 0.1, n),
            "b": rng.normal(1.1, 0.1, n),
            "theta": rng.uniform(-np.pi / 2, np.pi / 2, n),
            "fwhm": rng.normal(hfd * 0.9, 0.2, n),
        }
    )


def test_basic_statistics():
    s = summarise_stars(star_table(), ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["n_stars"] == 400
    assert s["hfd_median"] == pytest.approx(3.0, abs=0.06)
    assert s["n_bright"] > 0
    assert np.isfinite(s["hfd_median_center"])
    assert 0.0 <= s["ecc_median"] <= 1.0


def test_bright_sample_is_flux_ranked_not_snr_thresholded():
    """ASTAP and sep report SNR on different scales, so ranking must be used.

    Scaling every flux by a constant must not change which stars are 'bright'.
    """
    a = summarise_stars(star_table(seed=1), ProfileParams(), naxis1=9600, naxis2=6422)
    b = summarise_stars(
        star_table(seed=1, flux_scale=1000.0), ProfileParams(), naxis1=9600, naxis2=6422
    )
    assert a["n_bright"] == b["n_bright"]
    assert a["hfd_median_bright"] == pytest.approx(b["hfd_median_bright"])


def test_saturated_stars_are_rejected():
    t = star_table()
    t.loc[:19, "peak"] = 65000.0
    s = summarise_stars(t, ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["n_rej_saturated"] == 20


def test_faint_end_bias_is_detected():
    """A background-biased frame inflates faint stars only."""
    t = star_table(n=600, seed=2)
    faint = t["flux"] < t["flux"].quantile(0.25)
    t.loc[faint, "hfd"] += 0.8
    s = summarise_stars(t, ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["hfd_faint_minus_bright"] > 0.5


def test_saturation_rise_is_detected():
    t = star_table(n=600, seed=3)
    top = t["flux"] > t["flux"].quantile(0.75)
    t.loc[top, "hfd"] += 0.9
    s = summarise_stars(t, ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["hfd_saturation_rise"] > 0.5


def test_empty_table_returns_finite_schema():
    s = summarise_stars(pd.DataFrame(), ProfileParams())
    assert s["n_stars"] == 0
    assert s["n_bright"] == 0
    for key in ("hfd_median", "hfd_median_bright", "ecc_median"):
        assert key in s


def test_shape_uses_bright_stars_only():
    """Noisy faint moments must not drive the eccentricity."""
    t = star_table(n=600, seed=4)
    faint = t["flux"] < t["flux"].quantile(0.5)
    t.loc[faint, "b"] = 0.2  # wildly elongated noise blobs
    s = summarise_stars(t, ProfileParams(), naxis1=9600, naxis2=6422)
    assert s["ecc_median"] < 0.5


def test_astap_csv_parsing(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text(
        "x,y,hfd,snr,flux,ra[0..360],dec[0..360]\n"
        "878.31,21.37,8.6289,558,5814445\n"
        "3134.95,19.53,5.8026,1086,2337415\n"
        "bad,row,here\n"
        "\n"
        "6017.85,38.94,4.0286,591,597990\n"
    )
    df = read_astap_csv(str(p))
    assert len(df) == 3, "short and malformed rows must be skipped, not crash"
    assert df["hfd"].iloc[0] == pytest.approx(8.6289)
    assert {"x", "y", "hfd", "snr", "flux", "peak", "a", "b"} <= set(df.columns)


def test_astap_csv_empty(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("")
    assert read_astap_csv(str(p)).empty
