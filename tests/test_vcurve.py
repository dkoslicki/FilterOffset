"""Defocus geometry, V-curve fitting and sweep detection."""

import numpy as np
import pandas as pd
import pytest

from filteroffset.vcurve import (
    detect_sweeps,
    fit_vcurve,
    focus_tolerances,
    geometric_slope_px_per_micron,
    hyperbola_hfd,
    implied_defocus_from_excess,
    step_size_microns_from_slope,
    steps_to_microns_table,
)


def test_vcurve_recovers_best_focus_and_slope():
    truth_best, truth_min, truth_slope = 2000.0, 2.2, 0.11
    pos = np.arange(1940.0, 2061.0, 10.0)
    hfd = hyperbola_hfd(pos, truth_best, truth_min, truth_slope)
    fit = fit_vcurve(pos, hfd, "Ha", focal_ratio=3.3125, pixel_um=3.76)
    assert fit.identifiable
    assert fit.best_position == pytest.approx(truth_best, abs=1.0)
    assert fit.slope_px_per_step == pytest.approx(truth_slope, rel=0.1)
    assert fit.hfd_min == pytest.approx(truth_min, rel=0.1)


def test_vcurve_recovers_best_focus_with_noise():
    rng = np.random.default_rng(3)
    pos = np.arange(1950.0, 2051.0, 5.0)
    hfd = hyperbola_hfd(pos, 2003.0, 2.3, 0.10) + rng.normal(0, 0.03, pos.size)
    fit = fit_vcurve(pos, hfd, "Ha")
    assert fit.best_position == pytest.approx(2003.0, abs=2.5)


def test_flat_bottom_is_reported_as_unidentifiable():
    """A sweep too narrow to show curvature must not claim a best focus."""
    pos = np.arange(1998.0, 2003.0, 1.0)
    hfd = hyperbola_hfd(pos, 2000.0, 2.2, 0.11)
    fit = fit_vcurve(pos, hfd, "Ha")
    assert not fit.identifiable
    assert any("flat" in n or "Widen" in n for n in fit.notes)


def test_step_size_round_trip():
    """A slope generated from a known step size must invert back to it."""
    fr, px, um_per_step = 3.3125, 3.76, 2.63
    slope = geometric_slope_px_per_micron(fr, px) * um_per_step
    assert step_size_microns_from_slope(slope, fr, px) == pytest.approx(um_per_step)


def test_focus_tolerances_scale_as_the_square_of_focal_ratio():
    a = focus_tolerances(3.0)["diffraction_total_um"]
    b = focus_tolerances(6.0)["diffraction_total_um"]
    assert b / a == pytest.approx(4.0)


def test_implied_defocus_inverts_the_quadrature_sum():
    h0, slope, e = 2.24, 0.1135, 7.5
    hfd = np.sqrt(h0**2 + (slope * e) ** 2)
    excess = hfd / h0 - 1.0
    got = implied_defocus_from_excess(excess, h0, slope)
    assert got["implied_defocus_steps"] == pytest.approx(e, rel=0.02)


def test_zero_excess_implies_zero_defocus():
    got = implied_defocus_from_excess(0.0, 2.2, 0.11)
    assert got["implied_defocus_steps"] == 0.0


def test_steps_to_microns_table_marks_significance():
    tbl = steps_to_microns_table({"R": -6.5}, tolerance_um=18.6)
    assert {"um_per_step", "R_um", "fraction_of_tolerance", "matters"} <= set(tbl.columns)
    small = tbl[tbl["um_per_step"] == 0.5].iloc[0]
    big = tbl[tbl["um_per_step"] == 5.0].iloc[0]
    assert not bool(small["matters"]) and bool(big["matters"])


def _sweep_frames(positions, gap_s=90, exptime=30, filt="Ha"):
    t0 = pd.Timestamp("2026-09-20T02:00:00Z")
    return pd.DataFrame(
        {
            "path": [f"/d/{i}.fit" for i in range(len(positions))],
            "focus_pos": [float(p) for p in positions],
            "date_obs": [t0 + pd.Timedelta(seconds=i * gap_s) for i in range(len(positions))],
            "exptime": exptime,
            "filter": filt,
            "session": 0,
        }
    )


def test_detect_sweeps_finds_a_real_sweep():
    got = detect_sweeps(_sweep_frames([1980, 1990, 2000, 2010, 2020, 2030]))
    assert len(got) == 1
    assert got.iloc[0]["n_positions"] == 6


def test_detect_sweeps_rejects_separate_autofocus_runs():
    """The false positive that matters: one position per AF run across a night.

    Fitting a V-curve to these recovers the night's seeing trend, not focus, so
    they must not be mistaken for a sweep.
    """
    got = detect_sweeps(
        _sweep_frames([1996, 2001, 2003, 2012, 2014], gap_s=2200, exptime=600)
    )
    assert got.empty


def test_detect_sweeps_requires_enough_positions():
    assert detect_sweeps(_sweep_frames([2000, 2010])).empty


def test_detect_sweeps_handles_missing_columns():
    assert detect_sweeps(pd.DataFrame()).empty
    assert detect_sweeps(pd.DataFrame({"focus_pos": [1, 2, 3, 4]})).empty
