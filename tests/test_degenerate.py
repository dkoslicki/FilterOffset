"""Behaviour on awkward, short and sensor-starved datasets.

These are the cases a tool meets in the wild rather than in development: a
probe that never reported, a cooled camera whose only temperature is its own
set point, a focuser with half a million fine steps, a session of two blocks.
The requirement is that the tool degrade to a smaller honest answer rather than
produce a confident wrong one.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_blocks

from filteroffset.model import (
    ModelSpec,
    _huber,
    _wls,
    assess_temperature_sources,
    fit_offsets,
)


# ------------------------------------------------- temperature sources -----
def test_dead_probe_reporting_zero_is_rejected():
    b = make_blocks(n_cycles=4)
    b["ambient_temp"] = 0.0
    b["ambient_temp_first"] = 0.0
    col, why = assess_temperature_sources(b)
    assert col != "ambient_temp"
    assert "constant" in why["ambient_temp"] and "not reporting" in why["ambient_temp"]


def test_regulated_ccd_temperature_is_rejected():
    """A cooled sensor held at its set point is cooler noise, not weather."""
    b = make_blocks(n_cycles=4)
    b["ambient_temp"] = 0.0
    b["ambient_temp_first"] = 0.0
    b["ccd_temp"] = -7.0 + np.linspace(-0.2, 0.2, len(b))
    b["set_temp"] = -7.0
    col, why = assess_temperature_sources(b)
    assert col is None
    assert "regulated" in why["ccd_temp"]


def test_unregulated_ccd_temperature_is_allowed():
    """On an uncooled camera the sensor does follow the environment."""
    b = make_blocks(n_cycles=4)
    b["ambient_temp"] = 0.0
    b["ambient_temp_first"] = 0.0
    b["ccd_temp"] = np.linspace(18.0, 8.0, len(b))
    b["set_temp"] = np.nan
    col, _ = assess_temperature_sources(b)
    assert col == "ccd_temp"


def test_real_probe_is_preferred_over_ccd():
    b = make_blocks(n_cycles=4)
    b["ccd_temp"] = np.linspace(20.0, 0.0, len(b))
    b["set_temp"] = np.nan
    assert assess_temperature_sources(b)[0] == "ambient_temp"


# --------------------------------------------- fitting without temperature --
def test_offsets_are_still_fitted_without_any_temperature():
    """Offsets are differences between filters; they do not need temperature."""
    truth = {"B": 0.0, "R": -6.0, "G": -5.5}
    b = make_blocks(n_cycles=5, offsets=truth, k=0.0, noise=0.4, seed=3)
    b["ambient_temp"] = 0.0
    b["ambient_temp_first"] = 0.0
    b["ccd_temp"] = -7.0
    b["set_temp"] = -7.0
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    assert not np.isfinite(fit.temp_coeff), "a coefficient must not be invented"
    assert fit.diagnostics["has_temperature"] is False
    for f, want in truth.items():
        assert fit.offsets[f] == pytest.approx(want, abs=1.0), f
    assert np.isfinite(fit.offsets_se["R"]) and fit.offsets_se["R"] > 0


def test_no_temperature_produces_a_finding_and_no_compensation_advice():
    from filteroffset.apt import instructions
    from filteroffset.recommend import build_recommendations

    b = make_blocks(n_cycles=5, k=0.0, noise=0.3, seed=4)
    b["ambient_temp"] = 0.0
    b["ambient_temp_first"] = 0.0
    b["ccd_temp"] = -7.0
    b["set_temp"] = -7.0
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    rec = build_recommendations(
        fit, b.assign(qc_pass=True), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert "no_temperature_source" in {f.code for f in rec.findings}
    # ...and none of the temperature-dependent findings fire.
    codes = {f.code for f in rec.findings}
    for absent in ("narrow_temperature_range", "thermal_lag", "shared_temp_slope",
                   "filter_temp_separated", "temp_coeff_imprecise"):
        assert absent not in codes, absent
    text = " ".join(instructions(fit))
    assert "leave it off" in text and "nan" not in text.lower()


def test_no_temperature_leaves_no_nan_in_the_json():
    from filteroffset.apt import to_json

    b = make_blocks(n_cycles=5, k=0.0, noise=0.3, seed=5)
    b["ambient_temp"] = 0.0
    b["ambient_temp_first"] = 0.0
    b["ccd_temp"] = -7.0
    b["set_temp"] = -7.0
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    payload = to_json(fit)
    assert "NaN" not in payload
    import json

    assert json.loads(payload)["temperature_coefficient_steps_per_C"] is None


# ------------------------------------------------- robust scale collapse ----
def test_huber_does_not_collapse_on_quantised_positions():
    """Integer positions repeat, so half the residuals coincide and MAD -> 0.

    Left unguarded, Huber judges every non-zero residual against a scale of
    nothing, weights whole blocks out, and reports zero scatter with zero-width
    intervals.
    """
    filt = np.array(["L"] * 4 + ["B"] * 4 + ["G"] * 3 + ["R"] * 2)
    y = np.array(
        [499953.0, 499954, 499954, 499955,
         499929, 499929, 499929, 499930,
         499929, 499929, 499930,
         499929, 499929]
    )
    X = np.column_stack(
        [np.ones(len(y))] + [(filt == f).astype(float) for f in ("B", "G", "R")]
    )
    beta_ols, cov_ols, _, s2_ols, _ = _wls(X, y)
    beta, cov, resid, s2, dof, w, scale = _huber(X, y)
    assert scale > 0.0, "robust scale must not collapse to zero"
    assert np.sqrt(s2) > 0.1, "reported scatter must not be zero"
    assert (w > 0).all(), "no block should be weighted entirely out"
    # ...and the answer should stay close to least squares on clean data.
    assert beta[1] == pytest.approx(beta_ols[1], abs=0.5)


def test_huber_still_resists_a_real_outlier():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    y = 3.0 + 2.0 * x + rng.normal(0, 0.5, x.size)
    y[7] += 60.0
    X = np.column_stack([np.ones(x.size), x])
    beta_ols, *_ = _wls(X, y)
    beta_rob, *_ = _huber(X, y)
    assert abs(beta_rob[1] - 2.0) < abs(beta_ols[1] - 2.0)


# ----------------------------------------------------- very small datasets --
def test_two_blocks_one_filter_fails_clearly():
    b = make_blocks(n_cycles=1, filters=("B",))
    with pytest.raises(ValueError, match="cannot identify|blocks"):
        fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))


def test_constant_focus_position_gives_zero_offsets_without_crashing():
    b = make_blocks(n_cycles=4, noise=0.0)
    b["focus_pos"] = 2000.0
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    for f in ("R", "G"):
        assert fit.offsets[f] == pytest.approx(0.0, abs=1e-6)


def test_large_focuser_step_counts_are_handled():
    """A fine-step focuser reports ~500000; nothing may overflow or round."""
    truth = {"B": 0.0, "R": -25.0, "G": -24.0}
    b = make_blocks(n_cycles=5, offsets=truth, k=0.0, noise=0.4, seed=6)
    b["focus_pos"] = b["focus_pos"] + 498000.0
    b["ambient_temp"] = 0.0
    b["ambient_temp_first"] = 0.0
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    for f, want in truth.items():
        assert fit.offsets[f] == pytest.approx(want, abs=1.0), f


# ------------------------------------- the estimator must not invent a gap --
def _campaign(k_true=-4.0, n_nights=5, cycles=4, dT=-0.5, noise=0.4, seed=0,
              constant_within_night=False):
    """Nights with ONE true temperature response and no between-night confounder."""
    rng = np.random.default_rng(seed)
    rows, blk = [], 0
    t0 = pd.Timestamp("2026-03-01T02:00:00Z")
    for n in range(n_nights):
        start = t0 + pd.Timedelta(days=3 * n)
        base = 15.0 + rng.uniform(-4, 4)
        for c in range(cycles):
            for f in ("B", "R", "G"):
                temp = base if constant_within_night else base + dT * c
                pos = (
                    2000.0 + {"B": 0.0, "R": -6.0, "G": -5.5}[f]
                    + k_true * (temp - 15.0) + rng.normal(0, noise)
                )
                tm = start + pd.Timedelta(minutes=40 * (c * 3 + ("B", "R", "G").index(f)))
                rows.append(
                    {
                        "block": blk, "session": n, "session_label": str(start.date()),
                        "filter": f, "focus_pos": round(pos, 0), "n_frames": 3,
                        "t_start": tm, "t_mid": tm, "t_end": tm,
                        "ambient_temp": temp, "ambient_temp_first": temp,
                        "ambient_temp_range": 0.2, "focus_temp": np.nan,
                        "ccd_temp": -10.0, "set_temp": np.nan, "altitude": 60.0,
                        "airmass": 1.1, "exptime": 300.0,
                        "dead_time_before_s": 440.0, "af_likely": True,
                        "temp_comp_active": False, "object": "S",
                        "instrument": "c", "telescope": "t", "qc_pass": True,
                    }
                )
                blk += 1
    d = pd.DataFrame(rows)
    d["elapsed_hours"] = (d.t_mid - d.t_mid.min()).dt.total_seconds() / 3600
    d["elapsed_hours_in_session"] = (
        d.t_mid - d.groupby("session").t_mid.transform("min")
    ).dt.total_seconds() / 3600
    return d


@pytest.mark.parametrize("dT", [-0.5, -0.25, -0.10])
def test_single_true_coefficient_is_returned_not_inflated(dT):
    """The complement to the decomposition test: one true k must come back as k.

    Without this, an estimator that manufactures a within/between gap from data
    that has none looks like it is discovering physics. Small within-night
    temperature ranges are where that failure appears.
    """
    ks = [
        fit_offsets(
            _campaign(dT=dT, seed=s), ModelSpec(bootstrap=0, reference_filter="B")
        ).temp_coeff
        for s in range(8)
    ]
    assert np.mean(ks) == pytest.approx(-4.0, abs=0.5), np.mean(ks)
    assert np.std(ks) < 0.6, f"scatter {np.std(ks):.3f} too large at dT={dT}"


def test_temperature_logged_once_per_night_falls_back_to_a_single_slope():
    """A within-night regressor of numerical dust must never be fitted.

    Rigs that write one temperature per sequence make the within-night
    regressor identically zero; a slope fitted on the residual relaxation
    artefact came back wrong by five orders of magnitude.
    """
    b = _campaign(k_true=-3.0, constant_within_night=True, n_nights=4, cycles=6)
    fit = fit_offsets(
        b, ModelSpec(bootstrap=0, reference_filter="B", thermal_lag_minutes="fit")
    )
    assert fit.temp_coeff == pytest.approx(-3.0, abs=0.5), fit.temp_coeff
    assert fit.temp_from == "pooled"
    assert any("within nights" in n for n in fit.notes)


def test_thermal_lag_is_not_fitted_by_default():
    """Searching the lag by default biased |k| and its error ignored the search."""
    assert ModelSpec().thermal_lag_minutes is None


def test_one_unparseable_timestamp_does_not_crash_the_fit():
    b = _campaign(seed=2)
    b.loc[7, "t_mid"] = pd.NaT
    fit = fit_offsets(
        b, ModelSpec(bootstrap=0, reference_filter="B", thermal_lag_minutes=60.0)
    )
    assert np.isfinite(fit.temp_coeff)
    assert all(np.isfinite(v) for v in fit.offsets.values())


def test_huber_standard_errors_are_not_badly_understated():
    """Weighted-least-squares covariance treats estimated weights as known."""
    from scipy import stats

    rng = np.random.default_rng(1)
    n, p, trials = 40, 4, 400
    est, se = [], []
    for _ in range(trials):
        X = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(p - 1)])
        beta = np.array([5.0, 2.0, -1.0, 0.5])
        y = X @ beta + rng.normal(0, 1.0, n)
        b, cov, *_ = _huber(X, y)
        est.append(b[1])
        se.append(np.sqrt(cov[1, 1]))
    est, se = np.array(est), np.array(se)
    assert se.mean() / est.std() > 0.90, se.mean() / est.std()
    crit = stats.t.ppf(0.975, n - p)
    coverage = float(np.mean(np.abs(est - 2.0) < crit * se))
    assert coverage > 0.90, coverage
