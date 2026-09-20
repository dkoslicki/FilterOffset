"""Which offsets the design can actually determine.

The failure these guard against is silent: a rank-deficient design solved by
least squares returns finite, plausible-looking coefficients with small
standard errors that mean nothing at all.
"""

import numpy as np
import pandas as pd
import pytest

from filteroffset.model import (
    ModelSpec,
    _drop_aliased_columns,
    filter_components,
    fit_offsets,
    identifiability,
    within_filter_temp_coeff,
)


def campaign(layout, k=-4.0, offsets=None, noise=0.3, seed=0, blocks_per=4,
             night_jump_sd=8.0, night_temp_spread=1.5):
    """Build a multi-night campaign from {session: [filters]}."""
    offsets = offsets or {"B": 0.0, "R": -5.0, "G": -6.0, "Ha": 18.0, "OIII": 2.0}
    rng = np.random.default_rng(seed)
    rows, block = [], 0
    t0 = pd.Timestamp("2026-07-22T02:00:00Z")
    for sess, filters in sorted(layout.items()):
        # Zero-point movement between nights; 0 models an undisturbed train.
        night_shift = rng.normal(0.0, night_jump_sd) if night_jump_sd else 0.0
        start = t0 + pd.Timedelta(days=3 * sess)
        # Temperature falls through each night and resets the next, as it does
        # in reality. Letting it fall monotonically across the whole campaign
        # instead would make date and night-mean temperature the same variable.
        # Alternating rather than random, so night temperature carries no trend
        # with date; otherwise a drift term and the between-night temperature
        # response describe the same thing and neither is identifiable.
        night_start_temp = 18.0 + night_temp_spread * (1 if sess % 2 == 0 else -1)
        for i in range(blocks_per):
            for f in filters:
                temp = night_start_temp - 0.7 * i + rng.normal(0, 0.1)
                pos = (
                    2000.0 + offsets.get(f, 0.0) + k * (temp - 15.0)
                    + night_shift + rng.normal(0, noise)
                )
                t = start + pd.Timedelta(minutes=40 * (i * len(filters) + filters.index(f)))
                rows.append(
                    {
                        "block": block, "session": sess,
                        "session_label": start.strftime("%Y-%m-%d"),
                        "filter": f, "focus_pos": float(np.round(pos)),
                        "n_frames": 3, "t_start": t, "t_mid": t,
                        "t_end": t + pd.Timedelta(minutes=30),
                        "ambient_temp": temp, "ambient_temp_first": temp,
                        "ambient_temp_range": 0.2, "focus_temp": np.nan,
                        "ccd_temp": -12.0, "altitude": 60.0, "airmass": 1.1,
                        "exptime": 600.0, "dead_time_before_s": 440.0,
                        "af_likely": True, "temp_comp_active": False,
                        "object": "Synth", "instrument": "cam",
                        "telescope": "ota", "qc_pass": True,
                    }
                )
                block += 1
    df = pd.DataFrame(rows)
    df["elapsed_hours"] = (df["t_mid"] - df["t_mid"].min()).dt.total_seconds() / 3600
    df["elapsed_hours_in_session"] = (
        df["t_mid"] - df.groupby("session")["t_mid"].transform("min")
    ).dt.total_seconds() / 3600
    return df


def test_components_of_a_fully_linked_campaign():
    b = campaign({0: ["B", "R", "G"], 1: ["B", "R", "G"]})
    assert filter_components(b) == [["B", "G", "R"]]


def test_components_split_when_a_filter_is_never_shared():
    """The real archive's shape: narrowband nights that share nothing."""
    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["Ha"], 3: ["B", "G", "R", "Ha"]})
    comps = filter_components(b)
    assert comps == [["B", "G", "Ha", "R"], ["OIII"]]


def test_components_link_through_a_chain():
    """A links to B, B links to C -> all three are mutually identifiable."""
    b = campaign({0: ["B", "G"], 1: ["G", "R"], 2: ["R", "Ha"]})
    assert filter_components(b) == [["B", "G", "Ha", "R"]]


def test_unlinked_filter_is_reported_not_invented():
    """With night intercepts in play, an unshared filter has no offset."""
    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["Ha"], 3: ["B", "G", "R", "Ha"]})
    fit = fit_offsets(
        b, ModelSpec(bootstrap=0, reference_filter="B", night_effects="on")
    )
    assert fit.not_identified == ["OIII"]
    assert not np.isfinite(fit.offsets["OIII"])
    assert not np.isfinite(fit.offsets_se["OIII"])
    # ...while the linked filters remain perfectly well determined.
    for f in ("G", "R", "Ha"):
        assert np.isfinite(fit.offsets[f])
        assert fit.offsets_se[f] < 5.0


def test_linked_filters_recover_truth_despite_nightly_zero_point_shifts():
    truth = {"B": 0.0, "R": -5.0, "G": -6.0, "Ha": 18.0, "OIII": 2.0}
    b = campaign(
        {0: ["OIII"], 1: ["OIII"], 2: ["Ha"], 3: ["B", "G", "R", "Ha"],
         4: ["B", "G", "R", "Ha"]},
        offsets=truth, seed=4,
    )
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    for f in ("G", "R", "Ha"):
        assert fit.offsets[f] == pytest.approx(truth[f], abs=1.5), f


def test_pooled_fallback_is_offered_for_unlinked_filters():
    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["Ha"], 3: ["B", "G", "R", "Ha"]})
    fit = fit_offsets(
        b, ModelSpec(bootstrap=0, reference_filter="B", night_effects="on")
    )
    assert "OIII" in fit.offsets_pooled
    assert np.isfinite(fit.offsets_pooled["OIII"])


def test_reference_is_chosen_from_the_largest_component():
    """Picking an orphan as reference would make every offset unidentifiable."""
    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["OIII"], 3: ["B", "G", "R"]})
    fit = fit_offsets(b, ModelSpec(bootstrap=0, night_effects="on"))
    assert fit.reference in {"B", "G", "R"}
    assert fit.not_identified == ["OIII"]


def test_no_vif_is_reported_as_a_huge_finite_number():
    """An aliased term must read as infinite, never as 1e12."""
    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["Ha"], 3: ["B", "G", "R", "Ha"]})
    fit = fit_offsets(
        b, ModelSpec(bootstrap=0, reference_filter="B", night_effects="on")
    )
    for name, v in fit.diagnostics["vif"].items():
        assert v < 1e6 or not np.isfinite(v), f"{name} reported as {v}"


def test_single_night_campaign_identifies_everything():
    b = campaign({0: ["B", "G", "R"]})
    info = identifiability(b, use_night=False, reference="B")
    assert info["not_identified"] == []


def test_stable_focuser_keeps_every_offset_measurable():
    """The point of trusting absolute positions: the orphan becomes measurable.

    With an undisturbed imaging train the between-night information is valid,
    and it is exactly that information which ties a filter used alone on its
    own nights to the rest.
    """
    truth = {"B": 0.0, "R": -5.0, "G": -6.0, "Ha": 18.0, "OIII": 2.0}
    b = campaign(
        {0: ["OIII"], 1: ["OIII"], 2: ["OIII"], 3: ["Ha"], 4: ["B", "G", "R", "Ha"]},
        offsets=truth, night_jump_sd=0.0, noise=0.4, seed=8,
    )
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    assert fit.not_identified == []
    assert fit.drift_test.verdict in ("stable", "drift")
    for f, want in truth.items():
        assert fit.offsets[f] == pytest.approx(want, abs=2.0), f


def test_real_zero_point_jumps_are_detected_and_protected_against():
    """When nights really do jump, the conservative model must come back."""
    b = campaign(
        {0: ["OIII"], 1: ["OIII"], 2: ["OIII"], 3: ["Ha"], 4: ["B", "G", "R", "Ha"]},
        night_jump_sd=25.0, noise=0.4, seed=9,
    )
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    assert fit.drift_test.verdict == "jumps"
    assert fit.diagnostics["night_effects"] is True
    assert fit.not_identified == ["OIII"]


def test_drift_test_separates_a_trend_from_jumps():
    """A smooth secular drift must not be mistaken for nights jumping."""
    from filteroffset.model import zero_point_drift_test

    b = campaign({i: ["OIII"] for i in range(7)}, night_jump_sd=0.0,
                 noise=0.4, seed=12).copy()
    days = (b["t_mid"] - b["t_mid"].min()).dt.total_seconds() / 86400.0
    b["focus_pos"] = b["focus_pos"] + 0.2 * days       # slow settling
    d = zero_point_drift_test(b, ModelSpec())
    assert d.tested and d.verdict == "drift"
    assert d.drift_per_day == pytest.approx(0.2, abs=0.08)
    assert d.between_sd_detrended < d.between_sd


def test_drift_term_keeps_offsets_identifiable():
    # Enough nights for a slow trend to be distinguishable from night noise.
    # Every night follows the same temperature curve, and the anchor filters
    # appear throughout the campaign rather than clustering at one end - if a
    # filter occupied only the late nights, its offset and the drift would be
    # the same thing.
    layout = {i: ["OIII", "Ha"] for i in range(6)}
    layout[6] = ["B", "G", "R", "Ha"]
    b = campaign(layout, night_jump_sd=0.0, noise=0.4, seed=13,
                 night_temp_spread=0.0)
    days = (b["t_mid"] - b["t_mid"].min()).dt.total_seconds() / 86400.0
    b = b.copy()
    b["focus_pos"] = b["focus_pos"] + 0.25 * days
    fit = fit_offsets(b, ModelSpec(bootstrap=0, reference_filter="B"))
    assert fit.not_identified == []
    assert np.isfinite(fit.drift_per_day)
    assert fit.drift_per_day == pytest.approx(0.25, abs=0.12)


def test_within_night_coefficient_is_immune_to_night_levels():
    """Between-night noise must not reach the temperature coefficient."""
    from filteroffset.model import estimate_k_within_night

    truth_k = -4.0
    for jump in (0.0, 30.0):
        b = campaign({i: ["OIII"] for i in range(6)}, k=truth_k,
                     night_jump_sd=jump, noise=0.3, seed=21)
        k, se, n = estimate_k_within_night(
            b, b["ambient_temp_first"].to_numpy(float),
            b["focus_pos"].to_numpy(float),
        )
        assert k == pytest.approx(truth_k, abs=0.4), f"jump sd {jump}"


def test_informative_night_count():
    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["Ha"], 3: ["B", "G", "R", "Ha"]})
    info = identifiability(b, use_night=True, reference="B")
    assert info["n_informative_nights"] == 1


def test_drop_aliased_columns_keeps_protected_terms_first():
    x = np.linspace(0, 1, 30)
    dup = x.copy()
    X = np.column_stack([np.ones(30), x, dup])
    Xk, names, dropped = _drop_aliased_columns(X, ["intercept", "temp", "filter[R]"])
    assert dropped == ["filter[R]"]
    assert names == ["intercept", "temp"]
    assert Xk.shape[1] == 2


def test_within_filter_check_conditions_on_night():
    """Without night conditioning, zero-point shifts masquerade as temperature."""
    truth_k = -4.0
    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["OIII"], 3: ["OIII"]},
                 k=truth_k, noise=0.2, seed=11)
    temp = b["ambient_temp_first"].to_numpy(float)
    naive = within_filter_temp_coeff(b, temp, by_night=False)
    conditioned = within_filter_temp_coeff(b, temp, by_night=True)
    assert conditioned["k_pooled"] == pytest.approx(truth_k, abs=0.5)
    # The night-conditioned estimate must be at least as close to the truth.
    assert abs(conditioned["k_pooled"] - truth_k) <= abs(naive["k_pooled"] - truth_k) + 1e-6


def test_mixed_targets_are_flagged_when_comparing_nights():
    """Pointing-dependent flexure confounds between-night comparisons."""
    from filteroffset.recommend import build_recommendations

    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["OIII"], 3: ["B", "G", "R"]},
                 night_jump_sd=0.0, noise=0.3, seed=31)
    b = b.copy()
    b.loc[b["session"] == 3, "object"] = "Another Target"
    fit = fit_offsets(
        b, ModelSpec(bootstrap=0, reference_filter="B", night_effects="off")
    )
    rec = build_recommendations(
        fit, b.assign(qc_pass=True), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert "mixed_targets" in {f.code for f in rec.findings}


def test_single_target_is_not_flagged():
    from filteroffset.recommend import build_recommendations

    b = campaign({0: ["OIII"], 1: ["OIII"], 2: ["OIII"], 3: ["B", "G", "R"]},
                 night_jump_sd=0.0, noise=0.3, seed=32)
    fit = fit_offsets(
        b, ModelSpec(bootstrap=0, reference_filter="B", night_effects="off")
    )
    rec = build_recommendations(
        fit, b.assign(qc_pass=True), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert "mixed_targets" not in {f.code for f in rec.findings}
