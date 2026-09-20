"""Quality control screening."""

import numpy as np
import pandas as pd

from filteroffset.qc import check_blocks, check_frames


def frames_and_profiles(n=12, backend="internal"):
    fr = pd.DataFrame(
        {
            "path": [f"/d/{i}.fit" for i in range(n)],
            "file": [f"{i}.fit" for i in range(n)],
            "date_obs": pd.date_range("2026-09-16", periods=n, freq="10min", tz="UTC"),
            "filter": ["B", "R", "G"] * (n // 3),
            "block": np.repeat(np.arange(n // 3), 3),
            "index_in_block": np.tile([0, 1, 2], n // 3),
            "session": 0,
            "focus_pos": 2000.0,
            "af_likely": True,
            "temp_comp_active": False,
            "n_frames": 3,
            "ambient_temp_range": 0.2,
        }
    )
    prof = pd.DataFrame(
        {
            "path": fr["path"],
            "backend": backend,
            "ok": True,
            "error": None,
            "n_stars": 1500,
            "hfd_median_bright": 2.4,
            "hfd_median": 2.4,
            "ecc_median": 0.35,
            "theta_concentration": 0.2,
            "background": 2500.0,
            "hfd_saturation_rise": 0.0,
        }
    )
    return fr, prof


def test_clean_data_passes():
    fr, prof = frames_and_profiles()
    out, flags = check_frames(fr, prof)
    assert out["qc_pass"].all()
    assert flags.empty


def test_cloud_loss_is_caught_via_star_count():
    fr, prof = frames_and_profiles()
    prof.loc[4, "n_stars"] = 300  # 20% of the nightly median
    out, flags = check_frames(fr, prof)
    assert not out.loc[4, "qc_pass"]
    assert (flags["check"] == "star_count_low").any()


def test_absolute_star_floor():
    fr, prof = frames_and_profiles()
    prof.loc[2, "n_stars"] = 5
    out, flags = check_frames(fr, prof)
    assert not out.loc[2, "qc_pass"]
    assert (flags["check"] == "too_few_stars").any()


def test_trailed_stars_are_caught_and_attributed_to_tracking():
    fr, prof = frames_and_profiles()
    prof.loc[7, "ecc_median"] = 0.80
    prof.loc[7, "theta_concentration"] = 0.9
    out, flags = check_frames(fr, prof)
    assert not out.loc[7, "qc_pass"]
    row = flags[flags["check"] == "elongated_stars"].iloc[0]
    assert "common elongation direction" in row["detail"]


def test_star_size_outlier_is_relative_to_its_own_filter():
    """An absolute size threshold would wrongly fail a soft-but-normal filter."""
    fr, prof = frames_and_profiles()
    prof["hfd_median_bright"] = 2.4
    prof.loc[fr["filter"] == "R", "hfd_median_bright"] = 4.0  # R uniformly softer
    out, flags = check_frames(fr, prof)
    assert out["qc_pass"].all(), "a uniformly softer filter must not be failed"
    prof.loc[1, "hfd_median_bright"] = 8.0  # one genuinely bad R frame
    out, flags = check_frames(fr, prof)
    assert not out.loc[1, "qc_pass"]


def test_saturation_warning():
    fr, prof = frames_and_profiles()
    prof.loc[3, "hfd_saturation_rise"] = 0.9
    out, flags = check_frames(fr, prof)
    assert (flags["check"] == "saturation_inflating_hfd").any()
    assert out.loc[3, "qc_pass"], "saturation is a warning, not a rejection"


def test_backend_disagreement_uses_normalised_scale():
    """Backends differ in absolute HFD by design; only the pattern must agree."""
    fr, a = frames_and_profiles(backend="astap")
    _, b = frames_and_profiles(backend="internal")
    a["hfd_median_bright"] = 4.0  # ASTAP reads systematically larger
    both = pd.concat([a, b], ignore_index=True)
    out, flags = check_frames(fr, both, primary_backend="internal")
    assert not (flags["check"] == "backend_disagreement").any()
    both.loc[both["backend"] == "astap", "hfd_median_bright"] = [
        4.0, 4.0, 4.0, 4.0, 9.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
    ]
    out, flags = check_frames(fr, both, primary_backend="internal")
    assert (flags["check"] == "backend_disagreement").any()


def test_measurement_failure_is_recorded():
    fr, prof = frames_and_profiles()
    prof.loc[5, "ok"] = False
    prof.loc[5, "error"] = "only 3 usable stars"
    out, flags = check_frames(fr, prof)
    assert (flags["check"] == "measurement_failed").any()


def test_block_flags_for_missing_autofocus_and_compensation():
    fr, prof = frames_and_profiles()
    out, _ = check_frames(fr, prof)
    blocks = pd.DataFrame(
        {
            "block": [0, 1, 2, 3],
            "af_likely": [True, False, True, True],
            "temp_comp_active": [False, False, True, False],
            "dead_time_before_s": [440.0, 16.0, 440.0, 440.0],
            "n_frames": [3, 3, 3, 3],
            "ambient_temp_range": [0.2, 0.2, 0.2, 1.8],
        }
    )
    bq, bf = check_blocks(blocks, out)
    checks = set(bf["check"])
    assert "autofocus_not_detected" in checks
    assert "temperature_compensation_active" in checks
    assert "temperature_drifted_within_block" in checks
    assert not bq.loc[1, "qc_pass"] and not bq.loc[2, "qc_pass"]
    assert bq.loc[0, "qc_pass"] and bq.loc[3, "qc_pass"]


def test_no_profiles_is_tolerated():
    fr, _ = frames_and_profiles()
    out, flags = check_frames(fr, pd.DataFrame())
    assert out["qc_pass"].all()
