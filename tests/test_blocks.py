"""Recovery of the autofocus structure from frame timestamps."""

import numpy as np
import pandas as pd

from filteroffset.blocks import BlockParams, segment_blocks, summarise_blocks


def build_frames(
    pattern=(("B", 1996), ("R", 1991), ("G", 1992), ("B", 2001)),
    n_per=3,
    exptime=600.0,
    af_gap=440.0,
    in_gap=16.0,
    start="2026-09-16T01:30:47Z",
):
    """Build a frame table imitating: change filter, autofocus, take N subs."""
    rows = []
    t = pd.Timestamp(start)
    first = True
    for filt, pos in pattern:
        for i in range(n_per):
            if not first:
                t = t + pd.Timedelta(seconds=exptime + (in_gap if i else af_gap))
            first = False
            rows.append(
                {
                    "path": f"/d/{filt}_{pos}_{i}.fit",
                    "file": f"{filt}_{pos}_{i}.fit",
                    "date_obs": t,
                    "filter": filt,
                    "focus_pos": float(pos),
                    "exptime": exptime,
                    "ambient_temp": 16.0,
                    "instrument": "cam",
                    "telescope": "ota",
                    "xbinning": 1.0,
                    "ybinning": 1.0,
                    "gain": 26.0,
                    "offset": 100.0,
                    "altitude": 70.0,
                    "airmass": 1.06,
                }
            )
    return pd.DataFrame(rows)


def test_blocks_follow_filter_and_position_changes():
    fr = segment_blocks(build_frames())
    assert fr["block"].nunique() == 4
    assert list(fr["block"]) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert list(fr["index_in_block"]) == [0, 1, 2] * 4
    assert fr["is_first_after_af"].sum() == 4


def test_autofocus_is_detected_from_dead_time():
    fr = segment_blocks(build_frames())
    firsts = fr[fr["index_in_block"] == 0]
    # The very first block of a session has no measurable preceding gap.
    assert pd.isna(firsts.iloc[0]["af_likely"])
    assert all(firsts.iloc[1:]["af_likely"] == True)  # noqa: E712


def test_missing_autofocus_is_flagged():
    """A filter change with no pause cannot have had an autofocus run."""
    fr = segment_blocks(build_frames(af_gap=16.0))
    firsts = fr[fr["index_in_block"] == 0]
    assert any(firsts.iloc[1:]["af_likely"] == False)  # noqa: E712


def test_same_position_after_long_gap_starts_a_new_block():
    """Autofocus can legitimately return the same position; time reveals it."""
    fr = segment_blocks(build_frames(pattern=(("B", 1996), ("B", 1996)), n_per=3))
    assert fr["block"].nunique() == 2


def test_temperature_compensation_is_detected():
    """A focuser creeping inside one filter run means compensation was on."""
    fr = build_frames(pattern=(("B", 1996),), n_per=4)
    fr.loc[1, "focus_pos"] = 1997.0
    fr.loc[2, "focus_pos"] = 1998.0
    seg = segment_blocks(fr)
    assert bool(seg["temp_comp_active"].any())


def test_sessions_split_on_a_long_gap():
    a = build_frames(pattern=(("B", 1996),), n_per=3)
    b = build_frames(pattern=(("B", 1999),), n_per=3, start="2026-09-17T01:30:47Z")
    seg = segment_blocks(pd.concat([a, b], ignore_index=True))
    assert seg["session"].nunique() == 2


def test_summarise_blocks_aggregates_correctly():
    fr = segment_blocks(build_frames())
    bl = summarise_blocks(fr)
    assert len(bl) == 4
    assert list(bl["filter"]) == ["B", "R", "G", "B"]
    assert list(bl["n_frames"]) == [3, 3, 3, 3]
    assert bl["elapsed_hours"].iloc[0] == 0.0
    assert bl["elapsed_hours"].is_monotonic_increasing


def test_config_change_splits_blocks():
    fr = build_frames(pattern=(("B", 1996),), n_per=4)
    fr.loc[2:, "xbinning"] = 2.0
    seg = segment_blocks(fr)
    assert seg["block"].nunique() == 2


def test_empty_input_is_handled():
    assert segment_blocks(pd.DataFrame()).empty
    assert summarise_blocks(pd.DataFrame()).empty


# --------------------------------------------------------- split threshold --
def test_split_threshold_lands_between_settles_and_autofocus():
    """The threshold must sit in the empty gap, not at a fixed guess.

    A fixed 240 s default looks safe until it meets a rig whose autofocus takes
    222 s, at which point two consecutive runs of the same filter merge silently.
    """
    from filteroffset.blocks import resolve_split_threshold

    # Shaped like a real archive: a continuum of imaging cadence and settle
    # times up to ~96 s, then a clean jump to autofocus runs of 222 s upward.
    settles = [14, 15, 15, 16, 16, 17, 18, 20, 22, 27, 31, 45, 58, 84, 91, 96]
    autofocus = [222, 228, 235, 241, 249]
    t = resolve_split_threshold(pd.Series(settles + autofocus), BlockParams())
    assert 96 < t < 222, f"threshold {t} did not land in the gap"


def test_split_threshold_adapts_to_a_slow_autofocus():
    from filteroffset.blocks import resolve_split_threshold

    slow = pd.Series([12, 14, 13, 15, 18, 22, 17, 14, 15, 19, 21, 16]
                     + [880, 895, 900, 910])
    t = resolve_split_threshold(slow, BlockParams())
    assert 22 < t < 880, f"threshold {t} did not land in the gap"


def test_split_threshold_honours_an_explicit_value():
    from filteroffset.blocks import resolve_split_threshold

    dead = pd.Series([15, 16, 222, 235])
    assert resolve_split_threshold(dead, BlockParams(block_split_dead_time_s=300.0)) == 300.0


def test_split_threshold_declines_to_guess_without_a_clear_gap():
    """With no separation between settles and autofocus, do not split at all."""
    from filteroffset.blocks import resolve_split_threshold

    flat = pd.Series(np.linspace(100, 200, 40))
    p = BlockParams()
    assert resolve_split_threshold(flat, p) == p.split_search_max_s


def test_a_lone_long_gap_starts_a_new_block():
    """One long gap among a tight cadence is a reasonable block boundary.

    Only one value is required above the gap: a short run legitimately contains
    a single autofocus gap, and a cloud break or meridian flip is itself worth
    treating as a new block. What is required is a real population of ordinary
    cadence *below* the gap, so one stray value cannot define where normal ends.
    """
    from filteroffset.blocks import resolve_split_threshold

    dead = pd.Series([15.0] * 50 + [600.0])
    t = resolve_split_threshold(dead, BlockParams())
    assert 15 < t < 600, t


def test_one_stray_value_cannot_define_the_cadence():
    """Too few ordinary values below the gap means no split is inferred."""
    from filteroffset.blocks import resolve_split_threshold

    p = BlockParams()
    assert resolve_split_threshold(pd.Series([15.0, 600.0]), p) == 240.0


def test_threshold_found_when_autofocus_is_fast_and_cadence_is_slow():
    """A 45 s settle with a 90 s autofocus must still be separated."""
    from filteroffset.blocks import resolve_split_threshold

    t = resolve_split_threshold(pd.Series([45.0] * 30 + [90.0] * 6), BlockParams())
    assert 45 < t < 90, t


def test_same_filter_runs_split_even_with_a_fast_autofocus():
    """The regression this guards: a 225 s autofocus under a 240 s threshold."""
    fr = build_frames(
        pattern=(("B", 1996), ("B", 2001)), n_per=3, exptime=60.0,
        af_gap=225.0, in_gap=15.0,
    )
    seg = segment_blocks(fr)
    assert seg["block"].nunique() == 2
    # ...and again when autofocus happened to return the same position.
    fr2 = build_frames(
        pattern=(("B", 1996), ("B", 1996)), n_per=3, exptime=60.0,
        af_gap=225.0, in_gap=15.0,
    )
    assert segment_blocks(fr2)["block"].nunique() == 2


def test_dither_pauses_do_not_split_a_block():
    fr = build_frames(pattern=(("L", 2020),), n_per=8, exptime=60.0, in_gap=15.0)
    fr.loc[4, "date_obs"] = fr.loc[4, "date_obs"] + pd.Timedelta(seconds=80)
    fr.loc[5:, "date_obs"] = fr.loc[5:, "date_obs"] + pd.Timedelta(seconds=80)
    seg = segment_blocks(fr)
    assert seg["block"].nunique() == 1, "an 80 s settle is not an autofocus run"
