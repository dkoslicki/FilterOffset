"""Recover the autofocus structure of a night from frame metadata.

The observing pattern this package assumes is the ordinary one: change filter,
run autofocus, take N subs, repeat.  Each such run of subs is a **focus block**,
and the focuser position held during the block *is* the autofocus routine's
answer for that filter at that temperature.  That makes the block, not the
individual sub, the natural unit of analysis for filter offsets.

A new block is declared when any of the following happens:

* the filter changes;
* the focuser position changes;
* the equipment configuration changes (binning, camera, telescope);
* there is a time gap long enough that an autofocus run plausibly happened
  even though the position came back the same.

The gap *before* each block is measured and compared with the imaging cadence.
An autofocus run costs real time, so a block whose preceding gap is no longer
than a normal frame-to-frame gap probably did **not** have a fresh autofocus in
front of it, and is flagged accordingly rather than silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BlockParams:
    """Segmentation thresholds."""

    #: Gap (hours) that separates one observing session/night from the next.
    session_gap_hours: float = 8.0
    #: Dead time (seconds, i.e. excluding the exposure itself) after which a
    #: fresh autofocus is assumed even though the focuser position came back
    #: unchanged.  Expressed as dead time rather than as the interval between
    #: exposure starts, because the latter is dominated by the exposure length
    #: and would mean quite different things for 30 s and 1800 s subs.
    #:
    #: ``"auto"`` (the default) reads the threshold off the data instead of
    #: guessing.  Dead times are strongly bimodal - a dither or settle costs
    #: tens of seconds, an autofocus run costs minutes - so the threshold is
    #: placed in the empty gap between the two populations.  A fixed default is
    #: genuinely risky here: 240 s looks safe until it meets a rig whose
    #: autofocus takes 222 s, at which point two consecutive runs of the same
    #: filter are silently merged into one block.
    block_split_dead_time_s: float | str = "auto"
    #: Bounds within which the automatic threshold is allowed to land. The
    #: floor is a little below `af_min_dead_time_s` so that a fast autofocus
    #: following a tight imaging cadence still has room for a threshold between
    #: the two.
    split_search_min_s: float = 40.0
    split_search_max_s: float = 900.0
    #: Absolute wall-clock gap (minutes) that also forces a split.  A safety net
    #: for frames whose EXPTIME is missing, where dead time cannot be computed.
    block_gap_minutes: float = 45.0
    #: Dead time (seconds) beyond the exposure that indicates an autofocus run.
    af_min_dead_time_s: float = 60.0
    #: Treat focuser positions differing by <= this many steps as equal.
    position_tolerance: float = 0.0
    #: Columns that must match for frames to belong to the same block.
    config_columns: tuple[str, ...] = (
        "instrument", "telescope", "xbinning", "ybinning", "gain", "offset",
    )


def resolve_split_threshold(
    dead: pd.Series, params: BlockParams, min_below: int = 3, min_above: int = 1
) -> float:
    """Pick the dead-time threshold that separates settles from autofocus runs.

    Dead times are bimodal: dithers and settles cost tens of seconds, an
    autofocus run costs minutes.  The threshold is placed in the widest empty
    stretch between those two populations, measured on a log scale so the gap
    is judged in proportional rather than absolute terms.
    """
    setting = getattr(params, "block_split_dead_time_s", "auto")
    if not (isinstance(setting, str) and setting.lower() == "auto"):
        return float(setting)

    vals = pd.to_numeric(dead, errors="coerce").dropna().to_numpy(dtype=float)
    lo, hi = float(params.split_search_min_s), float(params.split_search_max_s)
    # Duplicates are kept: they cost nothing (their ratio is 1) and dropping
    # them can leave too few values to judge a gap on a short run.
    vals = np.sort(vals[(vals > 0) & np.isfinite(vals)])
    if vals.size < 4:
        return 240.0

    # Only gaps between *observed* dead times count. Padding the list with
    # sentinels at the search bounds would manufacture an enormous gap between
    # the largest real value and the sentinel, and the threshold would land
    # there instead of between the two real populations.
    # The gap must have a real population of ordinary cadence below it, so that
    # the imaging rhythm - not one stray value - defines where "normal" ends.
    # Only one value is required above, because a short run legitimately
    # contains a single autofocus gap, and a lone long gap (a cloud break, a
    # meridian flip) is itself a reasonable place to start a new block.
    best_ratio, best_thresh = 0.0, None
    n = vals.size
    for i in range(n - 1):
        a, b_ = vals[i], vals[i + 1]
        mid = float(np.sqrt(a * b_))
        if not (lo <= mid <= hi):
            continue
        if (i + 1) < min_below or (n - i - 1) < min_above:
            continue
        ratio = float(b_ / max(a, 1e-9))
        if ratio > best_ratio:
            best_ratio, best_thresh = ratio, mid
    if best_thresh is not None and best_ratio >= 2.0:
        return float(best_thresh)

    # No clean separation between the two populations, so the data do not say
    # where a settle ends and an autofocus run begins. Rather than guess,
    # decline to split on dead time at all and let the unambiguous signals -
    # a filter change, a focuser move, a new session - do the work. Both
    # possible errors here are mild (a missed split merges two runs that shared
    # a position, costing one observation; a spurious split double-counts one
    # run), so acting only on evidence is the honest default.
    return float(hi)


def assign_sessions(frames: pd.DataFrame, params: BlockParams | None = None) -> pd.Series:
    """Label each frame with an integer session (night) index."""
    params = params or BlockParams()
    if frames.empty:
        return pd.Series(dtype="int64", index=frames.index)
    t = pd.to_datetime(frames["date_obs"], utc=True, errors="coerce")
    gap = t.diff().dt.total_seconds().fillna(0.0)
    new = gap > params.session_gap_hours * 3600.0
    return new.cumsum().astype("int64")


def segment_blocks(frames: pd.DataFrame, params: BlockParams | None = None) -> pd.DataFrame:
    """Add block/session bookkeeping columns to a frame table.

    Adds: ``session``, ``session_label``, ``block``, ``index_in_block``,
    ``block_size``, ``is_first_after_af``, ``dead_time_before_s``,
    ``af_likely``, ``temp_comp_active``.
    """
    params = params or BlockParams()
    if frames.empty:
        return frames.copy()

    df = frames.copy()
    df = df.sort_values("date_obs", kind="mergesort").reset_index(drop=True)
    df["session"] = assign_sessions(df, params)

    t = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")
    exptime = pd.to_numeric(df["exptime"], errors="coerce").fillna(0.0)
    # Dead time between the end of one exposure and the start of the next.
    dead = (t.diff().dt.total_seconds() - exptime.shift(1)).astype(float)
    df["dead_time_before_s"] = dead

    pos = pd.to_numeric(df["focus_pos"], errors="coerce")
    filt = df["filter"].astype("string")

    # NB: booleans are accumulated as a plain numpy array.  With pandas >= 3
    # comparisons on pyarrow-backed columns yield ``bool[pyarrow]``, which does
    # not implement ``cumsum``, so staying in numpy keeps this working across
    # pandas 2.x and 3.x alike.
    def _bool(series: pd.Series) -> np.ndarray:
        return series.fillna(False).to_numpy(dtype=bool, na_value=False)

    changed = np.zeros(len(df), dtype=bool)
    changed |= df["session"].diff().fillna(0).to_numpy(dtype=float) != 0.0
    changed |= _bool(filt.ne(filt.shift(1)))
    if params.position_tolerance > 0:
        changed |= (pos - pos.shift(1)).abs().fillna(0).to_numpy(dtype=float) > (
            params.position_tolerance
        )
    else:
        changed |= _bool(pos.ne(pos.shift(1)))
    # Split on dead time (gap minus the preceding exposure), with the raw
    # wall-clock gap kept only as a fallback for missing EXPTIME.
    split_s = resolve_split_threshold(dead, params)
    df.attrs["block_split_dead_time_s"] = float(split_s)
    # Also as a column: DataFrame.attrs does not survive the first merge, and
    # this is the most consequential preprocessing decision in the pipeline.
    df["block_split_dead_time_s"] = float(split_s)
    changed |= dead.fillna(0.0).to_numpy(dtype=float) > split_s
    gap_min = (t.diff().dt.total_seconds() / 60.0).fillna(0.0).to_numpy(dtype=float)
    changed |= gap_min > params.block_gap_minutes
    for col in params.config_columns:
        if col in df.columns:
            c = df[col].astype("string")
            changed |= _bool(c.ne(c.shift(1)))
    changed[0] = True

    df["block"] = np.cumsum(changed).astype("int64") - 1
    df["index_in_block"] = df.groupby("block").cumcount()
    df["block_size"] = df.groupby("block")["block"].transform("size")
    df["is_first_after_af"] = df["index_in_block"] == 0

    # Did an autofocus plausibly run in front of this block?  Compare the dead
    # time before the block with the typical in-block dead time.
    in_block = dead[df["index_in_block"] > 0]
    typical = float(np.nanmedian(in_block)) if in_block.notna().any() else 0.0
    thresh = max(typical + params.af_min_dead_time_s, params.af_min_dead_time_s)
    first = (df["index_in_block"] == 0).to_numpy(dtype=bool)
    af_likely = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    dead_np = dead.to_numpy(dtype=float)
    af_likely.loc[first] = [
        (bool(d > thresh) if np.isfinite(d) else pd.NA) for d in dead_np[first]
    ]
    # The very first block of a session has no measurable preceding gap.
    session_start = df.groupby("session").head(1).index
    af_likely.loc[session_start] = pd.NA
    df["af_likely"] = af_likely
    df.attrs["typical_in_block_dead_time_s"] = typical
    df.attrs["af_dead_time_threshold_s"] = thresh

    # If the focuser moved *within* what is otherwise one continuous filter run,
    # the capture software's own temperature compensation was probably enabled.
    # That breaks the "one position per autofocus" assumption, so detect it.
    df["temp_comp_active"] = _detect_temp_comp(df, params)

    sess_label = (
        t.dt.tz_convert("UTC") - pd.Timedelta(hours=12)
    ).dt.strftime("%Y-%m-%d")
    df["session_label"] = sess_label
    return df


def _detect_temp_comp(df: pd.DataFrame, params: BlockParams) -> pd.Series:
    """Flag runs where the focuser crept while filter and session held constant."""
    t = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")
    pos = pd.to_numeric(df["focus_pos"], errors="coerce")
    exptime = pd.to_numeric(df["exptime"], errors="coerce").fillna(0.0)
    run = pd.Series(
        [f"{a}|{b}" for a, b in zip(df["session"], df["filter"].astype("string"))],
        index=df.index,
        dtype="object",
    )
    same = run.eq(run.shift(1)).fillna(False).to_numpy(dtype=bool)
    dead = (t.diff().dt.total_seconds() - exptime.shift(1)).fillna(1e9)
    short_gap = dead.to_numpy(dtype=float) < params.af_min_dead_time_s
    moved = pos.ne(pos.shift(1)).fillna(False).to_numpy(dtype=bool)
    return pd.Series(same & short_gap & moved, index=df.index)


def summarise_blocks(frames: pd.DataFrame) -> pd.DataFrame:
    """Collapse a segmented frame table to one row per focus block."""
    if frames.empty:
        return pd.DataFrame()
    df = frames.copy()
    t = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")
    df["_t"] = t

    def agg(g: pd.DataFrame) -> pd.Series:
        def num(c: str) -> pd.Series:
            return (
                pd.to_numeric(g[c], errors="coerce") if c in g
                else pd.Series(dtype=float)
            )

        first = g.iloc[0]
        out = {
            "block": int(first["block"]),
            "session": int(first["session"]),
            "session_label": first.get("session_label"),
            "filter": first.get("filter"),
            "focus_pos": float(pd.to_numeric(first.get("focus_pos"), errors="coerce")),
            "n_frames": int(len(g)),
            "t_start": g["_t"].min(),
            "t_mid": g["_t"].min() + (g["_t"].max() - g["_t"].min()) / 2,
            "t_end": g["_t"].max(),
            "ambient_temp": float(num("ambient_temp").mean()),
            "ambient_temp_first": float(num("ambient_temp").iloc[0])
            if num("ambient_temp").notna().any() else np.nan,
            "ambient_temp_range": float(num("ambient_temp").max() - num("ambient_temp").min()),
            "focus_temp": float(num("focus_temp").mean()),
            "ccd_temp": float(num("ccd_temp").mean()),
            # Carried so the temperature assessment can tell a regulated sensor
            # (one tracking its set point) from one following the environment.
            "set_temp": float(num("set_temp").median())
            if "set_temp" in g and num("set_temp").notna().any() else np.nan,
            "altitude": float(num("altitude").mean()),
            "airmass": float(num("airmass").mean()),
            "exptime": float(num("exptime").median()),
            "dead_time_before_s": float(
                pd.to_numeric(first.get("dead_time_before_s"), errors="coerce")
            ),
            "af_likely": first.get("af_likely"),
            "temp_comp_active": bool(g["temp_comp_active"].any())
            if "temp_comp_active" in g else False,
            "object": first.get("object"),
            "instrument": first.get("instrument"),
            "telescope": first.get("telescope"),
        }
        return pd.Series(out)

    blocks = pd.DataFrame(
        [agg(g) for _, g in df.groupby("block", sort=True)]
    ).reset_index(drop=True)
    blocks["elapsed_hours"] = (
        (blocks["t_mid"] - blocks["t_mid"].min()).dt.total_seconds() / 3600.0
    )
    # Hours since the start of the block's own night. Across a campaign
    # spanning weeks the global figure is dominated by the gaps waiting for
    # weather, which makes it useless both as a regressor and as a plot axis.
    blocks["elapsed_hours_in_session"] = (
        (blocks["t_mid"] - blocks.groupby("session")["t_mid"].transform("min"))
        .dt.total_seconds() / 3600.0
    )
    return blocks
