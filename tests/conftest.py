import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def make_blocks(
    n_cycles: int = 6,
    filters: tuple[str, ...] = ("B", "R", "G"),
    offsets: dict[str, float] | None = None,
    k: float = -6.5,
    base: float = 2000.0,
    t_start: float = 17.0,
    t_end: float = 13.0,
    noise: float = 0.0,
    seed: int = 1,
    n_frames: int = 3,
    sessions: int = 1,
    interleave: bool = True,
) -> pd.DataFrame:
    """Synthesise a block table with known offsets and temperature coefficient.

    ``interleave=False`` produces the pathological design where each filter is
    observed in one contiguous temperature range, which is what confounding
    looks like.
    """
    offsets = offsets or {"B": 0.0, "R": -6.0, "G": -5.5}
    rng = np.random.default_rng(seed)
    order: list[str] = []
    for _ in range(n_cycles):
        order.extend(filters)
    if not interleave:
        order = sorted(order, key=lambda f: filters.index(f))
    n = len(order)
    temps = np.linspace(t_start, t_end, n)
    rows = []
    t0 = pd.Timestamp("2026-09-16T01:00:00Z")
    for i, (filt, temp) in enumerate(zip(order, temps)):
        pos = base + offsets.get(filt, 0.0) + k * (temp - float(np.mean(temps)))
        pos += rng.normal(0.0, noise) if noise > 0 else 0.0
        session = i * sessions // n
        t_mid = t0 + pd.Timedelta(minutes=38 * i)
        rows.append(
            {
                "block": i,
                "session": session,
                "session_label": f"2026-09-{16 + session:02d}",
                "filter": filt,
                "focus_pos": float(np.round(pos)),
                "n_frames": n_frames,
                "t_start": t_mid,
                "t_mid": t_mid,
                "t_end": t_mid + pd.Timedelta(minutes=30),
                "ambient_temp": temp,
                "ambient_temp_first": temp,
                "ambient_temp_range": 0.2,
                "focus_temp": np.nan,
                "ccd_temp": -12.0,
                "altitude": 60.0 + 20.0 * np.sin(i / 3.0),
                "airmass": 1.1,
                "exptime": 600.0,
                "dead_time_before_s": 440.0,
                "af_likely": True,
                "temp_comp_active": False,
                "object": "Synthetic",
                "instrument": "TestCam",
                "telescope": "TestScope",
                "qc_pass": True,
            }
        )
    df = pd.DataFrame(rows)
    df["elapsed_hours"] = (
        (df["t_mid"] - df["t_mid"].min()).dt.total_seconds() / 3600.0
    )
    return df


@pytest.fixture
def blocks():
    return make_blocks()
