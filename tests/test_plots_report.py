"""Plot styling, multi-night handling and report asset links."""


import pandas as pd
import pytest

from filteroffset.plots import (
    PlotContext,
    Theme,
    night_labels,
    night_local_hours,
)
from filteroffset.report import _relative_asset


# --------------------------------------------------------------- palette --
@pytest.mark.parametrize(
    "filt,expect_dashed",
    [("L", False), ("R", False), ("G", False), ("B", False),
     ("Ha", True), ("SII", True), ("OIII", True)],
)
def test_narrowband_filters_are_dashed(filt, expect_dashed):
    ctx = PlotContext(theme=Theme.get("light"), palette="filter")
    ctx.assign(["L", "R", "G", "B", "Ha", "SII", "OIII"])
    assert (ctx.style(filt).linestyle == "--") is expect_dashed


def test_filters_use_the_colour_of_their_own_light():
    ctx = PlotContext(theme=Theme.get("light"), palette="filter")
    ctx.assign(["L", "R", "G", "B", "Ha", "SII", "OIII"])

    def channel(hexcode):
        h = hexcode.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    r, g, b = (channel(ctx.style(f).color) for f in ("R", "G", "B"))
    assert r[0] > r[1] and r[0] > r[2], "R should be reddest"
    assert g[1] > g[0] and g[1] > g[2], "G should be greenest"
    assert b[2] > b[0] and b[2] > b[1], "B should be bluest"
    grey = channel(ctx.style("L").color)
    assert max(grey) - min(grey) < 20, "L should be neutral grey"
    # Ha shares R's hue but is distinguished by the dash and marker.
    assert ctx.style("Ha").color == ctx.style("R").color
    assert ctx.style("Ha").linestyle != ctx.style("R").linestyle
    assert ctx.style("Ha").marker != ctx.style("R").marker


def test_every_filter_gets_a_distinct_marker():
    ctx = PlotContext(theme=Theme.get("light"), palette="filter")
    filters = ["L", "R", "G", "B", "Ha", "SII", "OIII"]
    ctx.assign(filters)
    pairs = {(ctx.style(f).color, ctx.style(f).linestyle, ctx.style(f).marker)
             for f in filters}
    assert len(pairs) == len(filters), "styles must be unique per filter"


def test_colour_follows_the_filter_not_its_rank():
    a = PlotContext(theme=Theme.get("light"), palette="filter")
    a.assign(["R", "G", "B"])
    b = PlotContext(theme=Theme.get("light"), palette="filter")
    b.assign(["Ha", "OIII", "R", "G", "B", "SII"])
    for f in ("R", "G", "B"):
        assert a.style(f).color == b.style(f).color


def test_accessible_palette_available():
    ctx = PlotContext(theme=Theme.get("light"), palette="accessible")
    ctx.assign(["R", "G", "B"])
    assert ctx.style("R").color != "#d62728"


def test_dark_theme_has_its_own_steps():
    light = PlotContext(theme=Theme.get("light"), palette="filter")
    dark = PlotContext(theme=Theme.get("dark"), palette="filter")
    light.assign(["R"]), dark.assign(["R"])
    assert light.style("R").color != dark.style("R").color


def test_unknown_filter_still_gets_a_style():
    ctx = PlotContext(theme=Theme.get("light"), palette="filter")
    ctx.assign(["R", "MyDualBand"])
    st = ctx.style("MyDualBand")
    assert st.color and st.marker


# ------------------------------------------------------------ multi-night --
def _frames():
    rows = []
    for sess, day in enumerate(("2026-07-22", "2026-08-18", "2026-09-06")):
        for i in range(5):
            rows.append(
                {
                    "session": sess,
                    "date_obs": pd.Timestamp(f"{day}T02:00:00Z") + pd.Timedelta(hours=i),
                    "session_label": day,
                }
            )
    return pd.DataFrame(rows)


def test_night_local_hours_restarts_each_night():
    df = _frames()
    h = night_local_hours(df)
    assert h.max() == pytest.approx(4.0)
    for _, g in df.assign(h=h).groupby("session"):
        assert g["h"].min() == pytest.approx(0.0)


def test_night_local_hours_ignores_the_gap_between_nights():
    """Six weeks of waiting for weather must not enter the time axis."""
    df = _frames()
    span_days = (df["date_obs"].max() - df["date_obs"].min()).days
    assert span_days > 40
    assert night_local_hours(df).max() < 24


def test_night_labels_are_dates():
    labels = night_labels(_frames())
    assert set(labels.values()) == {"2026-07-22", "2026-08-18", "2026-09-06"}


# ----------------------------------------------------------- report links --
def test_plot_links_keep_their_directory(tmp_path):
    """The report sits beside plots/, so a bare basename resolves to nothing."""
    outdir = tmp_path / "out"
    (outdir / "plots").mkdir(parents=True)
    png = outdir / "plots" / "timeline.png"
    png.write_bytes(b"x")
    assert _relative_asset(str(png), str(outdir)) == "plots/timeline.png"


def test_plot_links_fall_back_safely():
    assert _relative_asset("/elsewhere/timeline.png", "/out") == "plots/timeline.png"
    assert _relative_asset("/a/plots/timeline.png", None) == "plots/timeline.png"
