"""Diagnostic plots.

Design notes worth keeping, because they look like mistakes otherwise:

* **Filters are not drawn in their "own" colours.**  Painting R red, G green and
  B blue is the obvious choice and the wrong one: red/green is exactly the pair
  the commonest form of colour blindness cannot separate.  Filters are assigned
  from a fixed, CVD-validated palette order instead, *and* given distinct
  marker shapes, so identity never rests on colour alone.
* **Colour follows the filter, not its rank.**  The mapping is keyed on a
  canonical filter order, so adding a filter later does not repaint the others
  and two reports of the same setup stay comparable.
* **No chart has two y-axes.**  Focus position and temperature share a time
  axis as stacked panels rather than being overlaid on twin scales, which would
  manufacture an apparent correlation from an arbitrary alignment of two
  ranges.
* Every figure writes a CSV of the values behind it, which doubles as the
  accessible table view and as the audit trail for a published report.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Categorical slots, in the CVD-validated order (see the dataviz palette).
# Used by the "accessible" palette and as the fallback for unknown filters.
_CATEGORICAL_LIGHT = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
_CATEGORICAL_DARK = (
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
)
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Canonical filter order, so a given filter always gets the same slot.
FILTER_ORDER = (
    "L", "R", "G", "B", "Ha", "OIII", "SII", "NII", "Hb",
    "LP", "UVIR", "Duo", "LeNhance", "LeXtreme", "LDN",
)

#: The "filter" palette: each filter is drawn in the colour of the light it
#: passes, which is what an imager expects to see.  Narrowband filters are
#: dashed, which keeps them distinguishable from the broadband filter of the
#: same hue (Ha vs R in particular) and, together with the per-filter marker
#: shapes, means identity is never carried by colour alone - important because
#: red/green is exactly the pair the commonest form of colour blindness cannot
#: separate.  Pass ``--palette accessible`` for the CVD-validated scheme.
_FILTER_PALETTE_LIGHT: dict[str, tuple[str, str, str]] = {
    # filter: (colour, linestyle, marker)
    "L":    ("#7a7a74", "-",  "o"),
    "R":    ("#d62728", "-",  "o"),
    "G":    ("#2ca02c", "-",  "s"),
    "B":    ("#1f6fd0", "-",  "^"),
    "Ha":   ("#d62728", "--", "D"),
    "SII":  ("#e8791a", "--", "v"),
    "OIII": ("#17a2a2", "--", "P"),
    "NII":  ("#b5179e", "--", "X"),
    "Hb":   ("#4a6fe3", "--", "*"),
}
_FILTER_PALETTE_DARK: dict[str, tuple[str, str, str]] = {
    "L":    ("#a8a8a0", "-",  "o"),
    "R":    ("#f2534f", "-",  "o"),
    "G":    ("#3fc23f", "-",  "s"),
    "B":    ("#4d94f0", "-",  "^"),
    "Ha":   ("#f2534f", "--", "D"),
    "SII":  ("#ff9a3c", "--", "v"),
    "OIII": ("#2ec4c4", "--", "P"),
    "NII":  ("#e05fc7", "--", "X"),
    "Hb":   ("#7c96f0", "--", "*"),
}


@dataclass
class Theme:
    """Surface and ink colours for one mode."""

    name: str = "light"
    surface: str = "#fcfcfb"
    text_primary: str = "#0b0b0b"
    text_secondary: str = "#52514e"
    grid: str = "#e4e3df"
    neutral: str = "#8a8984"
    good: str = "#1baf7a"
    warning: str = "#eda100"
    critical: str = "#e34948"
    categorical: tuple[str, ...] = _CATEGORICAL_LIGHT
    filter_palette: dict = field(default_factory=lambda: dict(_FILTER_PALETTE_LIGHT))

    @classmethod
    def get(cls, name: str = "light") -> Theme:
        if str(name).lower() == "dark":
            return cls(
                name="dark", surface="#1a1a19", text_primary="#ffffff",
                text_secondary="#c3c2b7", grid="#383835", neutral="#7a7a74",
                good="#199e70", warning="#c98500", critical="#e66767",
                categorical=_CATEGORICAL_DARK,
                filter_palette=dict(_FILTER_PALETTE_DARK),
            )
        return cls()


@dataclass
class FilterStyle:
    color: str
    marker: str
    linestyle: str = "-"


@dataclass
class PlotContext:
    """Shared style state for a report's figures."""

    theme: Theme = field(default_factory=Theme.get)
    outdir: str = "."
    dpi: int = 150
    styles: dict[str, FilterStyle] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    #: "filter" draws each filter in its own light's colour (narrowband
    #: dashed); "accessible" uses the CVD-validated categorical order.
    palette: str = "filter"

    def style(self, filt: str) -> FilterStyle:
        if filt not in self.styles:
            self.styles[filt] = FilterStyle(self.theme.neutral, "o", "-")
        return self.styles[filt]

    def assign(self, filters: Iterable[str]) -> None:
        """Map filters to styles, deterministically and stably.

        Colour follows the filter, never its rank or count, so adding a filter
        later never repaints the others and two reports of the same rig stay
        comparable.
        """
        present = [f for f in dict.fromkeys(filters) if f is not None]
        ranked = sorted(
            present,
            key=lambda f: (
                FILTER_ORDER.index(f) if f in FILTER_ORDER else len(FILTER_ORDER),
                str(f),
            ),
        )
        n = len(self.theme.categorical)
        for i, f in enumerate(ranked):
            spec = self.theme.filter_palette.get(f) if self.palette == "filter" else None
            if spec is not None:
                self.styles[f] = FilterStyle(spec[0], spec[2], spec[1])
            elif i < n:
                self.styles[f] = FilterStyle(
                    self.theme.categorical[i], _MARKERS[i % len(_MARKERS)], "-"
                )
            else:
                # Past the palette, fall back to neutral plus a distinct marker
                # rather than inventing a hue.
                self.styles[f] = FilterStyle(
                    self.theme.neutral, _MARKERS[i % len(_MARKERS)], "-"
                )

    def save(self, fig, name: str, data: pd.DataFrame | None = None) -> str:
        os.makedirs(self.outdir, exist_ok=True)
        path = os.path.join(self.outdir, f"{name}.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=self.theme.surface)
        plt.close(fig)
        self.written.append(path)
        if data is not None and not data.empty:
            data.to_csv(os.path.join(self.outdir, f"{name}.csv"), index=False)
        return path


def _new_fig(ctx: PlotContext, figsize=(8.0, 5.0), nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=ctx.theme.surface, **kw)
    for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel():
        _style_axes(ctx, ax)
    return fig, axes


def _style_axes(ctx: PlotContext, ax) -> None:
    t = ctx.theme
    ax.set_facecolor(t.surface)
    ax.grid(True, color=t.grid, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.grid)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t.text_secondary, labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(t.text_primary)
    ax.yaxis.label.set_color(t.text_primary)


def _title(ctx: PlotContext, ax, title: str, subtitle: str = "") -> None:
    """Left-aligned title with a wrapped subtitle above the axes.

    The subtitle is wrapped to the axes width and the title padded by the
    resulting line count; otherwise long explanatory subtitles either collide
    with the title or run off the right edge of the figure.
    """
    lines: list[str] = []
    if subtitle:
        import textwrap

        width_in = ax.get_window_extent().width / ax.figure.dpi
        # ~11 characters per inch at 9.5pt is a good approximation.
        ncols = max(int(width_in * 11.5), 28)
        lines = textwrap.wrap(subtitle, width=ncols)
    pad = 8 + 11.5 * len(lines)
    ax.set_title(
        title, color=ctx.theme.text_primary, fontsize=12, fontweight="semibold",
        loc="left", pad=pad,
    )
    if lines:
        ax.annotate(
            "\n".join(lines), xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
            textcoords="offset points", ha="left", va="bottom",
            color=ctx.theme.text_secondary, fontsize=9.5, linespacing=1.35,
        )


def _fig_title(ctx: PlotContext, fig, title: str, subtitle: str = "") -> None:
    """Title and subtitle spanning the whole figure.

    Multi-panel figures must not hang their title off the first axes: the text
    then wraps to one panel's width and collides with that panel's own label.
    """
    import textwrap

    width_in = fig.get_size_inches()[0]
    lines = textwrap.wrap(subtitle, width=max(int(width_in * 12.5), 40)) if subtitle else []
    fig.suptitle(
        title, color=ctx.theme.text_primary, fontsize=12.5, fontweight="semibold",
        x=0.0, y=1.0, ha="left", va="bottom",
    )
    if lines:
        fig.text(
            0.0, 1.0, "\n".join(lines), color=ctx.theme.text_secondary,
            fontsize=9.5, ha="left", va="top", linespacing=1.35,
        )


def night_local_hours(frame: pd.DataFrame, time_col: str = "date_obs") -> pd.Series:
    """Hours elapsed within each night, rather than since the first frame ever.

    A campaign spanning weeks spends almost all of its wall-clock time waiting
    for weather: in the development archive, 46 days of span contained 51 hours
    of imaging, so 95% of a continuous time axis was empty and every night
    collapsed into an unreadable sliver. Time is therefore measured from the
    start of each night, and nights are drawn as separate panels.
    """
    t = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    if "session" in frame.columns:
        base = t.groupby(frame["session"]).transform("min")
    else:
        base = t.min()
    return (t - base).dt.total_seconds() / 3600.0


def night_labels(frame: pd.DataFrame) -> dict:
    """Human-readable label per session, e.g. ``2026-08-21``."""
    out = {}
    if "session" not in frame.columns:
        return out
    t = pd.to_datetime(frame.get("date_obs", frame.get("t_mid")), utc=True, errors="coerce")
    for sess, idx in frame.groupby("session").groups.items():
        lab = None
        if "session_label" in frame.columns:
            vals = frame.loc[idx, "session_label"].dropna()
            lab = str(vals.iloc[0]) if len(vals) else None
        if not lab or lab == "nan":
            tv = t.loc[idx].dropna()
            lab = tv.iloc[0].strftime("%Y-%m-%d") if len(tv) else f"night {sess}"
        out[sess] = lab
    return out


def _panel_label(ctx: PlotContext, ax, text: str) -> None:
    """Name one panel of a multi-panel figure, inside its own axes."""
    ax.annotate(
        text, xy=(1, 0), xycoords="axes fraction", xytext=(-6, 6),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=9.5, color=ctx.theme.text_secondary,
    )


def _legend(ctx: PlotContext, ax, handles=None, labels=None, **kw) -> None:
    leg = ax.legend(
        handles=handles, labels=labels, frameon=False, fontsize=9,
        labelcolor=ctx.theme.text_secondary, **kw,
    )
    if leg:
        for text in leg.get_texts():
            text.set_color(ctx.theme.text_secondary)


def _fig_legend(ctx: PlotContext, fig, filters: Iterable[str], extra=None) -> None:
    """Legend for the whole figure.

    Faceted plots must not hang the legend off one panel: a night that used a
    single filter would then advertise only that filter for the entire figure.
    """
    handles = _filter_handles(ctx, filters)
    if extra:
        handles.extend(extra)
    leg = fig.legend(
        handles=handles, frameon=False, fontsize=9,
        loc="upper right", bbox_to_anchor=(1.0, 1.0),
        ncol=min(len(handles), 6),
    )
    for text in leg.get_texts():
        text.set_color(ctx.theme.text_secondary)


def _filter_handles(ctx: PlotContext, filters: Iterable[str]) -> list[Line2D]:
    out = []
    for f in filters:
        s = ctx.style(f)
        out.append(
            Line2D([], [], color=s.color, marker=s.marker, linestyle=s.linestyle,
                   markersize=7, linewidth=2.0, label=str(f))
        )
    return out


# --------------------------------------------------------------------------- #
# 1. the money plot: position versus temperature
# --------------------------------------------------------------------------- #
def plot_focus_vs_temperature(ctx: PlotContext, fit, name="focus_vs_temperature") -> str:
    """Block focus position against temperature, with parallel per-filter fits.

    The vertical gaps between the parallel lines *are* the filter offsets, which
    is the clearest way to show that the offsets and the thermal slope were
    estimated together rather than one after the other.

    With multiple nights the fitted per-night intercept is subtracted first.
    Without that the points from nine different nights sit at nine different
    levels and the offsets are invisible; removing the night term shows the data
    in the same frame the model actually fits them in.
    """
    # Without a thermal regressor there is nothing to plot against: the x-axis
    # would be a constant smeared out by floating-point noise and the fitted
    # line would be undefined. The offsets themselves are shown by
    # plot_offsets_forest, which does not need temperature.
    if not fit.diagnostics.get("has_temperature", True):
        return ""
    bl = fit.blocks.copy()
    temp_col = fit.temp_column if fit.temp_column in bl else "ambient_temp"
    # Use the temperature the model actually regressed on. When a thermal lag
    # is applied these differ from the raw header values, and plotting the
    # fitted line against the raw column puts the points and the line on
    # different x variables - which shows up as every series sitting parallel
    # to, but offset from, its own fit.
    temp_used = np.asarray(getattr(fit, "temp_values", []), dtype=float)
    if temp_used.size == len(bl):
        bl["_T"] = temp_used
    else:
        bl["_T"] = pd.to_numeric(bl[temp_col], errors="coerce")
    bl["_p"] = pd.to_numeric(bl["focus_pos"], errors="coerce")

    # Show each block where the model puts it relative to its own filter's
    # line, by adding the residual back onto that line. This removes whichever
    # nuisance terms the fit happens to contain - per-night levels, the drift
    # with date, the between-night temperature response - without the plot
    # needing to know which. Plotting raw positions instead makes every series
    # sit parallel to but offset from its own fit.
    resid = np.asarray(getattr(fit, "resid", []), dtype=float)
    adjusted_terms = []
    if getattr(fit, "night_offsets", None):
        adjusted_terms.append("per-night levels")
    if np.isfinite(getattr(fit, "drift_per_day", float("nan"))):
        adjusted_terms.append("drift with date")
    if np.isfinite(getattr(fit, "temp_coeff_between", float("nan"))):
        adjusted_terms.append("the between-night temperature response")
    adjusted = bool(adjusted_terms) and resid.size == len(bl)
    if adjusted:
        line = (
            fit.intercept
            + bl["filter"].map(lambda f: fit.offsets.get(f, np.nan)).to_numpy(float)
            + fit.temp_coeff * (bl["_T"].to_numpy(float) - fit.temp_ref)
        )
        bl["_p"] = line + resid
    # Filters the design cannot identify would plot against a meaningless line.
    unident = set(getattr(fit, "not_identified", []) or [])
    filters = sorted(bl["filter"].dropna().unique().tolist())
    ctx.assign(filters)

    fig, ax = _new_fig(ctx, figsize=(8.6, 5.6))
    xs = np.linspace(bl["_T"].min() - 0.25, bl["_T"].max() + 0.25, 100)
    for f in filters:
        st = ctx.style(f)
        m = bl["filter"] == f
        faded = f in unident
        ax.plot(
            bl.loc[m, "_T"], bl.loc[m, "_p"], linestyle="none", marker=st.marker,
            markersize=8, color=st.color, markeredgecolor=ctx.theme.surface,
            markeredgewidth=1.2, label=f"{f} (offset not identified)" if faded else str(f),
            alpha=0.40 if faded else 1.0, zorder=3,
        )
        if faded:
            continue
        pred = (
            fit.intercept
            + fit.offsets.get(f, 0.0)
            + fit.temp_coeff * (xs - fit.temp_ref)
        )
        ax.plot(xs, pred, color=st.color, linewidth=2.0, alpha=0.9,
                linestyle=st.linestyle, zorder=2)

    lag = getattr(fit, "thermal_lag_minutes", None)
    ax.set_xlabel(
        f"Temperature ({temp_col}"
        + (f", lagged {lag:.0f} min" if lag else "")
        + ") [\u00b0C]"
    )
    ax.set_ylabel(
        "Autofocus position, adjusted [steps]" if adjusted
        else "Autofocus position [steps]"
    )
    sub = (
        f"Parallel fits: shared slope k = {fit.temp_coeff:+.2f} steps/\u00b0C; "
        f"vertical separation = filter offset. Reference {fit.reference}."
    )
    if adjusted:
        sub += " Removed from the points: " + ", ".join(adjusted_terms) + "."
    if unident:
        sub += (
            f" Faded: {', '.join(sorted(unident))} - never shared a night with "
            "the reference, so no offset exists to draw."
        )
    _title(ctx, ax, "Focus position versus temperature, by filter", sub)
    _legend(ctx, ax, loc="best", ncol=min(len(filters), 4), title=None)
    out = bl[["block", "filter", "_T", "_p", "session"]].rename(
        columns={"_T": temp_col, "_p": "focus_pos_adjusted" if adjusted else "focus_pos"}
    )
    return ctx.save(fig, name, out)


# --------------------------------------------------------------------------- #
# 2. offsets with intervals
# --------------------------------------------------------------------------- #
def plot_offsets_forest(ctx: PlotContext, fit, name="offsets_forest") -> str:
    """Dot-and-interval plot of the offsets."""
    tbl = fit.offset_table()
    ctx.assign(tbl["filter"].tolist())
    fig, ax = _new_fig(ctx, figsize=(7.4, 0.8 + 0.62 * max(len(tbl), 2)))

    ax.axvline(0.0, color=ctx.theme.neutral, linewidth=1.2, linestyle="--", alpha=0.8)
    ys = np.arange(len(tbl))[::-1]
    unident = set(getattr(fit, "not_identified", []) or [])
    pooled = getattr(fit, "offsets_pooled", {}) or {}
    for y, row in zip(ys, tbl.itertuples()):
        st = ctx.style(row.filter)
        lo, hi = row.ci_lo, row.ci_hi
        if row.filter in unident or not np.isfinite(row.offset_steps):
            # Say so in words: an empty row reads as a rendering fault, and a
            # drawn point would imply a measurement that does not exist.
            pv = pooled.get(row.filter, float("nan"))
            txt = f"not identified - never shared a night with {fit.reference}"
            if np.isfinite(pv):
                txt += f"   (provisional {pv:+.1f})"
            ax.annotate(
                txt, xy=(0.02, y), xycoords=("axes fraction", "data"),
                xytext=(0, 0), textcoords="offset points", ha="left", va="center",
                fontsize=9, style="italic", color=ctx.theme.text_secondary,
            )
            continue
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=st.color, linewidth=2.0,
                    solid_capstyle="round", alpha=0.9, zorder=2)
        ax.plot([row.offset_steps], [y], marker=st.marker, markersize=9, color=st.color,
                markeredgecolor=ctx.theme.surface, markeredgewidth=1.4, zorder=3)
        label = f"{row.offset_steps:+.1f}"
        if row.is_reference:
            label += "  (reference)"
        elif np.isfinite(lo):
            label += f"  [{lo:+.1f}, {hi:+.1f}]"
        ax.annotate(
            label, xy=(row.offset_steps, y), xytext=(0, 11), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color=ctx.theme.text_secondary,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{r.filter}  (n={r.n_blocks})" for r in tbl.itertuples()],
        color=ctx.theme.text_primary,
    )
    ax.set_ylim(-0.7, len(tbl) - 0.15)
    ax.set_xlabel("Focus offset relative to reference [steps]")
    ax.grid(axis="y", visible=False)
    _title(
        ctx, ax, "Filter offsets with confidence intervals",
        f"{100 * getattr(fit, 'confidence', 0.95):g}% intervals ({fit.ci_method}); "
        "autofocus repeatability "
        f"{fit.resid_scale:.2f} steps per run. Intervals crossing the dashed line "
        "are not distinguishable from zero.",
    )
    return ctx.save(fig, name, tbl)


# --------------------------------------------------------------------------- #
# 3. the night, as stacked panels sharing a time axis
# --------------------------------------------------------------------------- #
def plot_timeline(ctx: PlotContext, fit, frames: pd.DataFrame, name="timeline") -> str:
    """Focus position and temperature through each night.

    One column of panels per night, with time measured from that night's own
    start. A single continuous axis across a multi-week campaign is almost all
    empty gap, which squeezes every night into an unreadable sliver.
    """
    bl = fit.blocks.copy()
    temp_col = (fit.temp_column if fit.temp_column in bl else "ambient_temp")
    frame_temp_col = temp_col.replace("_first", "")
    filters = sorted(bl["filter"].dropna().unique().tolist())
    ctx.assign(filters)

    fr = frames.copy()
    bl["_h"] = night_local_hours(bl, "t_mid")
    fr["_h"] = night_local_hours(fr, "date_obs")
    labels = night_labels(bl)
    sessions = sorted(bl["session"].unique().tolist())
    n_nights = len(sessions)

    # Two rows (focus, temperature) x one column per night, sharing scales so
    # nights are directly comparable.
    max_cols = 6
    shown = sessions[:max_cols]
    ncols = max(len(shown), 1)
    fig, axes = plt.subplots(
        2, ncols, figsize=(max(3.0 * ncols, 6.0), 6.2),
        facecolor=ctx.theme.surface, sharey="row", squeeze=False,
    )
    for ax in axes.ravel():
        _style_axes(ctx, ax)

    for col, sess in enumerate(shown):
        axf, axt = axes[0, col], axes[1, col]
        mb = bl["session"] == sess
        for f in filters:
            st = ctx.style(f)
            m = mb & (bl["filter"] == f)
            if not m.any():
                continue
            axf.plot(
                bl.loc[m, "_h"], pd.to_numeric(bl.loc[m, "focus_pos"], errors="coerce"),
                linestyle="none", marker=st.marker, markersize=7, color=st.color,
                markeredgecolor=ctx.theme.surface, markeredgewidth=1.1,
                label=str(f) if col == 0 else None,
            )
        mf = fr["session"] == sess
        axt.plot(
            fr.loc[mf, "_h"], pd.to_numeric(fr.loc[mf, frame_temp_col], errors="coerce"),
            color=ctx.theme.neutral, linewidth=2.0, alpha=0.95,
        )
        axf.set_title(labels.get(sess, str(sess)), color=ctx.theme.text_secondary,
                      fontsize=9.5, loc="center", pad=4)
        axt.set_xlabel("h into night")
        if col:
            axf.tick_params(labelleft=False)
            axt.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("Focus position\n[steps]")
    axes[1, 0].set_ylabel("Temperature\n[\u00b0C]")

    extra = (
        f" Showing the first {max_cols} of {n_nights} nights; the CSV has them all."
        if n_nights > max_cols else ""
    )
    _fig_title(
        ctx, fig, "Each night, on its own time axis",
        "Time runs from the start of each night, so weeks of waiting for clear "
        "weather do not compress the data into slivers. Focus and temperature "
        "are separate rows; no two quantities share a y-scale." + extra,
    )
    fig.tight_layout()
    _fig_legend(ctx, fig, filters)
    data = pd.DataFrame(
        {
            "session": bl["session"], "night": bl["session"].map(labels),
            "hours_into_night": bl["_h"], "block": bl["block"],
            "filter": bl["filter"], "focus_pos": bl["focus_pos"],
            temp_col: bl[temp_col], "altitude": bl.get("altitude"),
        }
    )
    return ctx.save(fig, name, data)


# --------------------------------------------------------------------------- #
# 4. residual diagnostics
# --------------------------------------------------------------------------- #
def plot_residuals(ctx: PlotContext, fit, name="residuals") -> str:
    """Residuals against temperature, time and influence."""
    bl = fit.blocks.copy()
    resid = np.asarray(fit.resid, dtype=float)
    if len(resid) != len(bl):
        return ""
    temp_col = fit.temp_column if fit.temp_column in bl else "ambient_temp"
    filters = sorted(bl["filter"].dropna().unique().tolist())
    ctx.assign(filters)
    cook = np.asarray(fit.diagnostics.get("cooks_distance", []), dtype=float)
    # Within-night hours, not hours since the campaign began: across a
    # multi-week archive the latter piles every night on top of a few pixels.
    hours = night_local_hours(bl, "t_mid")
    n_nights = int(bl["session"].nunique())

    fig, axes = _new_fig(ctx, figsize=(9.2, 6.6), nrows=2, ncols=2)
    (axa, axb), (axc, axd) = axes

    for ax, x, xlabel in (
        (axa, pd.to_numeric(bl[temp_col], errors="coerce"), "Temperature [C]"),
        (axb, hours, "Hours into night" if n_nights > 1 else "Hours since first block"),
    ):
        ax.axhline(0, color=ctx.theme.neutral, linewidth=1.0, linestyle="--", alpha=0.8)
        for f in filters:
            s = ctx.style(f)
            m = (bl["filter"] == f).to_numpy()
            ax.plot(np.asarray(x)[m], resid[m], linestyle="none", marker=s.marker,
                    markersize=7.5, color=s.color, markeredgecolor=ctx.theme.surface,
                    markeredgewidth=1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residual [steps]")

    # Influence: which blocks are driving the fit.
    if cook.size == len(bl):
        thresh = 4.0 / max(len(bl), 1)
        colors = [
            ctx.theme.critical if c > thresh else ctx.theme.neutral for c in cook
        ]
        axc.bar(bl["block"].astype(int), cook, color=colors, width=0.68)
        axc.axhline(thresh, color=ctx.theme.warning, linewidth=1.5, linestyle="--")
        axc.annotate(
            f"4/n = {thresh:.2f}", xy=(1, thresh), xycoords=("axes fraction", "data"),
            xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom",
            fontsize=8.5, color=ctx.theme.warning,
        )
        axc.set_xlabel("Block")
        axc.set_ylabel("Cook's distance")

    # Normal quantile plot of residuals.
    from scipy import stats

    if resid.size >= 4:
        osm, osr = stats.probplot(resid, dist="norm", fit=False)
        axd.plot(osm, osr, linestyle="none", marker="o", markersize=6.5,
                 color=ctx.theme.categorical[0], markeredgecolor=ctx.theme.surface,
                 markeredgewidth=1.0)
        lim = np.array([np.nanmin(osm), np.nanmax(osm)])
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        axd.plot(lim, lim * (scale if scale > 0 else np.std(resid)),
                 color=ctx.theme.neutral, linewidth=1.5, linestyle="--")
        axd.set_xlabel("Theoretical quantile")
        axd.set_ylabel("Residual [steps]")

    _fig_title(
        ctx, fig, "Residual diagnostics",
        f"Durbin-Watson {fit.diagnostics.get('durbin_watson', float('nan')):.2f} "
        "(2 = no serial correlation). Bars above the dashed line are blocks with "
        "outsized influence on the result.",
    )
    data = pd.DataFrame(
        {
            "block": bl["block"], "filter": bl["filter"], temp_col: bl[temp_col],
            "hours": hours, "residual": resid,
            "cooks_distance": cook if cook.size == len(bl) else np.nan,
            "huber_weight": fit.diagnostics.get("huber_weights", [np.nan] * len(bl)),
        }
    )
    return ctx.save(fig, name, data)


# --------------------------------------------------------------------------- #
# 5. design balance - the confounding question, drawn
# --------------------------------------------------------------------------- #
def plot_design_balance(ctx: PlotContext, fit, name="design_balance") -> str:
    """Temperature coverage per filter, plus the collinearity numbers.

    This is the picture that answers "are my offsets just soaking up thermal
    drift?".  Well-interleaved filters produce overlapping horizontal spans; a
    confounded night produces disjoint ones.
    """
    bl = fit.blocks.copy()
    temp_col = fit.temp_column if fit.temp_column in bl else "ambient_temp"
    bl["_T"] = pd.to_numeric(bl[temp_col], errors="coerce")
    per = fit.diagnostics.get("per_filter", {})
    filters = sorted(per.keys())
    ctx.assign(filters)

    fig, axes = _new_fig(ctx, figsize=(8.8, 0.9 + 0.78 * max(len(filters), 2)),
                         ncols=2, gridspec_kw={"width_ratios": [2.4, 1.0]})
    ax, axv = axes
    ys = np.arange(len(filters))[::-1]
    for y, f in zip(ys, filters):
        s = ctx.style(f)
        info = per[f]
        lo, hi = info.get("temp_min"), info.get("temp_max")
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=s.color, linewidth=6.0, alpha=0.30,
                    solid_capstyle="round", zorder=1)
        m = (bl["filter"] == f).to_numpy()
        ax.plot(bl.loc[m, "_T"], np.full(m.sum(), y), linestyle="none",
                marker=s.marker, markersize=8, color=s.color,
                markeredgecolor=ctx.theme.surface, markeredgewidth=1.2, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{f}  (n={per[f]['n_blocks']})" for f in filters],
        color=ctx.theme.text_primary,
    )
    ax.set_ylim(-0.7, len(filters) - 0.3)
    ax.set_xlabel("Temperature at autofocus [C]")
    ax.grid(axis="y", visible=False)
    _fig_title(
        ctx, fig, "Was each filter sampled across the temperature range?",
        "Overlapping spans mean filter and temperature are separable. "
        "Disjoint spans mean they are not, and the offsets would absorb drift.",
    )

    # VIF panel: the quantitative version of the same question.
    vif = {k.replace("filter[", "").rstrip("]"): v
           for k, v in (fit.diagnostics.get("vif") or {}).items()
           if k.startswith("filter[")}
    if vif:
        keys = [f for f in filters if f in vif]
        vals = [vif[f] for f in keys]
        yv = np.arange(len(keys))[::-1]
        axv.barh(yv, vals, color=[ctx.style(f).color for f in keys], height=0.55)
        axv.axvline(5.0, color=ctx.theme.critical, linewidth=1.5, linestyle="--")
        axv.annotate("VIF 5\n(confounded)", xy=(5.0, axv.get_ylim()[1]),
                     xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                     fontsize=8.5, color=ctx.theme.critical)
        axv.set_yticks(yv)
        axv.set_yticklabels(keys, color=ctx.theme.text_primary)
        axv.set_ylim(-0.7, len(keys) - 0.3)
        axv.set_xlabel("Variance inflation")
        axv.set_xlim(0, max(6.0, max(vals) * 1.25))
        axv.grid(axis="y", visible=False)

    rows = [{"filter": f, **per[f], "vif": vif.get(f, np.nan)} for f in filters]
    return ctx.save(fig, name, pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# 6. does focus actually degrade within a block?
# --------------------------------------------------------------------------- #
def plot_hfd_within_block(
    ctx: PlotContext, frames: pd.DataFrame, metric="hfd_median_bright",
    name="hfd_within_block",
) -> str:
    """Star size against position within the block.

    Tests the working assumption that the first sub after autofocus is the
    sharpest: temperature keeps drifting while the focuser stays put, so star
    size should creep upward through a block.  Plotted as a change relative to
    each block's own first frame, which removes the night's seeing trend.
    """
    df = frames.copy()
    if metric not in df.columns or "index_in_block" not in df.columns:
        return ""
    df["_h"] = pd.to_numeric(df[metric], errors="coerce")
    df = df[np.isfinite(df["_h"])]
    if df.empty:
        return ""
    first = df[df["index_in_block"] == 0].set_index("block")["_h"]
    df["_delta"] = df["_h"] - df["block"].map(first)
    df = df[np.isfinite(df["_delta"])]
    # A single cloud-ruined frame would otherwise dominate the mean trend and
    # manufacture an apparent focus decay, so the summary uses only frames that
    # passed quality control; failures are still drawn, ringed.
    passed = (
        df["qc_pass"].fillna(True).to_numpy(dtype=bool)
        if "qc_pass" in df.columns else np.ones(len(df), dtype=bool)
    )
    filters = sorted(df["filter"].dropna().unique().tolist())
    ctx.assign(filters)

    fig, ax = _new_fig(ctx, figsize=(8.0, 5.0))
    ax.axhline(0, color=ctx.theme.neutral, linewidth=1.2, linestyle="--", alpha=0.85)
    rng = np.random.default_rng(7)
    for f in filters:
        s = ctx.style(f)
        m = df["filter"] == f
        jitter = (rng.random(int(m.sum())) - 0.5) * 0.16
        ax.plot(df.loc[m, "index_in_block"] + jitter, df.loc[m, "_delta"],
                linestyle="none", marker=s.marker, markersize=7.5, color=s.color,
                markeredgecolor=ctx.theme.surface, markeredgewidth=1.1, alpha=0.9,
                label=str(f))
    if (~passed).any():
        ax.plot(df.loc[~passed, "index_in_block"], df.loc[~passed, "_delta"],
                linestyle="none", marker="o", markersize=14, markerfacecolor="none",
                markeredgecolor=ctx.theme.critical, markeredgewidth=2.0,
                label="QC fail (excluded)", zorder=5)
    means = df[passed].groupby("index_in_block")["_delta"].mean()
    ax.plot(means.index, means.to_numpy(), color=ctx.theme.text_primary, linewidth=2.0,
            marker="_", markersize=18, label="mean (QC-passed)", zorder=4)
    for x, v in means.items():
        ax.annotate(f"{v:+.3f}", xy=(x, v), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color=ctx.theme.text_primary)
    ax.set_xlabel("Frame index within block (0 = first after autofocus)")
    ax.set_ylabel(f"Change in {metric} vs first frame [px]")
    ax.set_xticks(sorted(df["index_in_block"].unique()))
    _title(
        ctx, ax, "Does focus decay after the autofocus run?",
        "Each point is one sub, referenced to the first frame of its own block, "
        "which removes the night's seeing trend. An upward trend would be the "
        "thermal drift that temperature compensation exists to cancel.",
    )
    _legend(ctx, ax, loc="best", ncol=min(len(filters) + 1, 5))
    return ctx.save(fig, name, df[["path", "filter", "block", "index_in_block", "_h", "_delta"]])


# --------------------------------------------------------------------------- #
# 7. backend agreement
# --------------------------------------------------------------------------- #
def plot_backend_agreement(
    ctx: PlotContext, profiles: pd.DataFrame, metric="hfd_median_bright",
    name="backend_agreement",
) -> str:
    """Compare the two star-profile measurements.

    Absolute star sizes are *not* expected to match: half-flux diameter depends
    on the measurement aperture, and the two implementations choose it
    differently.  Each backend is therefore divided by its own median, so the
    question becomes whether they agree frame to frame - which is all the
    modelling requires.
    """
    if "backend" not in profiles.columns or metric not in profiles.columns:
        return ""
    wide = profiles.pivot_table(index="path", columns="backend", values=metric)
    backends = [c for c in wide.columns if wide[c].notna().any()]
    if len(backends) < 2:
        return ""
    a, b = backends[0], backends[1]
    scaled = wide[[a, b]].dropna()
    if scaled.empty:
        return ""
    norm = scaled / scaled.median(axis=0)

    fig, axes = _new_fig(ctx, figsize=(9.2, 4.6), ncols=2)
    ax, axr = axes
    lim = [float(norm.min().min()) * 0.98, float(norm.max().max()) * 1.02]
    ax.plot(lim, lim, color=ctx.theme.neutral, linewidth=1.5, linestyle="--", zorder=1)
    ax.plot(norm[a], norm[b], linestyle="none", marker="o", markersize=7,
            color=ctx.theme.categorical[0], markeredgecolor=ctx.theme.surface,
            markeredgewidth=1.1, zorder=3)
    ax.set_xlabel(f"{a}, scale-normalised")
    ax.set_ylabel(f"{b}, scale-normalised")
    r = float(np.corrcoef(norm[a], norm[b])[0, 1]) if len(norm) > 2 else float("nan")
    rho = float(pd.Series(norm[a]).corr(pd.Series(norm[b]), method="spearman")) \
        if len(norm) > 2 else float("nan")
    _fig_title(
        ctx, fig, "Do the two independent measurements agree?",
        f"Pearson r = {r:.3f}, Spearman rho = {rho:.3f} on {len(norm)} frames. "
        f"Absolute scale differs by design ({a} median {scaled[a].median():.2f} px "
        f"vs {b} {scaled[b].median():.2f} px).",
    )

    diff = (norm[a] - norm[b]) * 100.0
    axr.axhline(0, color=ctx.theme.neutral, linewidth=1.2, linestyle="--")
    axr.plot(np.arange(len(diff)), diff.to_numpy(), linestyle="none", marker="o",
             markersize=6.5, color=ctx.theme.categorical[1],
             markeredgecolor=ctx.theme.surface, markeredgewidth=1.0)
    axr.set_xlabel("Frame (sorted by path)")
    axr.set_ylabel("Disagreement [%]")
    out = norm.reset_index().rename(columns={a: f"{a}_norm", b: f"{b}_norm"})
    out["disagreement_pct"] = diff.to_numpy()
    return ctx.save(fig, name, out)


# --------------------------------------------------------------------------- #
# 8. the light-pollution / SNR bias check
# --------------------------------------------------------------------------- #
def plot_hfd_vs_flux(
    ctx: PlotContext, profiles: pd.DataFrame, frames: pd.DataFrame,
    name="hfd_vs_flux_quartile",
) -> str:
    """Star size against brightness quartile, per backend and filter.

    The decisive check on whether a filter only *looks* soft.  Defocus enlarges
    every star equally, so its profile across quartiles is flat.  Background and
    SNR bias - the light-pollution story - inflates the faint end only, giving a
    falling profile.  A rising top quartile means saturated cores instead.
    """
    cols = [f"hfd_fluxq{i}" for i in (1, 2, 3, 4)]
    if not all(c in profiles.columns for c in cols):
        return ""
    df = profiles.merge(frames[["path", "filter"]], on="path", how="left")
    backends = sorted(df["backend"].dropna().unique().tolist())
    filters = sorted(df["filter"].dropna().unique().tolist())
    ctx.assign(filters)

    fig, axes = _new_fig(ctx, figsize=(4.7 * max(len(backends), 1) + 0.6, 4.8),
                         ncols=max(len(backends), 1), squeeze=False)
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, be in zip(axes, backends):
        sub = df[df["backend"] == be]
        for f in filters:
            s = ctx.style(f)
            g = sub[sub["filter"] == f]
            if g.empty:
                continue
            vals = [pd.to_numeric(g[c], errors="coerce").median() for c in cols]
            ax.plot([1, 2, 3, 4], vals, color=s.color, linewidth=2.0, marker=s.marker,
                    markersize=8, markeredgecolor=ctx.theme.surface, markeredgewidth=1.2,
                    linestyle=s.linestyle, label=str(f))
            rows.append({"backend": be, "filter": f,
                         **dict(zip(cols, vals))})
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Q1\nfaintest", "Q2", "Q3", "Q4\nbrightest"])
        ax.set_xlabel("")
        ax.set_ylabel("Median HFD [px]")
        _panel_label(ctx, ax, be)
    _fig_title(
        ctx, fig, "Is the star size real, or a brightness artefact?",
        "Median star size by within-frame flux quartile. A flat profile means "
        "defocus or true seeing; falling to the right means faint-end background "
        "bias (light pollution); rising at Q4 means saturated cores. A gap between "
        "filters that persists at every quartile is real.",
    )
    _legend(ctx, axes[0], loc="best", ncol=min(len(filters), 4))
    return ctx.save(fig, name, pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# 9. star-size excess per filter
# --------------------------------------------------------------------------- #
def plot_hfd_excess(ctx: PlotContext, excess: dict[str, Any], name="hfd_excess") -> str:
    """Per-filter star-size excess across every metric/backend combination."""
    rows = []
    for label, res in (excess or {}).items():
        if res is None or not getattr(res, "excess_frac", None):
            continue
        for f, val in res.excess_frac.items():
            rows.append(
                {
                    "combination": label, "filter": f,
                    "excess_pct": 100.0 * float(val),
                    "se_pct": 100.0 * float(res.excess_log_se.get(f, np.nan)),
                    "p_value": float(res.p_values.get(f, np.nan)),
                    "is_reference": f == res.reference,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    df = df[~df["is_reference"]]
    if df.empty:
        return ""
    filters = sorted(df["filter"].unique().tolist())
    ctx.assign(filters)
    combos = sorted(df["combination"].unique().tolist())

    fig, ax = _new_fig(ctx, figsize=(8.6, 0.9 + 0.5 * len(df)))
    ax.axvline(0, color=ctx.theme.neutral, linewidth=1.2, linestyle="--", alpha=0.85)
    y = 0
    yticks, ylabels = [], []
    for f in filters:
        s = ctx.style(f)
        for combo in combos:
            g = df[(df["filter"] == f) & (df["combination"] == combo)]
            if g.empty:
                continue
            row = g.iloc[0]
            lo = row["excess_pct"] - 1.96 * row["se_pct"]
            hi = row["excess_pct"] + 1.96 * row["se_pct"]
            ax.plot([lo, hi], [y, y], color=s.color, linewidth=2.0, alpha=0.85,
                    solid_capstyle="round")
            ax.plot([row["excess_pct"]], [y], marker=s.marker, markersize=8,
                    color=s.color, markeredgecolor=ctx.theme.surface, markeredgewidth=1.2)
            star = " *" if np.isfinite(row["p_value"]) and row["p_value"] < 0.01 else ""
            ax.annotate(f"{row['excess_pct']:+.1f}%{star}", xy=(hi, y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=ctx.theme.text_secondary)
            yticks.append(y)
            ylabels.append(f"{f} - {combo}")
            y += 1
        y += 0.5
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color=ctx.theme.text_primary, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Star-size excess vs reference filter, conditions removed [%]")
    ax.grid(axis="y", visible=False)
    _title(
        ctx, ax, "Star size not explained by seeing or airmass",
        "An excess consistent across independent backends and metrics is real; "
        "* marks p < 0.01. It does not by itself prove defocus - see the report.",
    )
    return ctx.save(fig, name, df)


# --------------------------------------------------------------------------- #
# 10. quality control over the session
# --------------------------------------------------------------------------- #
def plot_qc_timeline(ctx: PlotContext, frames_qc: pd.DataFrame, name="qc_timeline") -> str:
    """Star counts, elongation and sky level through each night, failures ringed."""
    df = frames_qc.copy()
    t = pd.to_datetime(df["date_obs"], utc=True, errors="coerce")
    if t.isna().all():
        return ""
    df["_h"] = night_local_hours(df, "date_obs")
    labels = night_labels(df)
    sessions = sorted(df["session"].unique().tolist()) if "session" in df else [0]
    n_nights = len(sessions)
    filters = sorted(df["filter"].dropna().unique().tolist())
    ctx.assign(filters)

    panels = [
        ("n_stars", "Stars detected"),
        ("ecc_median", "Median eccentricity"),
        ("background", "Sky level [ADU]"),
    ]
    panels = [(c, lab) for c, lab in panels
              if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not panels:
        return ""

    max_cols = 6
    shown = sessions[:max_cols] if n_nights > 1 else sessions
    ncols = max(len(shown), 1)
    fig, axes = plt.subplots(
        len(panels), ncols, figsize=(max(3.0 * ncols, 7.0), 2.3 * len(panels) + 1.0),
        facecolor=ctx.theme.surface, sharey="row", squeeze=False,
    )
    for ax in axes.ravel():
        _style_axes(ctx, ax)

    bad_all = (~df["qc_pass"].fillna(True)).to_numpy(bool) if "qc_pass" in df else np.zeros(len(df), bool)
    for col, sess in enumerate(shown):
        ms = (df["session"] == sess).to_numpy(bool) if "session" in df else np.ones(len(df), bool)
        for row, (c, label) in enumerate(panels):
            ax = axes[row, col]
            vals = pd.to_numeric(df[c], errors="coerce")
            for f in filters:
                st = ctx.style(f)
                m = ms & (df["filter"] == f).to_numpy(bool)
                if not m.any():
                    continue
                ax.plot(df.loc[m, "_h"], vals[m], linestyle="none", marker=st.marker,
                        markersize=6, color=st.color, markeredgecolor=ctx.theme.surface,
                        markeredgewidth=0.9,
                        label=str(f) if (col == 0 and row == 0) else None)
            mb = ms & bad_all
            if mb.any():
                ax.plot(df.loc[mb, "_h"], vals[mb], linestyle="none", marker="o",
                        markersize=12, markerfacecolor="none",
                        markeredgecolor=ctx.theme.critical, markeredgewidth=1.8,
                        label="QC fail" if (col == 0 and row == 0) else None)
            if col == 0:
                ax.set_ylabel(label)
            else:
                ax.tick_params(labelleft=False)
            if row == 0 and n_nights > 1:
                ax.set_title(labels.get(sess, str(sess)),
                             color=ctx.theme.text_secondary, fontsize=9.5, pad=4)
            if row == len(panels) - 1:
                ax.set_xlabel("h into night" if n_nights > 1 else "Hours since first frame")

    extra = (
        f" Showing the first {max_cols} of {n_nights} nights; the CSV has them all."
        if n_nights > max_cols else ""
    )
    _fig_title(
        ctx, fig, "Quality control across the session",
        "Ringed markers failed quality control. Falling star counts mean "
        "transparency loss; rising eccentricity means tracking or wind, not "
        "focus." + extra,
    )
    fig.tight_layout()
    fail_handle = (
        [Line2D([], [], color=ctx.theme.critical, marker="o", linestyle="none",
                markerfacecolor="none", markeredgewidth=1.8, markersize=10,
                label="QC fail")]
        if bad_all.any() else None
    )
    _fig_legend(ctx, fig, filters, extra=fail_handle)
    keep = ["path", "filter", "block", "session", "_h", "qc_pass", "qc_reasons"] + [c for c, _ in panels]
    return ctx.save(fig, name, df[[c for c in keep if c in df.columns]])


# --------------------------------------------------------------------------- #
# 11. thermal lag profile
# --------------------------------------------------------------------------- #
def plot_thermal_lag(ctx: PlotContext, lag_report: dict[str, Any], name="thermal_lag") -> str:
    """Residual scatter against the assumed thermal time constant."""
    tr = (lag_report or {}).get("trace")
    if tr is None or getattr(tr, "empty", True):
        return ""
    fig, ax = _new_fig(ctx, figsize=(8.0, 4.8))
    ax.plot(tr["tau_minutes"], tr["rss"], color=ctx.theme.categorical[0], linewidth=2.0,
            marker="o", markersize=7, markeredgecolor=ctx.theme.surface,
            markeredgewidth=1.1, label="residual sum of squares")
    best = lag_report.get("best_tau_by_rss")
    if best is not None and np.isfinite(best):
        ax.axvline(best, color=ctx.theme.categorical[1], linewidth=1.8, linestyle="--")
        ax.annotate(f"best fit {best:.0f} min", xy=(best, tr["rss"].max()),
                    xytext=(6, -6), textcoords="offset points", fontsize=9,
                    color=ctx.theme.categorical[1], va="top")
    chosen = lag_report.get("best_tau_by_aicc")
    label = f"{chosen:.0f} min" if chosen else "no lag"
    ax.set_xlabel("Assumed thermal time constant tau [minutes]")
    ax.set_ylabel("Unexplained scatter (RSS) [steps^2]")
    _title(
        ctx, ax, "Do the optics lag the air temperature?",
        f"A dip away from tau = 0 means the tube follows the air with a delay. "
        f"After penalising the extra parameter the selected value is {label}.",
    )
    _legend(ctx, ax, loc="best")
    return ctx.save(fig, name, tr)


# --------------------------------------------------------------------------- #
# 12. V-curve (focus sweeps)
# --------------------------------------------------------------------------- #
def plot_vcurves(ctx: PlotContext, sweeps: dict[str, Any], name="vcurves") -> str:
    """Measured star size against focuser position, with fitted hyperbolae."""
    if not sweeps:
        return ""
    from .vcurve import hyperbola_hfd

    filters = sorted(sweeps.keys())
    ctx.assign(filters)
    fig, ax = _new_fig(ctx, figsize=(8.4, 5.2))
    rows = []
    for f in filters:
        entry = sweeps[f]
        fit, pos, hfd = entry["fit"], np.asarray(entry["positions"]), np.asarray(entry["hfd"])
        s = ctx.style(f)
        ax.plot(pos, hfd, linestyle="none", marker=s.marker, markersize=8, color=s.color,
                markeredgecolor=ctx.theme.surface, markeredgewidth=1.2, label=str(f))
        if np.isfinite(fit.best_position):
            xs = np.linspace(pos.min(), pos.max(), 200)
            ax.plot(xs, hyperbola_hfd(xs, fit.best_position, fit.hfd_min,
                                      fit.slope_px_per_step),
                    color=s.color, linewidth=2.0, alpha=0.85, linestyle=s.linestyle)
            ax.axvline(fit.best_position, color=s.color, linewidth=1.2, linestyle=":",
                       alpha=0.8)
        for p, h in zip(pos, hfd):
            rows.append({"filter": f, "focus_pos": p, "hfd": h})
    ax.set_xlabel("Focuser position [steps]")
    ax.set_ylabel("Star size (HFD) [px]")
    _title(
        ctx, ax, "Focus sweeps: measured V-curves",
        "Dotted verticals mark each filter's fitted best focus. The separation "
        "between them is a direct, autofocus-independent measurement of the offsets.",
    )
    _legend(ctx, ax, loc="best", ncol=min(len(filters), 4))
    return ctx.save(fig, name, pd.DataFrame(rows))
