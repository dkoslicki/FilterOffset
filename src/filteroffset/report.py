"""Markdown and HTML reporting.

The report is ordered the way the question should actually be answered: what
the data can support, then the numbers, then the evidence, then what to do
next.  Every table is also written to CSV next to the report so results are
traceable and diff-able in version control.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .apt import instructions, offsets_for_capture_software, to_json
from .vcurve import focus_tolerances, steps_to_microns_table

GLOSSARY = """\
Every symbol and statistic used anywhere in this report.

**Quantities you act on**

| symbol | meaning |
| --- | --- |
| `offset` | Focus offset in **focuser steps**, relative to the reference filter. Add it to the reference filter's in-focus position to get this filter's. Negative = comes to focus at a lower step count. |
| `k` | **Temperature coefficient**, in focuser **steps per degree Celsius**. The slope of best-focus position against temperature, shared by all filters. Negative means the focuser must move to a lower step count as it gets colder. This is the number capture software calls "temperature compensation". |
| `reference filter` | The filter assigned offset 0. Changing it shifts every offset by a constant and changes nothing physical. |
| `tau` (thermal lag) | Time constant in **minutes** for how slowly the optics follow the air temperature. 0 means focus tracks the air instantly. |

**Units and measurements**

| symbol | meaning |
| --- | --- |
| `step` | One increment of the focuser motor. Its size in microns depends on the telescope the focuser drives, not only on the motor, so this report keeps everything in steps and converts only where marked. |
| `HFD` | **Half-flux diameter**: the diameter of the circle containing half a star's light, in **pixels**. The standard autofocus metric. It depends on the measurement aperture, so absolute values are not comparable between programs - only patterns are. |
| `FWHM` | **Full width at half maximum** of the star profile, in pixels. For a Gaussian star, FWHM and HFD coincide. |
| `eccentricity` | Star elongation, 0 = round, 1 = a line. Rising eccentricity means tracking, guiding or wind - not focus. |
| `airmass` | Thickness of atmosphere along the line of sight; 1.0 at the zenith, ~2 at 30 degrees altitude. |
| `block` | A run of consecutive subs sharing one filter and one focuser position, i.e. the subs taken after one autofocus run. The unit of analysis. |
| `session` / `night` | Frames separated from the next group by a long gap (default 8 hours). |

**Statistics**

| symbol | meaning |
| --- | --- |
| `SE` | **Standard error**: the estimated uncertainty of a fitted number. Roughly, the true value lies within about 2 SE of the estimate. |
| `CI` | **Confidence interval**: a range that would contain the true value 95% of the time on repeated data. An interval spanning 0 means the quantity is not distinguishable from zero. |
| `p` | **p-value**: the probability of seeing an effect this large if there were really no effect. Small `p` (below ~0.05) means "unlikely to be chance". It does **not** measure how *large* an effect is. |
| `F` | **F-statistic**: compares a bigger model against a smaller one. Large `F` with small `p` means the extra parameters earn their place. Reported as `F = value, p = value`. |
| `R^2` | Fraction of the variation the model explains, 0 to 1. 0.98 means the model accounts for 98% of the spread in focuser positions. |
| `VIF` | **Variance inflation factor**: how much a coefficient's uncertainty is inflated by overlap with the other regressors. 1 = no overlap, above ~5 = the terms are carrying nearly the same information and cannot be separated cleanly. Infinite = the term is an exact combination of others and does not exist independently at all. |
| `Cook's distance` | How much the whole fit would move if one block were deleted. Values above `4/n` mark blocks with outsized influence. |
| `Durbin-Watson` | Tests whether residuals are correlated with their neighbours in time. 2 = no correlation; below ~1.5 suggests structure the model is missing. |
| `AICc` | Model-selection score (smaller is better) that charges for each extra parameter, with a small-sample correction. Used to decide whether a thermal lag earns its keep. |
| `Huber M-estimator` | A robust alternative to least squares: outlying blocks get down-weighted instead of dragging the fit. |
| `bootstrap` | Uncertainty estimated by refitting on resampled data many times. Preferred when there are enough independent nights; otherwise Student-t intervals are used. |
| `block fixed effects` | Removing each block's own average before fitting, so only within-block variation contributes and night-to-night differences cannot leak in. |
| `night intercepts` | A free level per night, absorbing focuser zero-point shifts between sessions (re-seating the camera, a power cycle, backlash take-up). |
| `identified` / `not identified` | Whether the data can determine a quantity **at all**. With per-night intercepts, a filter offset is identified only if that filter shares a night with the reference filter, directly or through a chain of other filters. |
| `zero point` | The focuser step count corresponding to a given physical drawtube position. It stays fixed as long as the imaging train is not disturbed, which is what makes positions comparable between nights. |
| `drift (steps/day)` | Slow, smooth change in focus position with date -- mechanical settling, or a seasonal effect the temperature term misses. Absorbed by one parameter. |
| `jump ratio` | Night-to-night scatter after removing any drift, divided by the scatter within a night. At or below 1 the nights are as reproducible as autofocus itself; above 1 the zero point is genuinely moving. |
| `within-night` vs `pooled` | Which contrasts an estimate uses. *Within-night* compares blocks of the same filter on the same night, so nothing that differs between nights can enter. *Pooled* also compares across nights: more precise, but it imports any between-night confounder. |
| `anchor filter` | A filter observed on enough nights that a night's level can be told apart from that filter's own offset; used to measure zero-point stability. |
"""

_SEV_BADGE = {
    "critical": "**CRITICAL**",
    "warning": "**WARNING**",
    "info": "INFO",
    "ok": "OK",
}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_(nothing to report)_\n"
    d = df.copy()
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows)
    def fmt(v):
        if isinstance(v, float):
            if not np.isfinite(v):
                return "-"
            return floatfmt.format(v)
        if isinstance(v, (pd.Timestamp, _dt.datetime)):
            return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S")
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return str(v)
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in d.itertuples(index=False)
    )
    extra = ""
    if max_rows is not None and len(df) > max_rows:
        extra = f"\n\n_({len(df) - max_rows} further rows in the CSV.)_\n"
    return f"{header}\n{rule}\n{body}\n{extra}"


def _pct(conf: float) -> str:
    """Render a confidence level as a label, e.g. 0.9 -> '90%'."""
    return f"{100.0 * float(conf):g}%"


def build_markdown(
    fit,
    recommendations,
    frames_qc: pd.DataFrame,
    blocks_qc: pd.DataFrame,
    frame_flags: pd.DataFrame,
    block_flags: pd.DataFrame,
    profiles: pd.DataFrame,
    hfd_excess: dict[str, Any] | None,
    sensitivity: pd.DataFrame | None,
    lag_report: dict[str, Any] | None,
    plot_paths: list[str],
    step_size_um: float | None = None,
    inputs: dict[str, Any] | None = None,
    outdir: str | None = None,
) -> str:
    """Render the full report as Markdown."""
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    diag = fit.diagnostics or {}
    L: list[str] = []
    A = L.append

    A("# Filter offset & temperature compensation report\n")
    A(f"_Generated {now} by filteroffset v{__version__}._\n")

    # ---------------------------------------------------------------- summary
    eq = frames_qc.iloc[0] if not frames_qc.empty else {}
    A("## Equipment and data\n")
    kit = pd.DataFrame(
        [
            {"property": "Telescope", "value": eq.get("telescope")},
            {"property": "Camera", "value": eq.get("instrument")},
            {"property": "Focal length", "value": f"{eq.get('focal_len')} mm"},
            {"property": "Aperture", "value": f"{eq.get('aperture')} mm"},
            {
                "property": "Focal ratio",
                "value": f"f/{float(eq.get('focal_ratio')):.2f}"
                if eq.get("focal_ratio") else "-",
            },
            {"property": "Pixel size", "value": f"{eq.get('xpixsz')} um"},
            {
                "property": "Pixel scale",
                "value": f"{float(eq.get('pixel_scale')):.2f} arcsec/px"
                if eq.get("pixel_scale") else "-",
            },
            {"property": "Capture software", "value": eq.get("software")},
            {"property": "Target", "value": eq.get("object")},
            {"property": "Frames analysed", "value": int(len(frames_qc))},
            {"property": "Autofocus blocks", "value": int(len(blocks_qc))},
            {"property": "Nights", "value": int(diag.get("n_nights", 1))},
            {
                "property": "Temperature source",
                "value": f"{fit.temp_column} (header keyword)",
            },
        ]
    )
    A(_md_table(kit))

    A("## Verdict\n")
    A(f"**{recommendations.verdict.upper()}** -- {recommendations.headline}\n")
    counts = recommendations.counts()
    A(
        f"Findings: {counts['critical']} critical, {counts['warning']} warning, "
        f"{counts['info']} informational, {counts['ok']} checks passed.\n"
    )
    A(
        f"- Filter offsets: **{'trustworthy' if recommendations.trust_offsets else 'NOT trustworthy as-is'}**\n"
        f"- Temperature coefficient: **{'trustworthy' if recommendations.trust_temp_coeff else 'provisional'}**\n"
    )

    # ---------------------------------------------------------------- results
    A("\n## Recommended settings\n")
    soft = offsets_for_capture_software(fit)
    A(_md_table(soft))
    A("\n### How to apply\n")
    for line in instructions(fit, step_size_um):
        A(f"- {line}")
    A("")

    if not np.isfinite(fit.temp_coeff):
        A(
            "\n**No temperature coefficient is reported**: no header temperature "
            "could serve as a thermal regressor - see the findings for which "
            "sources were considered and why each was rejected. The offsets "
            "above are unaffected, being differences between filters.\n"
        )
    else:
      A(
        f"\nFitted temperature coefficient: **{fit.temp_coeff:+.3f} "
        f"+/- {fit.temp_coeff_se:.3f} steps/C** "
        f"(95% CI {fit.temp_coeff_ci[0]:+.2f} ... {fit.temp_coeff_ci[1]:+.2f}), "
        f"referenced at {fit.temp_ref:.2f} C.\n"
    )
    if fit.thermal_lag_minutes:
        A(
            f"A thermal lag of {fit.thermal_lag_minutes:.0f} minutes was applied to "
            "the temperature series before fitting.\n"
        )

    # ------------------------------------------------- physical significance
    A("\n## Is the offset physically significant?\n")
    fratio = _num(eq.get("focal_ratio"))
    pixel = _num(eq.get("xpixsz"))
    seeing = _median(frames_qc, "fwhm_median")
    tol = focus_tolerances(fratio or float("nan"), seeing_fwhm_px=seeing, pixel_um=pixel)
    if np.isfinite(tol.get("diffraction_total_um", float("nan"))):
        A(
            f"At f/{fratio:.2f} the depth of focus is about "
            f"**{tol['diffraction_total_um']:.1f} um** by the quarter-wave "
            "diffraction criterion"
            + (
                f", and about **{tol['seeing_total_um']:.1f} um** by the practical "
                f"seeing-limited criterion (measured FWHM {seeing:.2f} px)"
                if tol.get("seeing_total_um") else ""
            )
            + ". An offset much smaller than this will not change your stars.\n"
        )
    offs = {f: v for f, v in fit.offsets.items() if f != fit.reference}
    if offs:
        A(
            "\nBecause the focuser's microns-per-step depends on the telescope it is "
            "attached to (not only on the motor and gearbox), the table below gives "
            "the offsets in microns across plausible step sizes rather than assuming "
            "one:\n"
        )
        A(
            _md_table(
                steps_to_microns_table(
                    offs, tolerance_um=tol.get("seeing_total_um")
                ),
                floatfmt="{:.2f}",
            )
        )
        A(
            "\nTo replace this table with a measured value, take a focus sweep "
            "(several frames at deliberately spaced focuser positions). This tool "
            "fits the V-curve slope and inverts the defocus geometry to calibrate "
            "microns-per-step directly.\n"
        )

    # ---------------------------------------------------------- the findings
    A("\n## Findings\n")
    for f in recommendations.findings:
        A(f"### {_SEV_BADGE.get(f.severity, f.severity)} -- {f.title}\n")
        A(f"{f.detail}\n")
        if f.action:
            A(f"**What to do:** {f.action}\n")

    # ------------------------------------------------------------ the design
    A("\n## Can this data separate filter from temperature?\n")
    A(
        "This is the question that decides whether the offsets mean anything. "
        "Because temperature is fitted at the same time as the filter terms, the "
        "offsets are estimated *controlling for* thermal drift rather than against "
        "a moving zero point. That only works if the observing order actually "
        "spread each filter across the night's temperature range.\n"
    )
    per = diag.get("per_filter", {})
    rows = []
    vif = diag.get("vif", {})
    for f, info in sorted(per.items()):
        rows.append(
            {
                "filter": f,
                "blocks": info["n_blocks"],
                "frames": info["n_frames"],
                "temp_min": info["temp_min"],
                "temp_max": info["temp_max"],
                "temp_span": info["temp_span"],
                "corr_with_temp": info["corr_with_temp"],
                "VIF": vif.get(f"filter[{f}]", np.nan),
            }
        )
    A(_md_table(pd.DataFrame(rows)))
    A(
        f"\nDesign condition number: {diag.get('condition_number', float('nan')):.2f} "
        "(low is good). Variance inflation factors near 1 mean the filter terms and "
        "the temperature term carry independent information.\n"
    )
    wf = diag.get("within_filter_k") or {}
    if wf.get("k_pooled") is not None:
        A(
            "\n### Independent cross-check of the temperature coefficient\n"
            "The estimate below uses **only comparisons between blocks of the same "
            "filter**, so it is mathematically incapable of seeing filter offsets. "
            "If it agrees with the joint fit, the joint fit really did separate the "
            "two effects.\n"
        )
        wrows = [
            {
                "estimator": "joint fit (all filters, offsets included)",
                "k_steps_per_C": fit.temp_coeff,
                "se": fit.temp_coeff_se,
            },
            {
                "estimator": "within-filter only (offset-blind)",
                "k_steps_per_C": wf["k_pooled"],
                "se": wf.get("k_pooled_se", np.nan),
            },
        ]
        for f, e in sorted((wf.get("per_filter") or {}).items()):
            if "k" in e:
                wrows.append(
                    {
                        "estimator": f"  {f} alone ({e['n_blocks']} blocks, "
                        f"{e['temp_span']:.1f} C)",
                        "k_steps_per_C": e["k"],
                        "se": e.get("k_se", np.nan),
                    }
                )
        A(_md_table(pd.DataFrame(wrows)))

    # ------------------------------------------- focuser zero-point stability
    d = getattr(fit, "drift_test", None)
    if diag.get("n_nights", 1) > 1:
        A("\n## Are focuser positions comparable between nights?\n")
        A(
            "This analysis assumes the imaging train was not disturbed between "
            "sessions -- no camera re-seated, no spacers changed, no focuser "
            "rehomed -- so an absolute encoder puts every night on one scale. That "
            "assumption is what allows a filter used alone on its own nights to be "
            "compared with the rest at all, so it is tested rather than assumed.\n"
        )
        if d is not None and getattr(d, "tested", False):
            rows = [
                {"quantity": "nights tested", "value": d.n_nights},
                {"quantity": "anchor filters", "value": ", ".join(d.anchor_filters)},
                {"quantity": "campaign span (days)", "value": d.span_days},
                {"quantity": "drift with date (steps/day)", "value": d.drift_per_day},
                {"quantity": "drift over the campaign (steps)",
                 "value": d.drift_total_steps},
                {"quantity": "drift p-value", "value": d.drift_p},
                {"quantity": "between-night scatter (steps)", "value": d.between_sd},
                {"quantity": "  ... after removing the trend",
                 "value": d.between_sd_detrended},
                {"quantity": "within-night scatter (steps)", "value": d.within_sd},
                {"quantity": "jump ratio (detrended between / within)",
                 "value": d.jump_ratio},
                {"quantity": "verdict", "value": d.verdict},
            ]
            A(_md_table(pd.DataFrame(rows)))
            A(
                "\nA jump ratio at or below 1 means night-to-night scatter is no "
                "larger than autofocus repeatability itself, so there is nothing "
                "for per-night intercepts to absorb and the between-night "
                "information can be used. Above 1, the focuser's zero point is "
                "genuinely moving and per-night intercepts are applied instead -- "
                "which protects the offsets but can leave an unshared filter "
                "without one.\n"
            )
        else:
            A(
                f"_Not testable here: {getattr(d, 'reason', 'unknown')}._ Positions "
                "are being treated as comparable across nights on the strength of "
                "the assumption alone.\n"
            )
        A(
            f"\nPer-night intercepts were **{'used' if diag.get('night_effects') else 'not used'}** "
            f"in the fit. The temperature coefficient was estimated from "
            f"**{getattr(fit, 'temp_from', 'pooled').replace('_', '-')}** variation"
            + (
                f", against {fit.temp_coeff_pooled:+.2f} steps/C had between-night "
                "contrasts been pooled in."
                if np.isfinite(getattr(fit, "temp_coeff_pooled", float("nan")))
                else "."
            )
            + "\n"
        )
        if np.isfinite(getattr(fit, "drift_per_day", float("nan"))):
            A(
                f"\nA linear drift of **{fit.drift_per_day:+.3f} +/- "
                f"{fit.drift_per_day_se:.3f} steps/day** is included in the model.\n"
            )

    # ------------------------------------------------------ model robustness
    if sensitivity is not None and not sensitivity.empty:
        A("\n## Does the answer depend on how it was fitted?\n")
        A(
            "Each row refits the model under a different defensible specification. "
            "A number that moves is a number the data do not pin down. The final "
            "row is included as a warning, not a candidate: on a single night, "
            "elapsed time and temperature are nearly the same variable, so adding a "
            "time drift fits better while destroying the meaning of steps/C.\n"
        )
        cols = [
            c for c in (
                "variant", "k_steps_per_C", "k_se", "resid_rms",
                "thermal_lag_min", "drift_steps_per_hour", "corr_temp_elapsed",
            ) if c in sensitivity.columns
        ]
        cols += sorted(c for c in sensitivity.columns if c.startswith("offset_"))
        A(_md_table(sensitivity[cols], floatfmt="{:.3f}"))

    if lag_report and not getattr(lag_report.get("trace"), "empty", True):
        tr = lag_report["trace"]
        best = lag_report.get("best_tau_by_rss")
        A(
            f"\nThermal lag: focus follows the glass and tube, which lag the air. "
            f"Profiling the time constant over {len(tr)} values put the best fit at "
            f"{best:.0f} minutes"
            + (
                f" (residual sum of squares {lag_report.get('rss_best', float('nan')):.1f} "
                f"against {lag_report.get('rss_at_zero', float('nan')):.1f} with no lag)"
                if np.isfinite(lag_report.get("rss_best", float("nan"))) else ""
            )
            + ". The profile is plotted below and tabulated in the plots folder.\n"
        )

    # ---------------------------------------------------------- measurements
    A("\n## Star profile measurements\n")
    A(
        "Star sizes were measured twice, by independent implementations, because "
        "half-flux diameter is notoriously sensitive to how it is computed. The two "
        "are **not** expected to agree in absolute value -- HFD depends on the "
        "measurement aperture, and stellar profiles have extended wings, so there is "
        "no aperture-independent 'true' HFD. What must agree is the frame-to-frame "
        "pattern, which is all the modelling uses.\n"
    )
    if profiles is not None and not profiles.empty and "backend" in profiles:
        summ = (
            profiles.groupby("backend")
            .agg(
                frames=("path", "nunique"),
                median_hfd=("hfd_median", "median"),
                median_hfd_bright=("hfd_median_bright", "median"),
                median_stars=("n_stars", "median"),
                failures=("ok", lambda s: int((~s.fillna(True).astype(bool)).sum())),
            )
            .reset_index()
        )
        A(_md_table(summ))
    A(
        "\nTwo ASTAP behaviours are worth recording because they silently corrupt "
        "results: `-analyse` reports the median HFD rounded to one decimal place, "
        "which is too coarse for this work; and the `HFD_MEDIAN` that `-extract` "
        "prints on stdout disagrees with the median of the CSV it writes. This tool "
        "uses `-extract` and computes statistics from the per-star CSV.\n"
    )

    if hfd_excess:
        A("\n### Star size not explained by observing conditions\n")
        A(
            "Autofocus can only be as good as its own search. If it systematically "
            "lands off best focus for one filter, that error is invisible in the "
            "focuser positions and silently becomes part of the offset. This test "
            "removes the night's seeing trend and airmass, then asks whether any "
            "filter's stars remain larger than the rest.\n"
        )
        rows = []
        for label, res in hfd_excess.items():
            if res is None or not getattr(res, "excess_frac", None):
                continue
            for f, v in res.excess_frac.items():
                if f == res.reference:
                    continue
                rows.append(
                    {
                        "metric / backend": label,
                        "filter": f,
                        "excess_%": 100.0 * v,
                        "p_value": res.p_values.get(f, np.nan),
                        "n_frames": res.n_frames,
                    }
                )
        A(_md_table(pd.DataFrame(rows)))

    # -------------------------------------------------------- quality control
    A("\n## Quality control\n")
    n_fail = int((~frames_qc["qc_pass"]).sum()) if "qc_pass" in frames_qc else 0
    A(
        f"{len(frames_qc) - n_fail} of {len(frames_qc)} frames passed. Failures are "
        "excluded from the star-size analysis. They do **not** affect the offsets "
        "derived from autofocus positions, which depend on the focuser reading "
        "rather than on image quality.\n"
    )
    if not frame_flags.empty:
        ff = frame_flags.copy()
        ff["frame"] = ff["path"].map(os.path.basename)
        A(_md_table(ff[["frame", "check", "severity", "detail"]], max_rows=20))
    if not block_flags.empty:
        A("\n### Block-level flags\n")
        A(_md_table(block_flags))

    A("\n### Autofocus verification\n")
    A(
        "An autofocus run costs real time, so the dead time before each block is "
        "evidence for whether one actually ran -- this was checked rather than "
        "assumed.\n"
    )
    if "af_likely" in blocks_qc.columns:
        vals = blocks_qc["af_likely"]
        n_yes = int((vals == True).sum())  # noqa: E712
        n_no = int((vals == False).sum())  # noqa: E712
        n_na = int(len(vals) - n_yes - n_no)
        A(
            f"{n_yes} of {len(blocks_qc)} blocks show a clear pause consistent "
            f"with a filter change plus autofocus; {n_no} do not; {n_na} begin a "
            "session and have no measurable preceding gap.\n"
        )
    # One row per night rather than per block: the per-block detail lives in
    # blocks.csv, and on a long campaign it runs to a hundred rows here.
    if "session" in blocks_qc.columns:
        grp = blocks_qc.groupby("session")
        summary = pd.DataFrame(
            {
                "night": grp["session_label"].first()
                if "session_label" in blocks_qc.columns else grp.size().index,
                "blocks": grp.size(),
                "filters": grp["filter"].apply(lambda x: " ".join(sorted(set(x.dropna())))),
                "focus_min": grp["focus_pos"].min(),
                "focus_max": grp["focus_pos"].max(),
                "temp_min": grp["ambient_temp_first"].min()
                if "ambient_temp_first" in blocks_qc.columns else np.nan,
                "temp_max": grp["ambient_temp_first"].max()
                if "ambient_temp_first" in blocks_qc.columns else np.nan,
            }
        ).reset_index(drop=True)
        A(_md_table(summary, floatfmt="{:.1f}"))
        A("\n_Per-block detail is in `blocks.csv`._\n")

    # ------------------------------------------------------------- the plots
    if plot_paths:
        A("\n## Diagnostic plots\n")
        for p in plot_paths:
            if not p:
                continue
            base = os.path.basename(p)
            # The report sits at <outdir>/report.md while figures are written to
            # <outdir>/plots/, so the link must keep the directory component or
            # the images resolve to nothing once the folder is moved or opened
            # from disk.
            rel = _relative_asset(p, outdir)
            A(f"### {base.replace('.png', '').replace('_', ' ').title()}\n")
            A(f"![{base}]({rel})\n")
        A(
            "_Each figure has a matching `.csv` containing the exact values plotted, "
            "both as an accessible table view and as an audit trail._\n"
        )

    A("\n## What the symbols mean\n")
    A(GLOSSARY)

    A("\n## Method\n")
    A(_METHOD_TEXT)

    A("\n## Reproducing this run\n")
    if inputs:
        A("```\n" + "\n".join(f"{k}: {v}" for k, v in inputs.items()) + "\n```\n")
    return "\n".join(L)


_METHOD_TEXT = """\
**The standing assumption.** The imaging train is taken to be undisturbed
between sessions -- no camera re-seated, no spacers or filters swapped, no
focuser rehomed -- so an absolute-encoder focuser reports every night on one
scale. That assumption is what lets a filter used alone on its own nights be
compared with the rest at all. It is checked rather than trusted: residuals are
averaged per night and split into a smooth trend with date (slow settling, which
one drift parameter absorbs) and random jumps (a disturbed train, which only a
free level per night handles). Jumps are judged against the scatter a night mean
would show anyway from autofocus repeatability alone, and per-night levels are
introduced only when real movement is left over. See *Are focuser positions
comparable between nights?* above for what this data showed.

**Primary estimator -- the autofocus-position model.** After each filter change
the autofocus routine runs, and the position it settles on is a measurement of
best focus for that filter at that temperature. Frames are grouped into *focus
blocks* (consecutive subs sharing a filter, a focuser position and an equipment
configuration; the dead time that separates one block from the next is read off
the data, since a dither costs seconds and an autofocus run costs minutes), and
one regression is fitted across blocks:

    position = intercept + offset(filter)
               + k_within  x (temperature - its night's mean)
               + k_between x (that night's mean - overall mean)
               + drift x days            [when a trend with date is found]
               + per-night levels        [only when the zero point jumps]

Fitting temperature alongside the filter terms is what makes the offsets
absolute rather than relative to a drifting zero point. Splitting temperature
into its within-night and between-night parts matters: the two responses are
genuinely different, and a single slope is a compromise between them that pushes
the difference into the filter offsets. **k_within is the reported coefficient**,
because temperature compensation corrects focus as a night cools; k_between is
reported beside it, and a large gap between them means something that differs
between nights is tracking temperature without being caused by it. The thermal
lag is profiled on the same within-night variation, since the within-night slope
is strongly lag-dependent and estimating the two on different bases gives the
wrong coefficient.

Estimation is by a Huber M-estimator once there are enough blocks, so a single
failed autofocus run cannot drag the result; intervals are Student-t by default,
with a block-resampled bootstrap preferred once there are enough independent
nights for it to be stable.

**Secondary estimator -- the star-size model.** The position model inherits any
systematic bias in the autofocus routine itself. So measured star sizes are
modelled as `log HFD = c + gamma x log(airmass) + smooth(time) + eta(filter)`,
and a filter with significantly positive `eta` is softer than the conditions
explain. This is a flag, not a verdict: near best focus, an intrinsically softer
passband and a constant autofocus bias are indistinguishable. A deliberate focus
sweep breaks the degeneracy, and this tool fits one when the data contain it.

**Independence checks.** Star profiles are measured by two independent
implementations (ASTAP's C code and an in-process sep/photutils extractor) and
summarised by shared code, so disagreement is attributable to measurement rather
than statistics. Detections far below a frame's own stellar scale are rejected as
hot pixels and cosmic rays, which are bright enough to survive any
signal-to-noise cut. Star sizes are taken from a matched flux-percentile band
within each frame, so a filter sitting over a brighter sky is not mistaken for a
defocused one.
"""


def write_report(
    outdir: str,
    markdown: str,
    fit,
    recommendations,
    tables: dict[str, pd.DataFrame] | None = None,
    extra_json: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write the Markdown, HTML and JSON products plus every table as CSV."""
    os.makedirs(outdir, exist_ok=True)
    written: dict[str, str] = {}

    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w") as fh:
        fh.write(markdown)
    written["markdown"] = md_path

    json_path = os.path.join(outdir, "offsets.json")
    with open(json_path, "w") as fh:
        fh.write(to_json(fit, recommendations, extra_json))
    written["json"] = json_path

    csv_path = os.path.join(outdir, "offsets.csv")
    offsets_for_capture_software(fit).to_csv(csv_path, index=False)
    written["csv"] = csv_path

    for name, df in (tables or {}).items():
        if df is None or getattr(df, "empty", True):
            continue
        p = os.path.join(outdir, f"{name}.csv")
        df.to_csv(p, index=False)
        written[name] = p

    html_path = os.path.join(outdir, "report.html")
    with open(html_path, "w") as fh:
        fh.write(_html_wrap(markdown))
    written["html"] = html_path
    return written


def _html_wrap(markdown: str) -> str:
    """Self-contained HTML rendering of the Markdown report.

    Rendered with the `markdown` package when available and otherwise via a
    small built-in converter, so the HTML product never depends on an optional
    import.
    """
    try:
        import markdown as _md

        body = _md.markdown(markdown, extensions=["tables", "fenced_code"])
    except Exception:
        body = _minimal_markdown_to_html(markdown)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Filter offset report</title>
<style>
  :root {{
    color-scheme: light dark;
    --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
    --grid: #e4e3df; --accent: #2a78d6; --crit: #e34948; --warn: #eda100;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --grid:#383835;
             --accent:#3987e5; --crit:#e66767; --warn:#c98500; }}
  }}
  body {{ margin:0; background:var(--surface); color:var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
  main {{ max-width: 62rem; margin: 0 auto; padding: 2.5rem 1rem 6rem; }}
  h1,h2,h3 {{ line-height:1.25; margin:2rem 0 .6rem; }}
  h1 {{ font-size:1.9rem; }} h2 {{ font-size:1.4rem; border-bottom:1px solid var(--grid); padding-bottom:.35rem; }}
  h3 {{ font-size:1.1rem; color:var(--ink); }}
  p, li {{ color:var(--ink2); }}
  strong {{ color:var(--ink); }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.88em; }}
  pre {{ background:color-mix(in oklab, var(--surface) 92%, var(--ink)); padding:.9rem 1rem;
    border-radius:8px; overflow-x:auto; }}
  table {{ border-collapse: collapse; width:100%; margin:1rem 0; font-size:.88rem; display:block; overflow-x:auto; }}
  th, td {{ border-bottom:1px solid var(--grid); padding:.45rem .6rem; text-align:left; white-space:nowrap; }}
  th {{ color:var(--ink); font-weight:600; }}
  td {{ color:var(--ink2); }}
  img {{ max-width:100%; height:auto; border-radius:8px; margin:.5rem 0 1.5rem; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body><main>
{body}
</main></body></html>
"""


def _minimal_markdown_to_html(text: str) -> str:
    """Convert the subset of Markdown this module emits."""
    import html as _html
    import re

    out: list[str] = []
    in_table = False
    in_code = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(_html.escape(line))
            continue
        is_row = line.startswith("|") and line.endswith("|")
        if is_row:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append(
                "<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>"
            )
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if not line:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
        if m:
            out.append(f'<img alt="{_html.escape(m.group(1))}" src="{m.group(2)}">')
            continue
        if line.startswith("- "):
            out.append(f"<ul><li>{_inline(line[2:])}</li></ul>")
            continue
        out.append(f"<p>{_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_code:
        out.append("</pre>")
    # Merge adjacent single-item lists into one list.
    return "\n".join(out).replace("</ul>\n<ul>", "\n")


def _inline(text: str) -> str:
    import html as _html
    import re

    s = _html.escape(text)
    s = re.sub(r"!\[(.*?)\]\((.*?)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", s)
    return s


def _relative_asset(path: str, outdir: str | None) -> str:
    """Link an asset relative to the report, falling back to its basename."""
    if not outdir:
        return os.path.join("plots", os.path.basename(path))
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(outdir))
    except ValueError:  # different drives on Windows
        return os.path.join("plots", os.path.basename(path))
    if rel.startswith(".."):
        return os.path.join("plots", os.path.basename(path))
    return rel.replace(os.sep, "/")


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _median(df: pd.DataFrame, col: str) -> float | None:
    if df is None or col not in getattr(df, "columns", []):
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.median()) if len(v) else None
