"""Turn fit results and quality control into findings a person can act on.

Every finding carries a severity, what was observed, and what to do about it.
The intent is that the report answers "can I trust these offsets?" before it
answers "what are they?", because a confidently-quoted offset from a badly
conditioned night is worse than no offset at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .vcurve import focus_tolerances, implied_defocus_from_excess

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2, "ok": 3}


@dataclass
class Finding:
    severity: str
    code: str
    title: str
    detail: str
    action: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass
class Recommendations:
    findings: list[Finding] = field(default_factory=list)
    verdict: str = "unknown"
    headline: str = ""
    trust_offsets: bool = False
    trust_temp_coeff: bool = False

    def add(self, *args, **kwargs) -> None:
        self.findings.append(Finding(*args, **kwargs))

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame([f.as_row() for f in self.findings])
        if df.empty:
            return df
        df["_o"] = df["severity"].map(SEVERITY_ORDER).fillna(9)
        return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(SEVERITY_ORDER, 0)
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out


# --------------------------------------------------------------------------- #
def build_recommendations(
    fit,
    blocks_qc: pd.DataFrame,
    frames_qc: pd.DataFrame,
    frame_flags: pd.DataFrame,
    block_flags: pd.DataFrame,
    hfd_excess: dict[str, Any] | None = None,
    sensitivity: pd.DataFrame | None = None,
    step_size_um: float | None = None,
    lag_report: dict[str, Any] | None = None,
    decay: dict[str, Any] | None = None,
    min_blocks_per_filter: int = 3,
    good_blocks_per_filter: int = 6,
    min_temp_span: float = 3.0,
) -> Recommendations:
    """Assemble the findings list."""
    rec = Recommendations()
    diag = fit.diagnostics or {}
    per_filter = diag.get("per_filter", {})
    n_blocks = int(fit.n_blocks)
    n_nights = int(diag.get("n_nights", 1))
    temp_span = float(diag.get("temp_span", float("nan")))

    has_temp = _temperature_findings(rec, fit, diag)
    _design_findings(
        rec, fit, diag, per_filter, n_blocks, n_nights, temp_span,
        min_blocks_per_filter, good_blocks_per_filter, min_temp_span,
        has_temp=has_temp,
    )
    _zero_point_findings(rec, fit, diag)
    _pointing_findings(rec, fit, diag)
    _identifiability_findings(rec, fit, diag)
    _confounding_findings(rec, fit, diag)
    _offset_findings(rec, fit, step_size_um, frames_qc)
    _temp_findings(rec, fit, diag, sensitivity, lag_report, n_nights, temp_span)
    _autofocus_findings(rec, blocks_qc, block_flags)
    _quality_findings(rec, frames_qc, frame_flags)
    _excess_findings(rec, fit, hfd_excess, step_size_um, frames_qc)
    _contamination_findings(rec, frames_qc)
    _decay_findings(rec, decay)
    _stability_findings(rec, sensitivity)

    counts = rec.counts()
    rec.trust_offsets = counts["critical"] == 0
    rec.trust_temp_coeff = rec.trust_offsets and n_nights > 1 and temp_span >= min_temp_span
    if counts["critical"]:
        rec.verdict = "not usable as-is"
    elif counts["warning"] > 2:
        rec.verdict = "usable with caveats"
    else:
        rec.verdict = "usable"
    rec.headline = _headline(fit, rec, n_blocks, n_nights, temp_span)
    return rec


def _headline(fit, rec: Recommendations, n_blocks, n_nights, temp_span) -> str:
    parts = [
        f"{n_blocks} focus blocks over {n_nights} night(s), "
        f"{temp_span:.1f} C of temperature range"
    ]
    off = ", ".join(
        f"{f}{fit.offsets[f]:+.1f}" for f in sorted(fit.offsets) if f != fit.reference
    )
    parts.append(f"offsets vs {fit.reference}: {off} steps")
    parts.append(
        f"k = {fit.temp_coeff:+.2f} steps/C" if np.isfinite(fit.temp_coeff)
        else "no temperature coefficient (no usable sensor)"
    )
    parts.append(f"verdict: {rec.verdict}")
    return "; ".join(parts)


def _temperature_findings(rec, fit, diag) -> bool:
    """Report the thermal regressor, or the lack of one. Returns True if usable."""
    if diag.get("has_temperature", True) and np.isfinite(fit.temp_coeff):
        return True
    reasons = diag.get("temperature_sources") or {}
    detail = (
        "No header temperature can serve as a thermal regressor, so no "
        "steps/\u00b0C figure is reported. The filter offsets are unaffected: "
        "they are differences between filters and do not need temperature."
    )
    if reasons:
        detail += " Sources considered - " + "; ".join(
            f"**{k}**: {v}" for k, v in reasons.items()
        ) + "."
    rec.add(
        "warning", "no_temperature_source",
        "No usable temperature sensor, so no temperature compensation",
        detail,
        "Record an ambient or focuser-mounted probe in the FITS headers. A "
        "focuser probe is best: it reads the tube rather than the air. Note "
        "that a cooled camera's CCD-TEMP is held at its set point and can "
        "never serve this purpose.",
    )
    return False


def _design_findings(
    rec, fit, diag, per_filter, n_blocks, n_nights, temp_span,
    min_blocks, good_blocks, min_temp_span, has_temp: bool = True,
) -> None:
    thin = {f: v["n_blocks"] for f, v in per_filter.items() if v["n_blocks"] < min_blocks}
    modest = {
        f: v["n_blocks"]
        for f, v in per_filter.items()
        if min_blocks <= v["n_blocks"] < good_blocks
    }
    if thin:
        rec.add(
            "critical", "too_few_blocks",
            "Some filters have too few autofocus runs",
            ", ".join(f"{f}: {n} block(s)" for f, n in sorted(thin.items()))
            + f" - fewer than the {min_blocks} needed to separate a real offset "
            "from autofocus run-to-run scatter.",
            "Collect more autofocus runs for these filters, ideally on nights with "
            "different temperatures.",
        )
    if modest:
        rec.add(
            "warning", "modest_block_count",
            "Filter offsets rest on few autofocus runs",
            ", ".join(f"{f}: {n} blocks" for f, n in sorted(modest.items()))
            + f". The offsets are identified but their uncertainty is dominated by "
            f"autofocus repeatability (measured here as {fit.resid_scale:.2f} steps "
            "per run).",
            f"Aim for >= {good_blocks} blocks per filter to halve the interval.",
        )
    if n_nights == 1:
        rec.add(
            "warning", "single_night",
            "Everything comes from one night",
            "A focuser's zero point can shift between nights (re-seating the camera, "
            "a power cycle, a different starting position, backlash take-up). One "
            "night cannot show whether these offsets are stable, and it cannot "
            "separate temperature from time-of-night.",
            "Repeat on two or three further nights. The tool adds per-night "
            "intercepts automatically once more than one night is present, which "
            "makes the offsets immune to those zero-point shifts.",
        )
    if has_temp and np.isfinite(temp_span) and temp_span < min_temp_span:
        rec.add(
            "warning", "narrow_temperature_range",
            "Temperature range is narrow for a compensation fit",
            f"The night spanned {temp_span:.1f} C. The filter offsets do not need a "
            "wide range, but the temperature coefficient does.",
            f"For a dependable steps/C figure, gather data spanning >= {min_temp_span:.0f} C, "
            "which usually means several nights rather than one.",
        )


def _zero_point_findings(rec, fit, diag) -> None:
    """Report whether focuser positions are comparable between nights."""
    d = getattr(fit, "drift_test", None)
    n_nights = int(diag.get("n_nights", 1))
    if n_nights <= 1:
        return
    if d is None or not getattr(d, "tested", False):
        rec.add(
            "warning", "zero_point_untested",
            "Focuser stability across nights could not be verified",
            (getattr(d, "reason", "") or "not enough nights to test").capitalize()
            + ". Positions are being treated as directly comparable across "
            "nights, which is correct only if the imaging train was not "
            "disturbed between sessions - no camera re-seated, no spacers "
            "changed, no focuser rehomed.",
            "If anything in the train was touched between sessions, re-run with "
            "--night-effects on; the offsets will then rest on within-night "
            "comparisons only.",
        )
        return

    used_night = bool(diag.get("night_effects"))
    if d.verdict == "jumps":
        rec.add(
            "warning", "zero_point_jumps",
            "The focuser's zero point moves between nights",
            f"After removing any trend, nights still differ by "
            f"{d.between_sd_detrended:.2f} steps against {d.within_sd:.2f} "
            "within a night - jumps larger than autofocus repeatability. That is "
            "what re-seating a camera, changing spacers or a focuser losing its "
            "home looks like. Per-night intercepts have been included, which "
            "protects the offsets but means any filter never sharing a night "
            "with the others cannot be measured.",
            "Check whether anything in the imaging train was disturbed between "
            "these sessions.",
        )
        return

    drift_txt = ""
    if d.verdict == "drift":
        drift_txt = (
            f" There is a slow drift of {d.drift_per_day:+.3f} steps/day "
            f"({d.drift_total_steps:+.1f} steps across {d.span_days:.0f} days, "
            f"p = {d.drift_p:.3f}), consistent with gradual settling rather than "
            "anything being moved. One drift parameter absorbs it, which costs "
            "no offset its identifiability - unlike a free level per night."
        )
    rec.add(
        "ok", "zero_point_stable",
        "Focuser positions are comparable across nights",
        f"Tested on {', '.join(d.anchor_filters)} across {d.n_nights} nights: "
        f"night-to-night scatter after detrending is {d.between_sd_detrended:.2f} "
        f"steps, against {d.within_sd:.2f} steps within a night. The between-night "
        "scatter is no larger than autofocus repeatability itself, so there is "
        "nothing for per-night intercepts to absorb and the between-night "
        "information can be used." + drift_txt
        + (
            " Per-night intercepts were nonetheless forced on by configuration."
            if used_night else ""
        ),
        "",
    )


def _pointing_findings(rec, fit, diag) -> None:
    """Warn when between-night comparisons span different targets.

    Comparing focus positions across nights is only clean if the telescope was
    pointing at comparable places in the sky.  Any flexure in the optical train
    depends on where the tube is pointed, so merging campaigns on different
    targets puts a pointing term into what looks like a filter or temperature
    effect.  This matters only when between-night information is actually being
    used; with per-night intercepts it is absorbed anyway.
    """
    blocks = getattr(fit, "blocks", None)
    if blocks is None or blocks.empty or "object" not in blocks.columns:
        return
    if diag.get("night_effects"):
        return
    if int(diag.get("n_nights", 1)) < 2:
        return
    per_night = blocks.groupby("session")["object"].agg(
        lambda x: str(x.dropna().iloc[0]) if x.notna().any() else ""
    )
    targets = sorted({t for t in per_night if t})
    if len(targets) < 2:
        return
    counts = per_night.value_counts().to_dict()
    listed = ", ".join(f"{t} ({counts.get(t, 0)} night(s))" for t in targets)
    rec.add(
        "warning", "mixed_targets",
        "Nights are split across different targets",
        f"{listed}. Because focuser positions are being compared between nights, "
        "any flexure that depends on where the telescope points enters the "
        "comparison as though it were a filter or temperature effect. A filter "
        "observed on only one target then carries that target's pointing with it.",
        "Prefer analysing one target's campaign at a time, or make sure each "
        "filter appears on more than one target so the pointing term averages "
        "out. Re-running with --night-effects on removes the issue entirely, at "
        "the cost of any offset that needs between-night information.",
    )


def _identifiability_findings(rec, fit, diag) -> None:
    """Report offsets the design cannot determine at all."""
    ident = diag.get("identifiability") or {}
    not_id = list(getattr(fit, "not_identified", []) or [])
    comps = getattr(fit, "components", []) or ident.get("components", [])
    n_info = ident.get("n_informative_nights")

    if not_id:
        pooled = getattr(fit, "offsets_pooled", {}) or {}
        pooled_txt = ""
        vals = [f"{f} {pooled[f]:+.1f}" for f in not_id if f in pooled and np.isfinite(pooled[f])]
        if vals:
            pooled_txt = (
                " Fitting without per-night intercepts does produce a number for "
                f"them ({', '.join(vals)} steps), but that rests on the focuser's "
                "zero point being identical on every night - which is exactly the "
                "assumption the per-night intercepts exist to avoid. Treat it as a "
                "provisional starting value, not a measurement."
            )
        groups = " and ".join("{" + ", ".join(c) + "}" for c in comps)
        rec.add(
            "critical", "offsets_not_identified",
            f"No offset can be measured for: {', '.join(not_id)}",
            f"{', '.join(not_id)} was only ever imaged on nights where no other "
            "filter was used. A filter offset is a *difference*, so it can only be "
            "measured against another filter observed the same night - otherwise it "
            "is indistinguishable from that night's focuser zero point. Your filters "
            f"fall into separate groups that are never linked: {groups}. Offsets "
            "within a group are sound; between groups they do not exist in this "
            "data." + pooled_txt,
            "One night is enough to fix this: image "
            f"{not_id[0]} and any filter from the other group back to back, with "
            "an autofocus run for each, on the same night. That single link ties "
            "the groups together and makes every offset measurable.",
        )
    elif len(comps) > 1:
        rec.add(
            "warning", "filters_in_separate_groups",
            "Filters fall into groups that never share a night",
            "Offsets are comparable within a group but were only tied together "
            "because night intercepts are switched off.",
            "Image one filter from each group on the same night to link them.",
        )

    if n_info is not None and diag.get("n_nights", 1) > 1:
        if n_info == 0:
            rec.add(
                "critical", "no_informative_nights",
                "No night contains more than one filter",
                "Every night used a single filter, so there is nothing to compare "
                "within a night and no offset can be measured with night intercepts "
                "in place.",
                "Rotate at least two filters within a night, each with its own "
                "autofocus run.",
            )
        elif n_info == 1:
            rec.add(
                "warning", "single_informative_night",
                "Every offset rests on a single night",
                f"Only 1 of {diag.get('n_nights')} nights contains more than one "
                "filter, so all the offsets come from that night alone. The quoted "
                "intervals describe the scatter *within* that night and cannot "
                "capture whether the same answer would come back on another night.",
                "Repeat a multi-filter night two or three more times. Agreement "
                "across nights is the only evidence that these offsets are stable.",
            )


def _confounding_findings(rec, fit, diag) -> None:
    """Report the filter-vs-temperature separation, the central worry."""
    if not diag.get("has_temperature", True):
        return  # nothing to be confounded with
    vif = diag.get("vif", {})
    filt_vif = {k: v for k, v in vif.items() if k.startswith("filter[")}
    worst = max(filt_vif.values()) if filt_vif else float("nan")
    wf = diag.get("within_filter_k", {}) or {}
    k_within = wf.get("k_pooled")
    k_within_se = wf.get("k_pooled_se")
    corrs = {
        f: v.get("corr_with_temp") for f, v in (diag.get("per_filter") or {}).items()
    }
    max_corr = max((abs(v) for v in corrs.values() if np.isfinite(v)), default=float("nan"))

    if not np.isfinite(worst) and filt_vif:
        aliased = [k.replace("filter[", "").rstrip("]")
                   for k, v in filt_vif.items() if not np.isfinite(v)]
        rec.add(
            "critical", "filter_term_aliased",
            "A filter term is an exact duplicate of other terms",
            f"{', '.join(aliased)} carries no information of its own once the "
            "other terms are in the model, so no coefficient exists for it. This "
            "is reported rather than silently solved, because a least-squares "
            "routine will happily return a finite-looking number for it.",
            "See the identifiability finding above for what to observe.",
        )
    elif np.isfinite(worst) and worst > 5.0:
        rec.add(
            "critical", "filter_temp_confounded",
            "Filters are confounded with temperature",
            f"The largest variance inflation factor on a filter term is {worst:.1f} "
            "(anything above ~5 means the filter indicator and temperature are "
            "carrying nearly the same information). The reported offsets may be "
            "absorbing thermal drift rather than measuring the filters.",
            "Interleave the filters more finely, so each filter is revisited across "
            "the whole temperature range within a night.",
        )
    else:
        detail = (
            f"Largest filter-term VIF is {worst:.2f} and the strongest "
            f"filter/temperature correlation is {max_corr:.2f}, both effectively "
            "negligible - the rotating filter order spread every filter across the "
            "night's temperature range."
        )
        if k_within is not None and np.isfinite(fit.temp_coeff):
            detail += (
                f" As an independent check, using *only* within-filter comparisons "
                f"(which cannot see filter offsets at all) gives k = "
                f"{k_within:+.2f} +/- {k_within_se:.2f} steps/C against "
                f"{fit.temp_coeff:+.2f} +/- {fit.temp_coeff_se:.2f} from the joint "
                "fit. The agreement confirms the separation is real."
            )
        rec.add(
            "ok", "filter_temp_separated",
            "Filter offsets are cleanly separated from temperature",
            detail,
            "",
        )
    cond = diag.get("condition_number")
    if cond is not None and np.isfinite(cond) and cond > 30:
        rec.add(
            "warning", "ill_conditioned",
            "Design matrix is poorly conditioned",
            f"Condition number {cond:.0f}. Coefficients will be sensitive to small "
            "changes in the data.",
            "Add blocks that break the collinearity, especially repeat visits to the "
            "same filter at different temperatures.",
        )


def _offset_findings(rec, fit, step_size_um, frames_qc) -> None:
    noise = float(fit.resid_scale) if np.isfinite(fit.resid_scale) else float("nan")
    insignificant = []
    for f in sorted(fit.offsets):
        if f == fit.reference:
            continue
        lo, hi = fit.offsets_ci.get(f, (np.nan, np.nan))
        if np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi:
            insignificant.append(f)
    if insignificant:
        rec.add(
            "info", "offset_not_significant",
            "Some offsets are statistically indistinguishable from zero",
            f"{', '.join(insignificant)} (vs reference {fit.reference}). Their "
            "confidence intervals include zero.",
            "You may leave these at 0 in the capture software until more data "
            "narrows the interval; setting a small unreliable offset is worse than "
            "setting none.",
        )
    rec.add(
        "info", "af_repeatability",
        "Autofocus repeatability",
        f"Residual scatter about the fitted model is {noise:.2f} steps per "
        f"autofocus run (R^2 = {fit.r2:.3f} on {fit.n_blocks} blocks). That figure "
        "is the floor on how precisely any single autofocus run can be trusted.",
        "",
    )

    # Physical significance, via the depth of focus.
    fr = frames_qc
    fratio = _first_finite(fr, "focal_ratio")
    pixel = _first_finite(fr, "xpixsz")
    seeing = _median_finite(fr, "fwhm_median")
    tol = focus_tolerances(
        fratio or float("nan"), seeing_fwhm_px=seeing, pixel_um=pixel
    )
    span = max(
        (abs(v) for f, v in fit.offsets.items() if f != fit.reference), default=0.0
    )
    seeing_tol = tol.get("seeing_total_um")
    diff_tol = tol.get("diffraction_total_um")
    if step_size_um and np.isfinite(step_size_um):
        um = span * step_size_um
        if seeing_tol and np.isfinite(seeing_tol):
            frac = um / seeing_tol
            sev = "info" if frac > 0.5 else "ok"
            rec.add(
                sev, "offset_significance",
                "Are the offsets large enough to matter?",
                f"The largest offset is {span:.1f} steps = {um:.1f} um at "
                f"{step_size_um:.2f} um/step. The seeing-limited depth of focus here "
                f"is about {seeing_tol:.1f} um total (diffraction limit "
                f"{diff_tol:.1f} um), so the offset is {100*frac:.0f}% of the zone.",
                "Worth applying." if frac > 0.5 else
                "Small compared with the depth of focus; applying it is harmless but "
                "will not visibly change your stars.",
            )
    else:
        rec.add(
            "info", "step_size_unknown",
            "Focuser step size in microns is not known",
            "Capture software only needs the offsets in steps, so this does not "
            f"block anything. For reference the depth of focus is about "
            f"{seeing_tol:.1f} um (seeing-limited) / {diff_tol:.1f} um (diffraction) "
            "at this focal ratio, and the report includes a table converting the "
            "step offsets to microns across plausible step sizes."
            if seeing_tol and np.isfinite(seeing_tol) else
            "Supply --step-microns to enable the depth-of-focus comparison.",
            "A focus sweep (several frames at deliberately spaced positions) lets "
            "this tool calibrate microns-per-step directly from the V-curve slope, "
            "which beats relying on the motor's nominal figure.",
        )


def _temp_findings(rec, fit, diag, sensitivity, lag_report, n_nights, temp_span) -> None:
    if not np.isfinite(fit.temp_coeff) or not diag.get("has_temperature", True):
        return
    k, se = fit.temp_coeff, fit.temp_coeff_se
    rel = abs(se / k) if k else float("inf")
    if rel > 0.5:
        rec.add(
            "warning", "temp_coeff_imprecise",
            "Temperature coefficient is poorly determined",
            f"k = {k:+.2f} +/- {se:.2f} steps/C, a {100*rel:.0f}% relative "
            "uncertainty.",
            "Do not enable temperature compensation on this number yet.",
        )
    if n_nights == 1:
        # Quantify the single-night time/temperature collinearity explicitly.
        corr = None
        if sensitivity is not None and "corr_temp_elapsed" in getattr(
            sensitivity, "columns", []
        ):
            vals = pd.to_numeric(sensitivity["corr_temp_elapsed"], errors="coerce").dropna()
            if len(vals):
                corr = float(vals.iloc[0])
        detail = (
            "On a single night, temperature falls steadily with time, so anything "
            "else that drifts with time (tube shrinkage still catching up, cable "
            "flexure, mirror settling, altitude) is indistinguishable from a "
            "temperature effect."
        )
        if corr is not None:
            detail += (
                f" Here temperature and elapsed time correlate at r = {corr:+.3f}; "
                "a fit that adds a plain time drift alongside temperature explains "
                "the data better while making the steps/C figure meaningless for "
                "any other night."
            )
        rec.add(
            "warning", "temp_time_collinear",
            "Temperature and time-of-night cannot be separated",
            detail,
            "Gather nights with different thermal behaviour - a night that warms, a "
            "night that plateaus, a night that drops fast. Across several such "
            "nights the true temperature response separates from time-of-night "
            "drift, and this tool will then report a trustworthy steps/C.",
        )
    if lag_report:
        best_rss = lag_report.get("best_tau_by_rss")
        best_aicc = lag_report.get("best_tau_by_aicc")
        r0, rb = lag_report.get("rss_at_zero"), lag_report.get("rss_best")
        if best_rss and np.isfinite(best_rss) and best_rss > 0 and r0 and rb and r0 > 0:
            improvement = 100.0 * (1.0 - rb / r0)
            chosen = (
                f"{best_aicc:.0f} min" if best_aicc else "none (no lag applied)"
            )
            rec.add(
                "info", "thermal_lag",
                "The optics appear to lag the air temperature",
                f"Smoothing the ambient temperature with a time constant of about "
                f"{best_rss:.0f} minutes reduces the unexplained scatter by "
                f"{improvement:.0f}%, which is the signature of the tube and mirror "
                "following the air with a delay rather than instantly. After the "
                f"model-selection penalty for the extra parameter the chosen lag is "
                f"{chosen}, so this is suggestive rather than established at the "
                "current number of blocks.",
                "A focuser-mounted temperature probe reading the tube itself would "
                "remove the guesswork. Failing that, more blocks will settle whether "
                "the lag is real.",
            )
    within = getattr(fit, "temp_coeff_pooled", float("nan"))
    if (
        getattr(fit, "temp_from", "") == "within_night"
        and np.isfinite(within) and np.isfinite(fit.temp_coeff)
    ):
        gap = abs(fit.temp_coeff - within)
        if gap > 3 * max(fit.temp_coeff_se, 1e-6):
            rec.add(
                "info", "within_vs_pooled_k",
                "Within-night and between-night temperature slopes disagree",
                f"Using only variation inside nights gives k = {fit.temp_coeff:+.2f} "
                f"steps/C; also using contrasts between nights gives "
                f"{within:+.2f}. The reported figure is the within-night one, "
                "because that is the variation temperature compensation actually "
                "corrects. The gap means something that differs between nights - "
                "how long the rig had been cooling, the season, where the target "
                "sat in the sky - is tracking temperature without being caused by "
                "it, which would bias a pooled estimate toward zero.",
                "No action needed for compensation. If you want the two to "
                "converge, a focuser-mounted probe reading the tube itself removes "
                "most of the ambiguity.",
            )

    slopes = fit.per_filter_slopes or {}
    if slopes.get("tested") and not slopes.get("needed"):
        rec.add(
            "ok", "shared_temp_slope",
            "One temperature coefficient serves all filters",
            f"Allowing each filter its own steps/C does not improve the fit "
            f"(F = {slopes['F']:.2f}, p = {slopes['p_value']:.2f}). That is the "
            "expected result: focus drift is dominated by the tube and mirror, which "
            "all filters share. It also means a single compensation coefficient plus "
            "fixed offsets is the right model for your capture software.",
            "",
        )
    elif slopes.get("needed"):
        # Filters observed on only one night cannot distinguish a temperature
        # response from a time-of-night drift, so a "different slope" for such a
        # filter is very often that confounding rather than thermal physics.
        per_filter = diag.get("per_filter", {}) or {}
        thin = sorted(
            f for f, d in (slopes.get("delta_slopes") or {}).items()
            if per_filter.get(f, {}).get("n_nights", 99) <= 1
        )
        detail = (
            f"F = {slopes['F']:.2f}, p = {slopes['p_value']:.3f} (F compares the "
            "model with per-filter slopes against the one with a shared slope; "
            "small p means the extra slopes fit better). This is physically "
            "unusual - the thermal expansion driving focus drift comes from the "
            "tube and mirror, which every filter shares."
        )
        if thin:
            detail += (
                f" Note that {', '.join(thin)} appear on only a single night each. "
                "Within one night, temperature falls steadily with time, so a "
                "single-night filter cannot tell a temperature response apart "
                "from any other drift; a spurious slope difference is the "
                "expected symptom rather than a surprise."
            )
        rec.add(
            "warning", "per_filter_temp_slope",
            "Filters appear to need different temperature coefficients",
            detail,
            "Spread each filter over several nights before believing it. Most "
            "capture software cannot express per-filter coefficients anyway, so "
            "the practical answer remains one shared coefficient.",
        )


def _autofocus_findings(rec, blocks_qc, block_flags) -> None:
    if blocks_qc is None or blocks_qc.empty:
        return
    if "af_likely" in blocks_qc.columns:
        vals = blocks_qc["af_likely"]
        n_yes = int((vals == True).sum())  # noqa: E712
        n_no = int((vals == False).sum())  # noqa: E712
        if n_no:
            rec.add(
                "critical", "autofocus_missing",
                "Some blocks show no sign of an autofocus run",
                f"{n_no} block(s) began without enough dead time for an autofocus "
                "sequence, so their focuser position may have been inherited from a "
                "previous filter. Such blocks contaminate the offsets directly.",
                "Exclude them (they are already flagged), or confirm from your "
                "capture logs whether autofocus actually ran.",
            )
        if n_yes:
            rec.add(
                "ok", "autofocus_verified",
                "An autofocus run precedes every analysed block",
                f"{n_yes} of {len(blocks_qc)} blocks show a clear pause before the "
                "first frame, consistent with a filter change plus autofocus. This "
                "was verified from frame timestamps rather than assumed.",
                "",
            )
    if "temp_comp_active" in blocks_qc.columns and bool(
        blocks_qc["temp_comp_active"].any()
    ):
        rec.add(
            "critical", "temp_comp_enabled",
            "The capture software was moving the focuser during runs",
            "The focuser position changed inside a continuous filter run, which means "
            "temperature compensation was active. The recorded position is then a "
            "moving target rather than one autofocus result, and both the offsets and "
            "the steps/C figure are biased.",
            "Turn temperature compensation OFF while gathering calibration data, then "
            "re-run. Compensation is what you are trying to measure.",
        )
    else:
        rec.add(
            "ok", "temp_comp_off",
            "No focuser motion within blocks",
            "The focuser held position throughout each run, so every recorded "
            "position is a clean autofocus result and nothing was being compensated "
            "underneath the measurement.",
            "",
        )


def _quality_findings(rec, frames_qc, frame_flags) -> None:
    if frames_qc is None or frames_qc.empty:
        return
    n = len(frames_qc)
    n_fail = int((~frames_qc["qc_pass"]).sum()) if "qc_pass" in frames_qc else 0
    if n_fail:
        by_check = (
            frame_flags[frame_flags["severity"] == "fail"]["check"].value_counts().to_dict()
            if not frame_flags.empty else {}
        )
        summary = ", ".join(f"{k} ({v})" for k, v in by_check.items())
        sev = "warning" if n_fail / n < 0.25 else "critical"
        rec.add(
            sev, "frames_rejected",
            f"{n_fail} of {n} frames failed quality control",
            f"Reasons: {summary}. These are excluded from the star-size analysis; "
            "note that they do not affect the offsets from autofocus positions, which "
            "depend on the focuser reading rather than on image quality.",
            "Inspect the flagged frames. Repeated elongation at low altitude usually "
            "means guiding or wind rather than focus.",
        )
    if not frame_flags.empty:
        ecc = frame_flags[frame_flags["check"] == "elongated_stars"]
        if len(ecc) >= 2:
            rec.add(
                "warning", "guiding_degraded",
                "Star elongation grows late in the session",
                f"{len(ecc)} frame(s) show elongated stars with a shared elongation "
                "direction, which is the signature of tracking or guiding error (or "
                "wind), not of focus. Trailing inflates measured star size and will "
                "masquerade as defocus if not excluded.",
                "Check guiding logs and balance for the low-altitude end of the "
                "session; consider stopping before the target drops that low.",
            )
        dis = frame_flags[frame_flags["check"] == "backend_disagreement"]
        if not dis.empty:
            rec.add(
                "info", "backend_disagreement",
                "The two star-profile measurements disagree on some frames",
                f"{len(dis)} frame(s). Disagreement is concentrated on frames that "
                "already fail other checks, which is the expected behaviour: both "
                "measurements are reliable on clean frames and diverge on damaged "
                "ones.",
                "",
            )


def _excess_findings(rec, fit, hfd_excess, step_size_um, frames_qc) -> None:
    """Report filters whose stars are larger than conditions explain."""
    if not hfd_excess:
        return
    agree: dict[str, list[tuple[str, float, float]]] = {}
    for label, res in hfd_excess.items():
        if res is None or not getattr(res, "excess_frac", None):
            continue
        for f, val in res.excess_frac.items():
            if f == res.reference:
                continue
            p = res.p_values.get(f, np.nan)
            agree.setdefault(f, []).append((label, float(val), float(p)))

    for f, entries in sorted(agree.items()):
        sig = [e for e in entries if np.isfinite(e[2]) and e[2] < 0.01 and e[1] > 0.03]
        if len(sig) < 2:
            continue
        vals = [e[1] for e in sig]
        lo, hi = 100 * min(vals), 100 * max(vals)
        backends = sorted({e[0].split("/")[0] for e in sig})
        hfd0 = _median_finite(frames_qc, "hfd_median_bright") or float("nan")
        implied = ""
        if step_size_um and np.isfinite(step_size_um):
            from .vcurve import geometric_slope_px_per_micron

            pixel = _first_finite(frames_qc, "xpixsz")
            fratio = _first_finite(frames_qc, "focal_ratio")
            slope = geometric_slope_px_per_micron(fratio, pixel) * step_size_um
            d = implied_defocus_from_excess(float(np.mean(vals)), hfd0, slope)
            if np.isfinite(d["implied_defocus_steps"]):
                implied = (
                    f" If the excess were purely defocus, it would correspond to about "
                    f"{d['implied_defocus_steps']:.0f} steps of focus error (sign "
                    "not recoverable from near-focus data)."
                )
        rec.add(
            "warning", f"star_size_excess_{f}",
            f"{f} frames are softer than the observing conditions explain",
            f"After removing the night's seeing trend and airmass, {f} stars are "
            f"{lo:.0f}-{hi:.0f}% larger than the reference filter "
            f"({fit.reference}), consistent across {len(sig)} metric/backend "
            f"combinations ({', '.join(backends)}) at p < 0.01." + implied +
            " Two explanations fit this equally well and near-focus data cannot "
            "separate them: (a) autofocus systematically lands off best focus in "
            f"{f}, in which case the offset above inherits that error; or (b) {f} is "
            "intrinsically softer - plausible causes include residual chromatic "
            "correction in the corrector and, in back-illuminated CMOS sensors, "
            "deeper absorption of red photons broadening the point spread function.",
            f"Settle it with a deliberate focus sweep in {f}: take a handful of "
            "frames at spaced focuser positions either side of the autofocus result. "
            "If the V-curve minimum sits away from where autofocus lands, it is "
            "(a) and the offset needs correcting; if the minimum sits at the "
            "autofocus position but with a higher floor, it is (b) and the offset is "
            "already right. This tool fits that sweep automatically.",
        )


def _contamination_findings(rec, frames_qc) -> None:
    """Report detections discarded as too compact to be stars."""
    if frames_qc is None or "frac_rej_toosmall" not in getattr(frames_qc, "columns", []):
        return
    frac = pd.to_numeric(frames_qc["frac_rej_toosmall"], errors="coerce").dropna()
    if frac.empty:
        return
    med = float(frac.median())
    worst = float(frac.max())
    heavy = int((frac > 0.30).sum())
    if med < 0.15 and heavy == 0:
        return
    sev = "info" if med < 0.35 else "warning"
    rec.add(
        sev, "hot_pixel_contamination",
        "Many detections were discarded as too small to be stars",
        f"A median of {100*med:.0f}% of detections per frame (worst frame "
        f"{100*worst:.0f}%; {heavy} frame(s) above 30%) were smaller than the "
        "frame's own stellar scale and were rejected as hot pixels or cosmic "
        "rays. This is common on long narrowband subs, where the sky is dark "
        "enough that sensor defects stand out at high significance - they are "
        "bright, so no signal-to-noise cut removes them, only a size test does. "
        "Left in, they drag the median star size below one pixel and corrupt "
        "every star-size comparison.",
        "Nothing to do for the offsets, which come from focuser positions. If "
        "you want cleaner star statistics, applying a bad-pixel map or dark "
        "calibration before analysis will remove most of these at source.",
    )


def _decay_findings(rec, decay) -> None:
    """Report the within-block focus decay test."""
    if not decay or not decay.get("tested"):
        return
    slope = decay["slope_px_per_frame"]
    se, p = decay["se"], decay.get("p_value")
    pred = decay.get("predicted_px_per_frame")
    n = decay["n_frames"]
    common = (
        f"Measured change in star size per frame after the autofocus run: "
        f"{slope:+.4f} +/- {se:.4f} px (p = {p:.2f}, {n} frames in "
        f"{decay['n_blocks']} blocks, estimated with block fixed effects so the "
        "night's seeing trend cannot leak in)."
    )
    if decay.get("significant"):
        rec.add(
            "info", "within_block_decay",
            "Focus measurably decays after each autofocus run",
            common + " The first sub after autofocus really is the sharpest here.",
            "This is what temperature compensation is for; applying the fitted "
            "steps/C should flatten it.",
        )
        return
    detail = common + " No decay is detectable."
    if pred is not None and np.isfinite(pred):
        detail += (
            f" Nor should it be: with the fitted temperature coefficient and the "
            f"measured drift of {decay.get('mean_dT_per_frame', float('nan')):+.3f} "
            f"C per frame, the focuser only goes "
            f"{decay.get('predicted_defocus_steps_per_frame', float('nan')):+.2f} "
            f"steps out of focus per frame, and because the V-curve is quadratic at "
            f"its minimum that predicts only {pred:.5f} px of growth - far below the "
            "measurement noise."
        )
    rec.add(
        "ok", "within_block_decay",
        "No focus decay within a block, as expected",
        detail,
        "Good news for this analysis: every sub in a block is equally usable, so "
        "the autofocus position can be treated as one clean measurement for the "
        "whole block rather than only for its first frame.",
    )


def _stability_findings(rec, sensitivity) -> None:
    if sensitivity is None or sensitivity.empty:
        return
    df = sensitivity[~sensitivity.get("variant", pd.Series(dtype=str)).astype(str).str.contains(
        "CONFOUNDED", na=False
    )]
    if "duplicate_of_baseline" in df.columns:
        df = df[~df["duplicate_of_baseline"].fillna(False).astype(bool)]
    off_cols = [c for c in df.columns if c.startswith("offset_")]
    spreads = {}
    for c in off_cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) >= 3 and v.abs().max() > 0:
            spreads[c.replace("offset_", "")] = float(v.max() - v.min())
    k = pd.to_numeric(df.get("k_steps_per_C"), errors="coerce").dropna()
    k_spread = float(k.max() - k.min()) if len(k) >= 3 else float("nan")
    if spreads:
        worst_f, worst = max(spreads.items(), key=lambda kv: kv[1])
        n_distinct = int(len(df))
        detail = (
            f"Refitting under {n_distinct} distinct specifications (robust vs least "
            "squares, with and without a thermal lag, block-start vs block-mean "
            "temperature, with and without an altitude term; specifications that "
            "collapse onto the baseline are not counted) moves the offsets by at "
            f"most {worst:.2f} steps (worst: {worst_f})"
        )
        if np.isfinite(k_spread):
            detail += (
                f", while the temperature coefficient moves by {k_spread:.2f} "
                "steps/C across the same set"
            )
        sev = "ok" if worst < 1.0 else "warning"
        if sev == "ok":
            detail += (
                ". The offsets are therefore a property of the data; the "
                "temperature coefficient is partly a property of the modelling "
                "choices."
            )
        else:
            detail += (
                f". A spread of {worst:.2f} steps is comparable to the autofocus "
                "repeatability itself, so treat it as the practical uncertainty "
                "on the offsets - wider than any single fit's confidence "
                "interval suggests."
            )
        rec.add(
            sev, "specification_stability",
            "Offsets are stable across model specifications"
            if sev == "ok" else "Offsets move appreciably across model specifications",
            detail,
            "" if sev == "ok" else
            "Treat the spread above as the practical uncertainty, not the "
            "confidence interval from any single fit.",
        )


def _first_finite(df: pd.DataFrame | None, col: str) -> float | None:
    if df is None or col not in getattr(df, "columns", []):
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else None


def _median_finite(df: pd.DataFrame | None, col: str) -> float | None:
    if df is None or col not in getattr(df, "columns", []):
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.median()) if len(v) else None
