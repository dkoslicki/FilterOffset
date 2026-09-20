"""End-to-end orchestration: a directory of subs in, offsets and a report out."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .blocks import segment_blocks, summarise_blocks
from .cache import FrameCache
from .config import Config
from .ingest import scan_directory
from .measure import measure_frames
from .model import (
    choose_temp_column,
    fit_hfd_excess,
    fit_offsets,
    sensitivity_analysis,
    thermal_lag_report,
    within_block_decay,
)
from .qc import check_blocks, check_frames
from .recommend import build_recommendations
from .report import build_markdown, write_report
from .vcurve import detect_sweeps, fit_vcurve


@dataclass
class RunResult:
    config: Config
    frames: pd.DataFrame = field(default_factory=pd.DataFrame)
    blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    frames_qc: pd.DataFrame = field(default_factory=pd.DataFrame)
    blocks_qc: pd.DataFrame = field(default_factory=pd.DataFrame)
    frame_flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    block_flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit: Any = None
    recommendations: Any = None
    sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    hfd_excess: dict[str, Any] = field(default_factory=dict)
    lag_report: dict[str, Any] = field(default_factory=dict)
    sweeps: dict[str, Any] = field(default_factory=dict)
    decay: dict[str, Any] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)
    written: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def run(cfg: Config, make_plots: bool = True, make_report: bool = True) -> RunResult:
    """Run the whole analysis."""
    res = RunResult(config=cfg)
    os.makedirs(cfg.outdir, exist_ok=True)
    cache_path = cfg.cache or os.path.join(cfg.outdir, "cache.sqlite")

    with FrameCache(cache_path) as cache:
        scan = scan_directory(
            cfg.inputs,
            cache=cache,
            recursive=cfg.recursive,
            extra_aliases=cfg.extra_aliases or None,
            image_types=tuple(cfg.image_types) if cfg.image_types else None,
            progress=cfg.progress,
        )
        if scan.errors:
            res.warnings.append(
                f"{len(scan.errors)} file(s) could not be read; first: {scan.errors[0]}"
            )
        if scan.frames.empty:
            raise SystemExit(
                f"No usable light frames found under: {', '.join(cfg.inputs)}"
            )
        res.frames = segment_blocks(scan.frames, cfg.blocks)
        _require_columns(res.frames)

        if not cfg.skip_measure and cfg.backends:
            m = measure_frames(
                res.frames,
                backends=tuple(cfg.backends),
                params=cfg.profiles,
                cache=cache,
                workers=cfg.workers,
                workdir=cfg.outdir,
                progress=cfg.progress,
            )
            res.profiles = m.profiles
            if m.failures:
                res.warnings.append(
                    f"{len(m.failures)} measurement failure(s); first: {m.failures[0]}"
                )

    # ------------------------------------------------------------------- QC
    res.frames_qc, res.frame_flags = check_frames(
        res.frames, res.profiles, cfg.qc,
        primary_backend=cfg.primary_backend, metric=cfg.metric,
    )
    res.blocks = summarise_blocks(res.frames)
    res.blocks_qc, res.block_flags = check_blocks(res.blocks, res.frames_qc, cfg.qc)

    # ---------------------------------------------------------------- model
    usable = res.blocks_qc[res.blocks_qc["qc_pass"]] if "qc_pass" in res.blocks_qc \
        else res.blocks_qc
    if len(usable) < 3:
        raise SystemExit(
            f"Only {len(usable)} usable focus block(s) after quality control; "
            "at least 3 are needed. See the flags in the output directory."
        )
    res.fit = fit_offsets(usable, cfg.model)
    res.sensitivity = sensitivity_analysis(usable, cfg.model)

    # thermal lag diagnostic (meaningless without a thermal regressor)
    try:
        if not res.fit.diagnostics.get("has_temperature", True):
            raise RuntimeError("no usable temperature source")
        spec = cfg.model
        tcol = choose_temp_column(usable, spec)
        src = f"{tcol}_first" if spec.temp_at_block_start and f"{tcol}_first" in usable else tcol
        raw_temp = pd.to_numeric(usable[src], errors="coerce").to_numpy(dtype=float)
        res.lag_report = thermal_lag_report(
            usable, raw_temp, res.fit.reference, spec,
            use_night=bool(res.fit.diagnostics.get("night_effects")),
        )
    except RuntimeError:
        pass
    except Exception as exc:
        res.warnings.append(f"thermal lag diagnostic unavailable: {exc}")

    # ------------------------------------------------------- star-size model
    res.hfd_excess = _run_hfd_excess(res, cfg)

    # ------------------------------------------------------------- V-curves
    res.sweeps = _run_sweeps(res, cfg)

    # ------------------------------------------ within-block focus decay test
    try:
        from .vcurve import geometric_slope_px_per_micron

        eq = res.frames_qc.iloc[0] if not res.frames_qc.empty else {}
        vslope = None
        if cfg.step_microns:
            g = geometric_slope_px_per_micron(
                _num(eq.get("focal_ratio")) or float("nan"),
                _num(eq.get("xpixsz")) or float("nan"),
            )
            vslope = g * cfg.step_microns if np.isfinite(g) else None
        res.decay = within_block_decay(
            res.frames_qc, metric=cfg.metric,
            temp_column=(res.fit.temp_column or "ambient_temp").replace("_first", ""),
            temp_coeff=res.fit.temp_coeff, slope_px_per_step=vslope,
        )
    except Exception as exc:
        res.warnings.append(f"within-block decay test unavailable: {exc}")

    # ---------------------------------------------------------- suggestions
    res.recommendations = build_recommendations(
        res.fit, res.blocks_qc, res.frames_qc, res.frame_flags, res.block_flags,
        hfd_excess=res.hfd_excess, sensitivity=res.sensitivity,
        step_size_um=cfg.step_microns, lag_report=res.lag_report,
        decay=res.decay,
    )

    # -------------------------------------------------------------- outputs
    if make_plots:
        res.plots = _make_plots(res, cfg)
    if make_report:
        md = build_markdown(
            res.fit, res.recommendations, res.frames_qc, res.blocks_qc,
            res.frame_flags, res.block_flags, res.profiles, res.hfd_excess,
            res.sensitivity, res.lag_report, res.plots,
            step_size_um=cfg.step_microns,
            outdir=cfg.outdir,
            inputs={
                "inputs": ", ".join(cfg.inputs),
                "frames": len(res.frames),
                "blocks": len(res.blocks_qc),
                "backends": ", ".join(cfg.backends),
                "metric": cfg.metric,
                "reference_filter": res.fit.reference,
                "profile_params_key": cfg.profiles.key(),
                # The most consequential preprocessing decision: what dead time
                # counts as an autofocus run rather than a dither.
                "block_split_dead_time_s": _first_value(
                    res.frames, "block_split_dead_time_s"
                ),
                "temperature_source": res.fit.temp_column or "none usable",
                "thermal_lag_minutes": res.fit.thermal_lag_minutes,
                "night_effects": res.fit.diagnostics.get("night_effects"),
                "temperature_from": res.fit.temp_from,
            },
        )
        res.written = write_report(
            cfg.outdir, md, res.fit, res.recommendations,
            tables={
                "frames": res.frames_qc,
                "blocks": res.blocks_qc,
                "profiles": res.profiles,
                "frame_flags": res.frame_flags,
                "block_flags": res.block_flags,
                "sensitivity": res.sensitivity,
                "findings": res.recommendations.table(),
            },
        )
        cfg.dump(os.path.join(cfg.outdir, "config_used.yaml"))
    return res


def _require_columns(frames: pd.DataFrame) -> None:
    missing = [
        c for c in ("filter", "focus_pos", "date_obs")
        if c not in frames.columns or frames[c].isna().all()
    ]
    if missing:
        raise SystemExit(
            "These FITS headers lack the keywords this analysis needs: "
            + ", ".join(missing)
            + ". Filter offsets require a filter name, a focuser position and a "
            "timestamp in every light frame. Use `filteroffset inspect` to see "
            "what your headers do contain, and `extra_aliases` in the config to "
            "map non-standard keyword names."
        )


def _run_hfd_excess(res: RunResult, cfg: Config) -> dict[str, Any]:
    """Fit the star-size model for every available metric/backend pair."""
    out: dict[str, Any] = {}
    if res.profiles is None or res.profiles.empty:
        return out
    keep = [
        c for c in ("path", "filter", "block", "date_obs", "airmass", "altitude",
                    "qc_pass")
        if c in res.frames_qc.columns
    ]
    merged = res.profiles.merge(res.frames_qc[keep], on="path", how="left")
    metrics = [
        m for m in (cfg.metric, "hfd_median", "fwhm_median")
        if m in merged.columns
    ]
    for backend in sorted(merged["backend"].dropna().unique().tolist()):
        for metric in dict.fromkeys(metrics):
            try:
                fitted = fit_hfd_excess(
                    merged, metric=metric, backend=backend,
                    reference=res.fit.reference,
                )
            except Exception:
                continue
            if fitted and fitted.excess_frac:
                out[f"{backend}/{metric}"] = fitted
    return out


def _run_sweeps(res: RunResult, cfg: Config) -> dict[str, Any]:
    """Fit V-curves wherever the data actually contain a focus sweep."""
    out: dict[str, Any] = {}
    if res.profiles is None or res.profiles.empty:
        return out
    prim = res.profiles[res.profiles["backend"] == cfg.primary_backend]
    if prim.empty:
        prim = res.profiles
    cols = [c for c in ("path", "filter", "focus_pos", "session", "qc_pass")
            if c in res.frames_qc.columns]
    merged = prim.merge(res.frames_qc[cols], on="path", how="left")
    if "qc_pass" in merged:
        merged = merged[merged["qc_pass"].fillna(True)]
    found = detect_sweeps(merged)
    if found.empty:
        return out
    eq = res.frames_qc.iloc[0] if not res.frames_qc.empty else {}
    fratio, pixel = _num(eq.get("focal_ratio")), _num(eq.get("xpixsz"))
    metric = cfg.metric if cfg.metric in merged.columns else "hfd_median"
    for row in found.itertuples():
        g = merged[merged["filter"] == getattr(row, "filter", None)]
        if "session" in merged.columns and hasattr(row, "session"):
            g = g[g["session"] == row.session]
        pos = pd.to_numeric(g["focus_pos"], errors="coerce").to_numpy(dtype=float)
        hfd = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
        fitted = fit_vcurve(
            pos, hfd, filter_name=str(getattr(row, "filter", "")),
            focal_ratio=fratio, pixel_um=pixel,
        )
        out[str(getattr(row, "filter", ""))] = {
            "fit": fitted, "positions": pos, "hfd": hfd,
        }
    return out


def _make_plots(res: RunResult, cfg: Config) -> list[str]:
    from . import plots as P

    ctx = P.PlotContext(
        theme=P.Theme.get(cfg.theme),
        outdir=os.path.join(cfg.outdir, "plots"),
        palette=getattr(cfg, "palette", "filter"),
    )
    ctx.assign(sorted(res.frames_qc["filter"].dropna().unique().tolist()))
    made: list[str] = []
    steps = [
        ("focus_vs_temperature", lambda: P.plot_focus_vs_temperature(ctx, res.fit)),
        ("offsets_forest", lambda: P.plot_offsets_forest(ctx, res.fit)),
        ("design_balance", lambda: P.plot_design_balance(ctx, res.fit)),
        ("timeline", lambda: P.plot_timeline(ctx, res.fit, res.frames_qc)),
        ("residuals", lambda: P.plot_residuals(ctx, res.fit)),
        ("hfd_within_block", lambda: P.plot_hfd_within_block(ctx, res.frames_qc, cfg.metric)),
        ("hfd_vs_flux", lambda: P.plot_hfd_vs_flux(ctx, res.profiles, res.frames_qc)),
        ("backend_agreement", lambda: P.plot_backend_agreement(ctx, res.profiles, cfg.metric)),
        ("hfd_excess", lambda: P.plot_hfd_excess(ctx, res.hfd_excess)),
        ("qc_timeline", lambda: P.plot_qc_timeline(ctx, res.frames_qc)),
        ("thermal_lag", lambda: P.plot_thermal_lag(ctx, res.lag_report)),
        ("vcurves", lambda: P.plot_vcurves(ctx, res.sweeps)),
    ]
    for label, fn in steps:
        try:
            path = fn()
            if path:
                made.append(path)
        except Exception as exc:
            res.warnings.append(f"plot {label!r} failed: {type(exc).__name__}: {exc}")
    return made


def _first_value(df: pd.DataFrame, col: str):
    if df is None or col not in getattr(df, "columns", []):
        return None
    vals = df[col].dropna()
    return vals.iloc[0] if len(vals) else None


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
