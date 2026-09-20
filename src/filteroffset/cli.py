"""Command line interface."""

from __future__ import annotations

import os
import sys

import click

from . import __version__
from .config import Config


def _common_options(fn):
    opts = [
        click.option("--outdir", "-o", default="filteroffset_out", show_default=True,
                     help="Directory for the report, plots, tables and cache."),
        click.option("--config", "-c", "config_path", type=click.Path(exists=True),
                     help="YAML configuration file; command line options override it."),
        click.option("--cache", type=click.Path(),
                     help="Cache database path [default: <outdir>/cache.sqlite]."),
        click.option("--no-recursive", is_flag=True, help="Do not descend into subdirectories."),
        click.option("--backend", "backends", multiple=True,
                     type=click.Choice(["astap", "internal"]),
                     help="Star-profile backend; repeatable. Default: both."),
        click.option("--primary-backend", default=None,
                     type=click.Choice(["astap", "internal"]),
                     help="Backend whose measurements drive QC and the star-size model."),
        click.option("--metric", default=None,
                     help="Star-size column to model [default: hfd_median_bright]."),
        click.option("--reference", "reference", default=None,
                     help="Reference filter that gets offset 0 [default: best sampled]."),
        click.option("--step-microns", type=float, default=None,
                     help="Focuser travel per step, if known; enables the "
                          "depth-of-focus comparison."),
        click.option("--workers", type=int, default=None,
                     help="Parallel workers [default: chosen from available memory]."),
        click.option("--theme", type=click.Choice(["light", "dark"]), default=None,
                     help="Plot theme [default: light]."),
        click.option("--night-effects", type=click.Choice(["auto", "on", "off"]),
                     default=None,
                     help="Per-night intercepts. 'auto' (default) tests the "
                          "focuser's zero-point stability and only adds them if "
                          "nights jump; 'on' forces them (safe, but a filter that "
                          "never shares a night then has no measurable offset); "
                          "'off' always trusts absolute focuser positions."),
        click.option("--temp-from", type=click.Choice(["auto", "within_night", "pooled"]),
                     default=None,
                     help="Where the steps/C figure comes from [default: auto]."),
        click.option("--palette", type=click.Choice(["filter", "accessible"]),
                     default=None,
                     help="Plot colours: 'filter' draws each filter in its own "
                          "light's colour with narrowband dashed [default]; "
                          "'accessible' uses a colour-blind-validated palette."),
        click.option("--crop", type=int, default=None,
                     help="Centred crop side length for the internal backend; "
                          "0 means whole frame."),
        click.option("--all-image-types", is_flag=True,
                     help="Do not filter on IMAGETYP (by default only lights)."),
        click.option("--no-progress", is_flag=True, help="Suppress progress bars."),
        click.option("--quiet", "-q", is_flag=True, help="Only print the summary."),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _build_config(inputs, config_path, **kw) -> Config:
    cfg = Config.load(config_path) if config_path else Config()
    if inputs:
        cfg.inputs = [os.fspath(p) for p in inputs]
    if kw.get("outdir"):
        cfg.outdir = kw["outdir"]
    if kw.get("cache"):
        cfg.cache = kw["cache"]
    if kw.get("no_recursive"):
        cfg.recursive = False
    if kw.get("backends"):
        cfg.backends = list(kw["backends"])
    if kw.get("primary_backend"):
        cfg.primary_backend = kw["primary_backend"]
    if kw.get("metric"):
        cfg.metric = kw["metric"]
    if kw.get("reference"):
        cfg.model.reference_filter = kw["reference"]
    if kw.get("step_microns") is not None:
        cfg.step_microns = kw["step_microns"]
    if kw.get("workers") is not None:
        cfg.workers = kw["workers"]
    if kw.get("theme"):
        cfg.theme = kw["theme"]
    if kw.get("palette"):
        cfg.palette = kw["palette"]
    if kw.get("night_effects"):
        cfg.model.night_effects = kw["night_effects"]
    if kw.get("temp_from"):
        cfg.model.temp_from = kw["temp_from"]
    if kw.get("crop") is not None:
        cfg.profiles.crop = None if kw["crop"] <= 0 else int(kw["crop"])
    if kw.get("all_image_types"):
        cfg.image_types = None
    if kw.get("no_progress"):
        cfg.progress = False
    if not cfg.inputs:
        raise click.UsageError("Give at least one directory or FITS file to analyse.")
    return cfg


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="filteroffset")
def main() -> None:
    """Data-driven filter offsets and temperature compensation from your own subs.

    Point `filteroffset run` at a directory of light frames. It reads the FITS
    headers, recovers the autofocus structure, measures star profiles twice with
    independent code, fits the offsets and the thermal coefficient together, and
    writes a report that says whether the data support the numbers.
    """


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@_common_options
def run(inputs, config_path, quiet, **kw):
    """Full analysis: offsets, temperature compensation, plots and a report."""
    from .pipeline import run as run_pipeline

    cfg = _build_config(inputs, config_path, **kw)
    res = run_pipeline(cfg)
    _print_summary(res, quiet=quiet)
    if res.recommendations and not res.recommendations.trust_offsets:
        sys.exit(2)


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@_common_options
def fit(inputs, config_path, quiet, **kw):
    """Offsets only: skip star measurement, plots and report (fast)."""
    from .pipeline import run as run_pipeline

    cfg = _build_config(inputs, config_path, **kw)
    cfg.skip_measure = True
    cfg.backends = []
    res = run_pipeline(cfg, make_plots=False, make_report=False)
    _print_summary(res, quiet=quiet)


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option("--limit", default=20, show_default=True, help="Rows to display.")
@click.option("--keywords", is_flag=True,
              help="Show which FITS keyword supplied each mapped value.")
@click.option("--no-recursive", is_flag=True)
def inspect(inputs, limit, keywords, no_recursive):
    """Show what the FITS headers contain and how they were interpreted.

    Run this first when a dataset does not work: it reveals whether the focuser
    position and temperature keywords were found at all.
    """
    import pandas as pd

    from .headers import iter_fits_files, normalise_header, read_primary_header
    from .ingest import scan_directory
    from .profiles import available_backends

    if not inputs:
        raise click.UsageError("Give at least one directory or FITS file.")
    click.echo("Star-profile backends:")
    for name, status in available_backends().items():
        click.echo(f"  {name:10s} {status}")
    click.echo("")

    paths = list(iter_fits_files(inputs, recursive=not no_recursive))
    if not paths:
        raise click.ClickException("No FITS files found.")
    click.echo(f"{len(paths)} FITS file(s) found. First file: {paths[0]}")

    rec = normalise_header(paths[0])
    if keywords:
        click.echo("\nKeyword mapping for the first file:")
        for key, value in rec.values.items():
            src = rec.provenance.get(key, "derived")
            click.echo(f"  {key:16s} = {str(value)[:44]:44s} <- {src}")
        raw = read_primary_header(paths[0])
        unmapped = sorted(set(raw) - set(rec.provenance.values()))
        click.echo(f"\nUnmapped keywords present: {', '.join(unmapped) if unmapped else '(none)'}")

    scan = scan_directory(inputs, recursive=not no_recursive, image_types=None)
    df = scan.frames
    if df.empty:
        raise click.ClickException("No frames could be read.")
    essential = ["filter", "focus_pos", "ambient_temp", "focus_temp", "date_obs"]
    click.echo("\nCoverage of the keywords this analysis needs:")
    for col in essential:
        if col not in df.columns:
            click.echo(f"  {col:14s} MISSING entirely")
            continue
        n = int(df[col].notna().sum())
        mark = "ok  " if n == len(df) else ("PART" if n else "NONE")
        click.echo(f"  {col:14s} {mark} {n}/{len(df)} frames")

    cols = [c for c in ("file", "date_obs", "filter", "focus_pos", "ambient_temp",
                        "focus_temp", "exptime", "image_type") if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        click.echo("\n" + df[cols].head(limit).to_string(index=False))
    if len(df) > limit:
        click.echo(f"... {len(df) - limit} more")


@main.command("config")
@click.argument("path", type=click.Path())
def write_config(path):
    """Write a fully commented default configuration file."""
    cfg = Config()
    cfg.dump(path)
    click.echo(f"Wrote default configuration to {path}")
    click.echo("Edit it and pass with --config, or override individual keys on the CLI.")


def _print_summary(res, quiet: bool = False) -> None:
    fit = res.fit
    rec = res.recommendations
    click.echo("")
    click.echo("=" * 74)
    click.echo("  FILTER OFFSETS")
    click.echo("=" * 74)
    tbl = fit.offset_table()
    click.echo(f"  reference filter: {fit.reference}   (offset 0 by definition)")
    import math

    for row in tbl.itertuples():
        if row.is_reference:
            continue
        if not getattr(row, "identified", True) or not math.isfinite(row.offset_steps):
            pooled = getattr(row, "offset_pooled_steps", float("nan"))
            hint = (
                f"  (provisional {pooled:+.1f} if night zero points are stable)"
                if isinstance(pooled, float) and math.isfinite(pooled) else ""
            )
            click.echo(
                f"  {row.filter:>6s}  NOT IDENTIFIED - never shared a night with "
                f"{fit.reference}{hint}"
            )
            continue
        click.echo(
            f"  {row.filter:>6s}  {row.offset_steps:+8.2f} steps   "
            f"+/- {row.se_steps:5.2f}   95% CI [{row.ci_lo:+7.2f}, {row.ci_hi:+7.2f}]"
            f"   n={row.n_blocks}"
        )
    click.echo("")
    if math.isfinite(fit.temp_coeff):
        click.echo(
            f"  temperature compensation: {fit.temp_coeff:+.3f} +/- {fit.temp_coeff_se:.3f}"
            f" steps/C   95% CI [{fit.temp_coeff_ci[0]:+.2f}, {fit.temp_coeff_ci[1]:+.2f}]"
        )
    else:
        click.echo(
            "  temperature compensation: NOT ESTIMABLE - no usable temperature "
            "sensor in the headers"
        )
    click.echo(
        f"  autofocus repeatability:  {fit.resid_scale:.2f} steps per run "
        f"(R2 = {fit.r2:.3f}, {fit.n_blocks} blocks)"
    )
    click.echo("")
    counts = rec.counts()
    click.echo(
        f"  VERDICT: {rec.verdict.upper()}   "
        f"({counts['critical']} critical, {counts['warning']} warning, "
        f"{counts['ok']} passed)"
    )
    steps_state = (
        "not estimable" if not math.isfinite(fit.temp_coeff)
        else ("trustworthy" if rec.trust_temp_coeff else "provisional")
    )
    click.echo(
        f"  offsets: {'TRUSTWORTHY' if rec.trust_offsets else 'NOT TRUSTWORTHY AS-IS'}"
        f"   |   steps/C: {steps_state}"
    )
    if not quiet:
        click.echo("")
        for f in rec.findings:
            if f.severity in ("critical", "warning"):
                click.echo(f"  [{f.severity:8s}] {f.title}")
    if res.warnings:
        click.echo("")
        for w in res.warnings[:6]:
            click.echo(f"  ! {w}")
    if res.written:
        click.echo("")
        click.echo(f"  report:  {res.written.get('markdown', '-')}")
        click.echo(f"  html:    {res.written.get('html', '-')}")
        click.echo(f"  json:    {res.written.get('json', '-')}")
        click.echo(f"  plots:   {len(res.plots)} figures (+ matching CSVs)")
    click.echo("=" * 74)


if __name__ == "__main__":  # pragma: no cover
    main()
