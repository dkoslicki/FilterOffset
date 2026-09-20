# filteroffset

**Data-driven filter offsets and temperature compensation for astrophotography
autofocus -- measured from your own subs, with the diagnostics to tell you
whether to believe them.**

Point it at a directory of light frames. It reads the FITS headers, recovers the
autofocus structure of each night, measures star profiles twice with independent
code, fits the filter offsets and the thermal coefficient together, and writes a
report that leads with whether the data can support the numbers at all.

```bash
filteroffset run /path/to/lights -o results
```

```
==========================================================================
  FILTER OFFSETS
==========================================================================
  reference filter: B   (offset 0 by definition)
       G     -5.63 steps   +/-  0.80   95% CI [  -7.51,   -3.74]   n=3
       R     -6.47 steps   +/-  0.79   95% CI [  -8.34,   -4.60]   n=3

  temperature compensation: -6.617 +/- 0.298 steps/C
  autofocus repeatability:  0.88 steps per run (R2 = 0.980, 11 blocks)

  VERDICT: USABLE WITH CAVEATS   (0 critical, 7 warning, 7 passed)
  offsets: TRUSTWORTHY   |   steps/C: provisional
==========================================================================
```

---

## Why this exists

The usual way to measure filter offsets is to autofocus on each filter in turn
and subtract. That works badly, because focus also drifts with temperature while
you are doing it -- so the difference you measure is part filter and part however
much the tube shrank between the two runs.

This tool treats the problem as a regression instead. Every autofocus run is one
measurement of best focus for *that filter at that temperature*, so with a
rotating filter order the two effects separate cleanly:

```
position = intercept + offset(filter)
           + k_within  x (temperature - its night's mean)
           + k_between x (that night's mean - overall mean)
           + drift x days          [only when a trend with date is found]
           + per-night levels      [only when the focuser zero point jumps]
```

Because temperature is in the same fit, the offsets are estimated **controlling
for** thermal drift rather than against a moving zero point. That is what makes
them absolute rather than an artefact of when you happened to measure.

The design only works if your observing order actually spread each filter across
the temperature range -- which is a property of your night, not of the fit. So
the tool measures that too, and says so plainly.

## What it will tell you that you did not ask for

- **Whether autofocus actually ran.** An autofocus run costs real time, so the
  dead time before each block is evidence. Blocks that show no pause are flagged
  rather than silently trusted.
- **Whether your capture software was moving the focuser underneath you.** If
  temperature compensation was already enabled, the recorded position is not one
  autofocus result and every number is biased. Detected automatically.
- **Whether the focuser's zero point is stable across nights**, separating slow
  settling from genuine jumps. See *Core assumption* below.
- **Whether an offset can be measured at all.** The alternative is a confident
  number with nothing behind it.
- **Whether the temperature sensor is telling the truth.** A probe that is not
  reporting usually writes a constant -- very often exactly zero -- rather than
  nothing, and a cooled camera's `CCD-TEMP` is held at its set point by the
  cooler. Neither can measure the optical train. Both are detected and named,
  and if nothing usable is left the coefficient is not estimated rather than
  fitted to noise. The offsets do not need temperature and are reported anyway.
- **Whether a filter is softer than conditions explain.** Autofocus can land off
  best focus for one filter; that error is invisible in the focuser positions
  and silently becomes part of the offset. A second model on measured star sizes
  catches it.
- **Whether a filter only *looks* soft.** Star sizes are taken from a matched
  flux-percentile band within each frame, and reported per brightness quartile,
  so light pollution and transparency differences cannot masquerade as defocus.
- **Whether hot pixels are being counted as stars.** They are bright, so no
  signal-to-noise cut removes them; on a long narrowband sub over a dark sky
  they can outnumber the real stars and drag the median star size below one
  pixel. Detections far below the frame's own stellar scale are rejected.

## Install

```bash
git clone https://github.com/dkoslicki/FilterOffset
cd FilterOffset
pip install -e ".[full]"
```

`[full]` adds `sep` and `photutils`, which enable the second, independent
star-profile backend. Via conda:

```bash
conda install -c conda-forge numpy scipy pandas matplotlib astropy click \
    pyyaml tqdm sep photutils
pip install -e . --no-deps
```

The statistics -- Huber M-estimation, variance inflation factors, the bootstrap
and the nested F-tests -- are implemented directly on numpy/scipy, so there is
no statsmodels dependency.

**ASTAP** (optional but recommended) provides the independent cross-check on
star profiles. Install from [hnsky.org](https://www.hnsky.org/astap.htm); the
tool looks for `astap_cli` on `$PATH`, at `/usr/local/bin/astap_cli`, or
wherever `$ASTAP_CLI` points.

Requires Python 3.10+.

## Usage

```bash
# Start here when a dataset does not work: see what your headers actually contain
filteroffset inspect /path/to/lights --keywords

# Full analysis: offsets, temperature compensation, plots, report
filteroffset run /path/to/lights -o results

# An archive of many nights; step size known, so include the depth-of-focus check
filteroffset run ~/astro/2026-*/lights -o results --step-microns 2.63

# Just the numbers, no image measurement (about a second, even on a big archive)
filteroffset fit /path/to/lights

# Pin every setting for reproducibility
filteroffset config myconfig.yaml
filteroffset run /path/to/lights --config myconfig.yaml
```

Useful options: `--reference L` to choose which filter gets offset 0,
`--backend astap` to use one measurement engine, `--crop 0` for whole-frame
measurement (slower; needed for tilt diagnostics), `--theme dark` for dark-mode
plots, `--palette accessible` to swap the filter-matched colours for a
colour-blind-validated set, `--night-effects on|off` to force or forbid
per-night intercepts, `--temp-from pooled` to use a single temperature slope.

Plots colour each filter as the light it passes -- L grey, R red, G green,
B blue, with narrowband dashed (Ha red, SII orange, OIII teal). Marker shapes
differ per filter as well, so identity never rests on colour alone; that matters
because red/green is exactly the pair the commonest form of colour blindness
cannot separate.

`filteroffset run` exits non-zero if the data cannot support the offsets, so it
can be used in a script.

## Output

```
results/
  report.md / report.html   the report, leading with the verdict, and
                            defining every symbol it uses in a glossary
  offsets.json              machine-readable results and findings
  offsets.csv               offsets ready to type into capture software
  findings.csv              every finding with severity and suggested action
  frames.csv, blocks.csv    per-frame and per-block tables with QC flags
  profiles.csv              star measurements from every backend
  frame_flags.csv           every quality-control flag, with its reason
  block_flags.csv           block-level flags
  sensitivity.csv           the same fit under every defensible specification
  config_used.yaml          exactly what was run
  cache.sqlite              header and measurement cache
  plots/                    figures, each with a matching .csv of its values
```

Twelve figures are defined; a given run produces those its data support (the
V-curve figure only appears when the data contain a focus sweep, the thermal-lag
figure only when a lag was profiled, and the focus-versus-temperature figure
only when there is a usable temperature).

## Scale

Header scanning reads only the 2880-byte header blocks at the front of each
file, so cost is independent of image size, and both headers and star
measurements are cached in SQLite keyed on `(path, mtime, size)`. Rescanning an
unchanged archive is a database query, not a file open. Worker counts are chosen
from *available memory* rather than core count, because one full frame from a
modern sensor can be several hundred MB once a backend has it in floating point.

Measured on the development machine (31 x 123 MB QHY600 frames):

| step | time |
|---|---|
| header read, per file | **0.065 ms** (about 7 s for 100,000 subs, single-threaded) |
| cached rescan of the whole set | 4 ms |
| `fit` (offsets only, no pixel data) | **0.9 s** |
| `run` (everything: two backends, 11 figures, report) | **6.6 s** |

Campaigns spanning weeks are handled as such: nights are detected from the gaps,
each gets its own panel in the plots, and time axes run from the start of each
night. On a 46-day archive with 51 hours of imaging, a single continuous axis
would be 95% empty.

## Core assumption: the imaging train is not disturbed between nights

**This tool assumes that nothing in the optical train changes between sessions**
-- the camera is not re-seated, spacers and filters are not swapped, the focuser
is not rehomed or reinstalled. Under that assumption an absolute-encoder focuser
reports positions on one fixed scale, so a position recorded after an OIII
autofocus run in July is directly comparable with one recorded after an Ha run
in September.

That assumption is worth a great deal, and it is the default. The alternative --
fitting a free level per night to absorb zero-point shifts -- is safe but throws
away all *between*-night information, and it is precisely that information which
ties a filter used alone on its own nights to the rest. On a real archive that
difference decided whether OIII had a measurable offset at all.

**The assumption is tested rather than taken on faith.** Residuals are averaged
per night and separated into:

- a **smooth trend with date** -- slow settling, or a seasonal effect the
  temperature model misses. One parameter absorbs it and no offset loses its
  identifiability.
- **random jumps between nights** -- what re-seating a camera or rehoming a
  focuser looks like. Only a free level per night handles this, and that is what
  costs offsets.

A night's mean residual scatters by `within-night sigma / sqrt(n)` even with a
perfectly stable zero point, so the test estimates the **variance component** --
how much night-to-night spread survives once that expected sampling scatter is
removed -- and requires it to exceed both a fraction of autofocus repeatability
and about one focuser step before calling it movement.

`--night-effects on` forces the conservative model; `off` always trusts absolute
positions; `auto` (default) decides from this test and reports what it found.

Two archives from the same rig gave opposite answers, which is the point of
testing: one showed a **+0.13 steps/day** drift with a zero-point variance
component of **0.0 steps** (per-night levels omitted, every filter measurable),
the other showed no drift but **5.1 steps** of genuine night-to-night movement
(per-night levels applied).

### The one requirement that remains

**An offset is a difference, so if per-night intercepts *are* needed, every
filter must share a night with the others.** A filter imaged only on nights
where it was the sole filter then has its offset absorbed completely by those
nights' levels -- there is no number to report, and a least-squares solve that
appears to produce one is producing an artefact. The tool reports which filters
are linked, which are orphaned, and which single pairing would join them. One
night of the orphaned filter alongside any filter from the other group, each
autofocused, fixes it permanently.

## Interpreting the two models

**Temperature enters as two separate terms** when there is enough within-night
range to support it (otherwise a single slope is fitted, with a note). The
within-night part and the between-night part get their own coefficients, because
the two responses are genuinely different -- on one archive **-4.2 steps/C
within nights against -1.4 between them**. A single slope is a compromise
between the two, and the difference it cannot express gets pushed into the
filter offsets: forcing one slope on that archive inflated the apparent
autofocus scatter from 3.6 to 12.4 steps and dropped R2 from 0.91 to 0.28.

**The reported coefficient is the within-night one**, because temperature
compensation corrects focus as a night cools. The between-night figure is
reported beside it, and a large gap is flagged.

**The thermal lag is not fitted by default.** It is available
(`thermal_lag_minutes: fit`) and reported in the sensitivity table, but
searching it by default was a net loss: shrinking the within-night temperature
amplitude scales the slope up while leaving residuals nearly flat, so the search
drifts toward larger |k|, and the quoted standard error ignored the fact that a
search had happened. On simulated data containing one true coefficient and no
lag at all, fitting the lag inflated |k| by up to 22% and multiplied its scatter
by 45 when the within-night temperature range was small.

**Model A (primary)** uses autofocus positions. It is robust, needs no pixel
data, and is what produces the offsets. Its blind spot is the autofocus routine
itself: a systematic bias is indistinguishable from a real offset.

**Model B (secondary)** uses measured star sizes, removing the night's seeing
trend and airmass, and asks whether any filter remains soft. It catches what
Model A cannot -- but near best focus, *an intrinsically softer passband and a
constant autofocus bias look identical*. When Model B flags a filter, the report
says so explicitly rather than guessing.

**The tie-breaker is a focus sweep.** Take a handful of frames at deliberately
spaced focuser positions in the suspect filter. If the V-curve minimum sits away
from where autofocus lands, autofocus is biased and the offset needs correcting;
if the minimum sits at the autofocus position but with a higher floor, the filter
is simply softer and the offset is already right. The tool detects sweeps in your
data and fits them automatically -- and it deliberately refuses to mistake several
separate autofocus runs across a night for a sweep, because fitting a V-curve to
those recovers the seeing trend and reports a confident best-focus position that
is pure artefact.

A sweep also calibrates **microns per step** from the V-curve slope, which beats
the motor's nominal figure: what matters is drawtube travel per step, and that
depends on the telescope the focuser is bolted to.

## A note on measuring HFD

Half-flux diameter is not aperture-independent. Real stellar profiles have
extended wings, so the measured HFD grows with the measurement aperture and never
plateaus -- on the development data, the same frame gives 2.43 px with an adaptive
aperture and 3.80 px at a fixed 10 px one. **There is no single "true" HFD**, only
a definition. Consequences, all handled:

- Two backends are expected to disagree on absolute scale, so they are compared
  after removing each one's own median. Only the frame-to-frame pattern matters.
- ASTAP's `-analyse` rounds the median HFD to one decimal place, too coarse for
  offsets that can be hundredths of a pixel. This tool uses `-extract` and
  computes statistics from the per-star CSV.
- The `HFD_MEDIAN` that ASTAP's `-extract` prints on stdout does **not** match
  the median of the CSV it writes (observed 21.5 vs 4.42 on one frame). The CSV
  is authoritative; the stdout value is recorded and ignored.
- ASTAP's HFD is strongly brightness-dependent (3.1 -> 5.0 px across flux
  quartiles on one frame) while the internal adaptive-aperture measurement is
  flat. Since filters have different throughput and field star colours, a naive
  median would be biased *between filters*. Hence the matched flux-percentile
  band.
- ASTAP writes its `-extract` CSV next to the input file and `-o` does not
  redirect it, so the tool runs ASTAP against symlinks in a scratch directory
  and never writes into your data directories.

## Supported capture software

Header keywords are aliased, so APT, N.I.N.A., SGP, MaxIm DL, Voyager and
Ekos/INDI all work. `FOCUSPOS`/`FOCPOS`/`FOCUSER`, `AMB-TEMP`/`FOCUSTEM`/
`TEMPERAT` and friends all map onto the same internal fields, and
`filteroffset inspect --keywords` shows exactly which keyword supplied each
value. Unusual keywords can be added via `extra_aliases` in the config without
touching code.

A focuser-mounted temperature probe (`FOCUSTEM`) is preferred over an ambient or
weather-station probe when both are present, because focus follows the
temperature of the glass and tube rather than the air.

## Requirements for a good measurement

| | Minimum | Good |
|---|---|---|
| Autofocus runs per filter | 3 | >= 6 |
| Nights | 1 | >= 3, with different thermal behaviour |
| Nights containing more than one filter | 1 | >= 3 |
| Every filter shares a night with the rest | required | required |
| Nights per filter | 1 | >= 2, spread across the campaign |
| Temperature range | ~1 C (offsets only) | >= 3 C for the coefficient |
| Filter order | rotating | rotating, revisited across the whole range |
| Temperature compensation | **off** | **off** |

Rotating the filter order is the single most important thing. Shooting all of B,
then all of R, then all of G confounds filter with temperature completely, and no
amount of modelling recovers it -- the tool will detect this and say so.

## Tunable constants

Every threshold the analysis depends on is a documented, configurable default
rather than a hidden constant. They are listed with their values and reasoning
in [docs/constants.md](docs/constants.md), and all of them can be overridden in
the YAML config.

## Contributing

```bash
pip install -e ".[full,dev]"
pytest              # 173 tests, about 40 s
ruff check src tests
```

Tests include synthetic-recovery checks (known offsets and coefficients must
come back out), confidence-interval coverage, positive *and* negative controls
for sweep detection and for defocus detection, degenerate-data cases (no usable
temperature sensor, quantised focuser positions, a fine-step focuser reporting
~500,000, unparseable timestamps, constant focus position), and an end-to-end
run over synthesised FITS frames.

## License

MIT -- see [LICENSE](LICENSE).
