# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] -- 2026-09-18

Independent audit, plus a fifth archive from a different imaging train. The
theme is **removing machinery that could not pay for itself**.

### Changed -- simplifications

- **The thermal lag is no longer fitted by default** (`thermal_lag_minutes`
  now defaults to `None`; `"fit"` remains available). Searching it is not free:
  shrinking the within-night temperature amplitude scales the slope up while
  leaving residuals nearly flat, so the grid search drifts toward longer time
  constants and larger |k|, and the reported standard error was conditional on
  the selected value rather than accounting for the selection. On simulated
  data with **one true coefficient and no lag at all**, fitting the lag inflated
  |k| by up to 22% and multiplied its scatter by 45x when the within-night
  temperature range was small. It changed only 2 of 5 real archives -- and
  removing it made the remaining archives *agree better* (Bodes and Methuselah
  now give -3.23 and -3.59 steps/C, against -3.23 and -4.20 before).
- **The within/between temperature split is now gated on having enough
  within-night range to estimate it** (`min_within_night_range_c`, default
  0.5 C), falling back to a single slope with a note. The split is kept
  because it genuinely protects the offsets (worth 1.9 steps on one archive)
  and fixed another archive's fit, but it needs real material to work on.
- **The bootstrap is no longer computed when it will be discarded.** It was
  being run and thrown away on 3 of 4 archives, costing 2-11 s each time.
- Duplicate specifications are excluded from the stability check, which was
  counting rows identical to the baseline as independent agreement.

### Fixed

- **A within-night regressor of numerical dust is no longer fitted.** Rigs that
  log one temperature per sequence make the within-night term identically zero;
  a relaxation artefact of ~1e-6 C survived the rank test and produced a
  coefficient **wrong by five orders of magnitude**, silently. The aliasing
  test is now scale-aware and honours its `protect` argument.
- **A single unparseable timestamp no longer crashes the fit.** One `NaT` made
  every later lagged temperature `NaN` and reached LAPACK as an
  uninterpretable `SVD did not converge`.
- **Huber standard errors** used the weighted-least-squares covariance, which
  treats estimated weights as known: measured coverage 91.8% for a nominal 95%.
  With the M-estimator correction, 93.4%.
- **The "every frame in this block failed" check could never fire**, because
  `float(n_pass or np.nan)` turned the only value it tests for -- zero -- into
  `NaN`. Such blocks entered the regression at full weight.
- **The automatic block-split threshold** fell back to a fixed 240 s in exactly
  the configuration it exists to avoid (a 45 s settle with a 90 s autofocus),
  and could be set by a single cloud gap. It now requires a real population of
  ordinary cadence below the gap, and declines to split on dead time at all
  when the data show no separation.
- Segmentation and model decisions (split threshold, temperature source,
  lag, night effects) were recorded in `DataFrame.attrs`, which the first merge
  discards; they are now in the report's provenance block.
- `temp_from` was labelled from a side effect rather than the decision.
- V-curve intervals were computed with the least-squares covariance while the
  fit used a robust loss (88% coverage); the fit now uses a linear loss.
- A constant focuser position now produces an explanatory note rather than a
  silent row of exact zeros.

### Tests

173 tests. Added the complement the audit identified as missing: **when a
single true coefficient generates the data, the estimator must return it** --
the test that would have caught both critical findings above. Also tightened a
coverage assertion from 0.80 to 0.86 (measured coverage is ~91%), and gave the
within-night lag profiler a direct test.

## [0.5.0] -- 2026-09-18

Robustness work driven by a fifth archive from an entirely different imaging
train (250 mm f/4.9 refractor, ASI6200MM, a focuser reporting ~500000 fine
steps, a short and noisy session, and no working temperature probe).

### Fixed

- **A temperature coefficient is no longer invented from a sensor that cannot
  provide one.** Temperature sources are now assessed and the reasoning is
  reported. Two failure modes are named explicitly: a probe that is not
  reporting (it writes a constant, very often exactly zero, so the column looks
  populated) and a *regulated* sensor -- a cooled camera's `CCD-TEMP` is held at
  its set point, so its wobble is control-loop noise. The previous fallback
  chain quietly used regulated CCD temperature and reported
  **k = +5.00 +/- 1.98 steps/C**, a physically backwards number fitted to cooler
  noise, in the headline verdict. When no source is usable the coefficient is
  simply not estimated; the offsets are unaffected, being differences between
  filters.
- **The robust estimator no longer collapses on quantised data.** Focuser
  positions are integers and autofocus often returns the same one repeatedly,
  so at least half the residuals coincide and the median absolute deviation
  falls to zero. Huber then judged every non-zero residual against a scale of
  nothing, weighted 4 of 13 blocks entirely out, and reported an autofocus
  repeatability of **0.00 steps with zero-width confidence intervals**. The
  robust scale is now floored at the granularity of the data itself, below
  which there is nothing to be robust about.
- `NaN` no longer leaks into the report, the capture-software instructions or
  the JSON (which now emits `null`, since JSON has no NaN).
- The focus-versus-temperature figure is skipped when there is no temperature,
  rather than drawn against an axis of floating-point noise.

### Verified unchanged

All four previously analysed archives produce identical offsets, coefficients
and block counts after these changes.

## [0.4.0] -- 2026-09-18

Correctness work driven by a third independent archive.

### Fixed

- **Temperature is now split into within-night and between-night terms.** The two
  responses differ materially (-4.2 vs -1.4 steps/C on one archive) and a single
  slope is a compromise that pushes the difference into the filter offsets.
  Splitting them cut apparent autofocus scatter from 12.4 to 3.6 steps and raised
  R^2 from 0.28 to 0.91 on that archive, and from 0.79 to 0.95 on another. This
  also replaces the two-stage "estimate k, then hold it fixed" construction with
  a single fit.
- **The zero-point test was measuring the wrong thing.** It fitted its own pooled
  temperature slope while the main model used a within-night one, so the two
  disagreed about what a night residual even was; temperature-correlated night
  structure (r = 0.91) was read as a 12-step wandering zero point and then
  averaged away into a "stable" verdict. It now uses the same split design.
- **Jump detection is now a variance component.** A night mean scatters by
  `within sigma / sqrtn` even with a perfectly stable zero point, so comparing the
  spread of night means directly against block-to-block scatter hid real
  movement on archives with many blocks per night and invented it on archives
  with few. An absolute floor of about one focuser step was added as well, since
  movement below that cannot be acted on.
- **Within-night scatter is pooled by degrees of freedom**, so a night with two
  blocks no longer dominates the estimate and hides real jumps behind it.
- **The block-split dead time is read off the data.** The fixed 240 s default was
  above the 222 s autofocus runs in one archive, which would have silently merged
  two consecutive same-filter runs into one block. The threshold is now placed in
  the empty gap between the settle and autofocus populations.
- **The focus-versus-temperature plot places each block against its own fitted
  line**, so whichever nuisance terms the model holds -- per-night levels, drift,
  the between-night temperature response -- are removed consistently. Previously
  points could sit parallel to but offset from their own fit.

### Changed

- The report is shorter: per-block tables are summarised per night (per-block
  detail remains in `blocks.csv`), the thermal-lag trace table is replaced by one
  sentence plus its existing plot, and flag listings are capped.
- The Method section now describes the undisturbed-imaging-train assumption, the
  test behind it, and the temperature split.

## [0.3.0] -- 2026-09-18

Treats absolute focuser positions as comparable across nights -- and verifies it.

### Changed

- **Per-night intercepts are no longer applied reflexively.** The documented
  assumption is now that the imaging train is undisturbed between sessions, so an
  absolute-encoder focuser puts every night on one scale. That between-night
  information is what ties a filter used alone on its own nights to the rest: on
  the development archive it is the difference between OIII having a measurable
  offset and having none. `--night-effects auto|on|off`.
- **The assumption is tested, not assumed.** `zero_point_drift_test` averages
  residuals per night and separates a *smooth trend with date* (slow settling --
  absorbed by one drift parameter, costing no offset its identifiability) from
  *random jumps between nights* (a disturbed train -- only per-night intercepts
  handle it). Jumps are measured after detrending and compared against
  within-night scatter. On the development archive: drift +0.145 steps/day, but
  detrended between-night scatter 1.74 steps against 2.64 within a night, so no
  jumps and the between-night information is used.
- **The temperature coefficient now comes from within-night variation only**,
  grouped by (night, filter), so nothing that differs between nights can reach
  it. The two estimators genuinely disagree on real data -- -4.2 vs -2.7
  steps/C -- and compensation corrects focus as the night cools, so the
  within-night figure is the right one. The pooled value is reported alongside
  and a large gap is flagged.
- **The thermal lag is profiled on the same within-night variation**, since the
  within-night slope is strongly lag-sensitive (-3.5 steps/C at zero lag against
  -4.3 at ninety minutes) and estimating the two on different bases gives the
  wrong coefficient. The development archive shows a clear 90-minute minimum.

### Added

- A linear `drift_term` (steps/day), reported with its standard error.
- A report section on focuser zero-point stability, and glossary entries for
  zero point, drift, jump ratio, within-night vs pooled, and anchor filter.
- A finding when nights span **different targets**: pointing-dependent flexure
  confounds between-night comparisons, which is invisible with per-night
  intercepts and material without them.
- `--temp-from` to override where the coefficient is estimated.

## [0.2.0] -- 2026-09-18

Hardening against real multi-night archives.

### Fixed

- **Offsets that cannot be measured are no longer invented.** With per-night
  intercepts, a filter's offset is only identifiable relative to another filter
  observed the *same night*. A filter that never shares a night is absorbed
  entirely by the night intercepts, and least squares silently returned a
  minimum-norm answer for it: finite, plausible, with a small standard error and
  no meaning. Such filters are now detected from the connected components of the
  night-filter graph and reported as *not identified*, with a clearly-labelled
  fallback fitted without night effects. This also removes the nonsense
  by-products -- variance inflation factors printed as `1000000000000.0`, and
  confidence intervals spanning 200 steps beside a 0.7-step standard error.
- **Figures now render in the HTML and Markdown reports.** Links kept only the
  file's basename while the figures live in `plots/`, so every image resolved to
  nothing once the report was opened from disk.
- **The offset-blind cross-check now conditions on night**, as the main fit
  does. Demeaning by filter alone let focuser zero-point shifts between nights
  enter as though they were a temperature response, so the "independent check"
  disagreed with the joint fit for a reason unrelated to filters (-2.93 against
  -4.23 steps/C on the development archive; they now agree at -4.16 vs -4.23).
- **Hot pixels and cosmic rays no longer corrupt star sizes.** They are bright,
  so no signal-to-noise cut removes them, and on long narrowband subs over a
  dark sky they outnumbered the real stars -- dragging one frame's median half-
  flux diameter to 0.87 px against a true 2.85 px. Detections far below the
  frame's own stellar scale are now rejected.
- **The fitted line is drawn against the temperature actually regressed on.**
  With a thermal lag applied, plotting against the raw header column put points
  and line on different variables, showing every series parallel to but offset
  from its own fit.
- Block splitting keys off dead time rather than wall-clock gap, so it means the
  same thing for 120 s and 600 s subs.
- `--elapsed-time` diagnostics and plot axes use time *within* each night; over a
  46-day campaign the global figure is 95% gaps between nights.

### Added

- **Filter-matched plot colours** (`--palette filter`, now the default): L grey,
  R red, G green, B blue, with narrowband dashed -- Ha red, SII orange, OIII
  teal. Marker shapes still differ per filter, so identity never rests on colour
  alone. `--palette accessible` restores the colour-blind-validated scheme.
- **Multi-night plots are faceted per night**, each on its own time axis.
- **A glossary** defining every symbol and statistic the report uses -- `k`, `F`,
  `p`, `VIF`, `R^2`, `SE`, confidence intervals, HFD, Cook's distance and the
  rest.
- Findings for unidentifiable offsets, single-informative-night designs, aliased
  terms, and hot-pixel contamination; the per-filter-slope finding now names
  filters observed on a single night, where a spurious slope difference is the
  expected symptom.
- A matched filter at detection, and reporting of `psf_scale_px`,
  `n_rej_toosmall` and `frac_rej_toosmall` per frame.
- Bootstrap intervals are discarded as degenerate when they collapse below the
  analytic standard error, which happens when the offsets rest on very few
  nights.

## [0.1.0] -- 2026-09-18

Initial release.

### Added

- **Model A**: joint regression of autofocus positions on filter and
  temperature, giving offsets that control for thermal drift rather than being
  measured against a moving zero point. Huber M-estimation, Student-t and
  block-bootstrap intervals, optional per-night intercepts and thermal lag.
- **Model B**: star-size model that removes the night's seeing trend and
  airmass, and flags filters that remain soft -- catching systematic autofocus
  bias, which Model A cannot see.
- **Two independent star-profile backends** (ASTAP `-extract` and an in-process
  `sep`/`photutils` extractor) summarised by shared code, so disagreement is
  attributable to measurement rather than statistics.
- **Design-adequacy diagnostics**: per-filter temperature coverage, variance
  inflation factors, condition number, and an offset-blind within-filter
  estimate of the temperature coefficient as an independent cross-check.
- **Autofocus verification** from inter-block dead time, and detection of
  capture-software temperature compensation running during acquisition.
- **Quality control** relative to the same filter on the same night: star
  counts, eccentricity with elongation-direction attribution, star-size
  outliers, background anomalies, saturation, backend disagreement.
- **V-curve fitting** with step-size calibration from defocus geometry, plus
  depth-of-focus tolerances; sweep detection that refuses to mistake separate
  autofocus runs for a sweep.
- Within-block focus-decay test with a predicted-versus-measured comparison.
- Specification-sensitivity analysis, including a deliberately confounded
  elapsed-time variant reported as a warning.
- 11 diagnostic plots, each with a matching CSV of the values plotted; Markdown,
  HTML and JSON reports; capture-software-ready offset export.
- SQLite caching of headers and measurements keyed on `(path, mtime, size)`, a
  header reader that touches only the front of each file, and memory-aware
  worker sizing -- for archives of hundreds of thousands of subs.
- CLI: `run`, `fit`, `inspect`, `config`.
