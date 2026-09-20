# Tunable constants, their values, and why

Every threshold the analysis depends on is listed here with its value, where it
lives, and the reasoning behind it. They fall into three honest categories:

- **Physical** -- derived from optics, geometry or the resolution of the data.
  These are the ones you should not need to touch.
- **Statistical** -- a conventional significance level, or a sample size below
  which an estimator is known to misbehave.
- **Empirical** -- chosen by looking at real data. These are defaults, not
  laws. If your rig differs, change them.

Anything in a `Params`/`Spec` dataclass can be set in the YAML config
(`filteroffset config myconfig.yaml` writes a commented template). A few
constants are function arguments rather than config fields; those are marked,
and are the ones most worth promoting to config if they ever bite you.

---

## Star detection and measurement (`profiles/base.py`, `profiles/internal.py`)

| Constant | Value | Kind | Why |
|---|---|---|---|
| `detect_sigma` | 5.0 | Empirical | Standard source-detection threshold in sigma above background. Lower finds more faint stars but admits noise; 5 sigma is the usual compromise and matches SExtractor practice. |
| `min_area` | 5 px | Empirical | Minimum connected pixels above threshold. Below about 5, single hot pixels and cosmic rays dominate. |
| `filter_fwhm` | 3.0 px | Empirical | FWHM of the Gaussian matched filter applied before thresholding, as SExtractor does by default. Roughly the stellar FWHM of a well-sampled system. Cut spurious detections by ~30% on test data without shifting measured sizes. |
| `crop` | 3000 px | Practical | Side of the centred square the in-process backend reads. Keeps peak memory to a few hundred MB on a 61-megapixel frame, and the centre is the cleanest probe anyway because field curvature, tilt and coma inflate corner HFD without being filter-dependent focus. `--crop 0` reads the whole frame. |
| `center_radius_frac` | 0.40 | Physical | Statistics suffixed `_center` use stars inside this fraction of the half-diagonal, for the same reason as above. |
| `bright_lo_pct` / `bright_hi_pct` | 70 / 97 | Empirical | The "bright" sample is a flux-percentile band *within each frame*. A band rather than a threshold because ASTAP's SNR and sep's flux/flux-error are not on the same scale, so any absolute cut would silently mean different things per backend. The upper bound excludes saturated cores, whose HFD is inflated. |
| `bright_min_stars` | 20 | Statistical | Below this the band's median is too noisy; the code falls back to all stars. |
| `snr_min` | 10.0 | Empirical | A junk filter only. Each backend interprets it in its own units, which is why no science statistic depends on it. |
| `saturation_frac` | 0.90 | Physical | Stars whose peak exceeds this fraction of full well are rejected: a flat-topped core inflates the half-flux radius. |
| `full_well_adu` | 65535 | Physical | Assumed full well when the header does not say. Correct for 16-bit output. |
| `hfd_max` | 40 px | Empirical | Discards satellites, galaxies and merged blends. **Caveat:** a deliberate focus sweep can legitimately produce stars larger than this, so raise it when analysing sweeps. |
| `min_hfd_frac_of_psf` | 0.55 | Empirical | Reject detections smaller than this fraction of the frame's own stellar scale. Hot pixels and cosmic rays are *bright*, so no signal-to-noise cut removes them; only a size test does. On a 600 s narrowband sub over a dark sky they outnumbered real stars and dragged the median HFD to 0.87 px against a true 2.85 px. |
| `psf_scale_pct` | 90 | Empirical | That stellar scale is the median HFD of detections above this flux percentile -- the brightest detections are reliably real stars. Self-calibrating, so it adapts across sampling: the floor came out at 1.14 px on a 3.1 arcsec/px rig and 1.4-2.3 px on a 1.46 arcsec/px rig. |
| `hfd_min_abs` | 1.0 px | Physical | Absolute backstop. No real star on any correctly sampled system has a half-flux diameter below one pixel. |
| `relative_floor_backends` | `("internal",)` | Empirical | The *relative* floor is valid only where measured size is independent of brightness for real stars. It is for the adaptive-aperture internal extractor; it is not for ASTAP, whose HFD rises steeply with flux by construction. Applying it to ASTAP was measured to discard ~15% of genuine faint stars from a clean frame. ASTAP also needs it less, since it validates stellar profiles during detection. |
| `min_stars` | 25 | Empirical | Below this a frame's measurement is not trusted. |

## Block segmentation (`blocks.py`)

| Constant | Value | Kind | Why |
|---|---|---|---|
| `session_gap_hours` | 8.0 | Practical | Gap that separates one night from the next. Longer than any within-night pause, shorter than any inter-night gap. |
| `block_split_dead_time_s` | `"auto"` | -- | Dead time (excluding the exposure) after which a fresh autofocus is assumed even though the position came back unchanged. Expressed as dead time, not wall-clock, because the latter is dominated by exposure length and would mean different things for 30 s and 1800 s subs. `"auto"` reads the threshold from the data. |
| `split_search_min_s` / `max_s` | 40 / 900 | Empirical | Bounds the automatic threshold. The floor sits just below `af_min_dead_time_s` so a fast autofocus following a tight cadence still leaves room between the two. |
| gap ratio for auto split | 2.0x | Empirical | The dead-time distribution is bimodal (a dither costs seconds, an autofocus costs minutes). The threshold is placed in the widest proportional gap; below a 2x ratio the populations are not separated and no dead-time split is inferred at all. |
| `min_below` / `min_above` *(function args, `resolve_split_threshold`)* | 3 / 1 | Empirical | A real population of ordinary cadence must sit below the gap, so one stray value cannot define where "normal" ends. Only one value is required above, because a short run legitimately contains a single autofocus gap. |
| `af_min_dead_time_s` | 60 s | Empirical | Dead time beyond the typical in-block cadence taken as evidence an autofocus ran. Observed autofocus runs took 222-947 s across five archives; dithers and settles took 10-96 s. |
| `block_gap_minutes` | 45 | Practical | Fallback wall-clock split for frames whose `EXPTIME` is missing, so dead time cannot be computed. |
| `position_tolerance` | 0 steps | Physical | Focuser positions are integers; any change is a real move. Raise it only for a focuser that reports jitter. |

## Quality control (`qc.py`)

All of these are **relative to the same filter on the same night** wherever
possible, because absolute limits on star counts or sizes are meaningless
across different filters, targets and skies.

| Constant | Value | Kind | Why |
|---|---|---|---|
| `star_count_frac` | 0.55 | Empirical | Fail a frame whose star count falls below 55% of the nightly median for its filter -- cloud, dew, or transparency loss. |
| `star_count_warn_frac` | 0.75 | Empirical | Warn below 75%. |
| `min_stars` | 25 | Empirical | Absolute floor regardless of the nightly median. |
| `ecc_fail` / `ecc_warn` | 0.60 / 0.50 | Empirical | Median stellar eccentricity. 0.6 corresponds to an axis ratio of 0.8, visible trailing. Measured on the bright band only, since faint second moments are noise-dominated. Reported with the elongation-direction concentration, which distinguishes tracking error from seeing. |
| `hfd_fail_ratio` / `hfd_warn_ratio` | 1.35 / 1.18 | Empirical | Fail a frame whose star size exceeds 1.35x the same-filter, same-night median. Deliberately relative: a uniformly softer filter must not be failed wholesale, only genuine outliers. |
| `background_z` | 5.0 | Statistical | Robust z-score (MAD-scaled) beyond which a frame's sky level is anomalous -- moon, cloud, or a gradient. |
| `backend_disagree_frac` | 0.25 | Empirical | The two backends are compared *after* removing each one's own median, since absolute HFD is aperture-definitional. A 25% residual disagreement on the scale-normalised value flags the frame. In practice this fires almost only on frames that already fail another check. |
| `saturation_rise_px` | 0.35 px | Empirical | If the brightest flux quartile's median HFD exceeds the next quartile's by this much, saturated cores are inflating sizes. Warning only. |
| `min_frames_per_block` | 1 | Practical | A block with no surviving frames is dropped. |

## Model (`model.py`)

| Constant | Value | Kind | Why |
|---|---|---|---|
| `method="auto"` -> robust | n >= 8 blocks | Statistical | Below ~8 blocks a Huber M-estimator has too few degrees of freedom to be worth its variance penalty; above it, robustness to a failed autofocus run is worth more. |
| Huber tuning constant `c` | 1.345 | Statistical | The textbook value, giving ~95% efficiency relative to least squares under Gaussian errors. |
| robust scale floor | half the smallest non-zero gap in the response | Physical | Focuser positions are integers and autofocus often returns the same one repeatedly, so at least half the residuals can coincide and the median absolute deviation collapses to zero. Below the quantisation of the measurement there is nothing to be robust about. Without this, Huber weighted whole blocks out and reported zero scatter with zero-width intervals. |
| bootstrap used | n >= 25 blocks | Statistical | Below this a percentile bootstrap is unstable (it can resample away a filter entirely), so Student-t intervals are reported. Below the threshold the bootstrap is not computed at all, since it would only be discarded. |
| `bootstrap` draws | 2000 | Statistical | Enough for a stable 95th percentile; the cost is negligible when it is actually used. |
| bootstrap minimum draws to trust | 100 | Statistical | Fewer surviving resamples than this and the percentile interval is not reported. |
| clustered resampling | >= 4 sessions | Statistical | Resampling nights rather than blocks needs enough nights to resample. Below 4 the bootstrap resamples blocks, which treats blocks within a night as independent -- one reason the t interval is preferred at small n. |
| `confidence` | 0.95 | Convention | Standard interval level. Configurable; the reported label follows it. |
| `min_within_night_range_c` | 0.5 C | Empirical | Minimum median per-night temperature range that will support a separate within-night slope. Rigs that log one temperature per sequence make that regressor identically zero, and a slope fitted on the residual numerical dust came back wrong by five orders of magnitude. |
| `thermal_lag_minutes` | `None` | Empirical | Fitting the lag is off by default. Shrinking the within-night temperature amplitude scales the slope up while leaving residuals nearly flat, so a grid search drifts toward larger \|k\|, and the reported standard error ignores the fact that a search happened. On simulated data with one true coefficient and *no* lag, fitting it inflated \|k\| by up to 22% and multiplied its scatter by 45x when the within-night range was small. |
| aliasing tolerance | 1e-8 (rank), 1e-10 of the largest column norm | Numerical | A column that is numerical dust next to the others is not a regressor, however linearly independent it looks once every column has been normalised to unit length. |

### Zero-point stability test (`zero_point_drift_test`, function arguments)

| Constant | Value | Kind | Why |
|---|---|---|---|
| `min_nights_per_anchor` | 2 | Statistical | A filter seen on only one night has its offset confounded with that night's level, so it cannot anchor the test. Two nights is the minimum that pins it. |
| `min_nights_for_between_temp` | 4 | Statistical | Below four nights, a between-night temperature term has nearly as many degrees of freedom as there are night means to explain, and it absorbs genuine zero-point movement. Below the threshold the night-mean temperature is instead regressed out at the night level, which costs one parameter rather than one per block. |
| `jump_threshold` | 0.5 x autofocus repeatability | Empirical | Zero-point movement smaller than half the run-to-run scatter of autofocus itself is not worth modelling. |
| `jump_min_steps` | 1.0 step | Physical | A focuser moves in whole steps, so movement below one step cannot be acted on however statistically significant it is. Also prevents very clean data from tripping a purely relative threshold. |
| trend significance | p < 0.05 | Statistical | Conventional level for declaring a drift with date real. Additionally required to exceed the within-night scatter in magnitude. |

## Findings and recommendations (`recommend.py`)

These control *wording and severity*, never the numbers.

| Constant | Value | Kind | Why |
|---|---|---|---|
| VIF alarm | > 5.0 | Convention | The usual rule of thumb for "these regressors carry nearly the same information". |
| condition number alarm | > 30 | Convention | Standard threshold for an ill-conditioned design. |
| blocks per filter | 3 minimum, 6 good | Empirical | Below 3 an offset cannot be separated from autofocus scatter; 6 roughly halves the interval. |
| temperature range | 3 C | Empirical | Below this the coefficient is too poorly constrained to enable compensation on. |
| frames rejected | 25% | Empirical | Above a quarter of frames failing, the session is called critical rather than merely flagged. |
| star-size excess significance | p < 0.01 and excess > 3% | Statistical + empirical | A filter is only called "softer than conditions explain" when it is significant *and* large enough to matter, across at least two metric/backend combinations. |
| hot-pixel contamination | 30% of detections | Empirical | Above this fraction rejected as too small, the frame gets an explicit finding. |

## V-curve (`vcurve.py`)

| Constant | Value | Kind | Why |
|---|---|---|---|
| `min_hfd_ratio` | 1.3x | Physical | If the worst star size in a sweep is not at least 1.3x the best, the curve's bottom is too flat to locate best focus, and a fit would report a precise-looking position that is unconstrained. |
| minimum sweep points | 4 positions, 5 for identifiability | Statistical | Three parameters are fitted (best position, floor, slope). |
| sweep detection | >= 4 distinct positions, <= 120 s dead time, <= 90 min span | Empirical | A sweep means deliberately stepping the focuser while imaging one filter, so the positions must be contiguous in time with no autofocus between. Without the time constraint, a normal night's separate autofocus runs are mistaken for a sweep, and fitting a V-curve to those recovers the seeing trend as a confident, fictitious best focus. |
| wavelength for depth of focus | 0.55 um | Physical | Visual band centre, the conventional choice for the quarter-wave criterion. |
| seeing blur fraction | 1/3 | Empirical | The seeing-limited depth of focus is where the geometric defocus blur reaches a third of the seeing disk -- the point at which defocus starts to matter against seeing. |

---

## If you change one thing

`min_hfd_frac_of_psf` (0.55) and `hfd_max` (40 px) are the two most likely to
need adjusting on an unfamiliar rig: the first if your sampling is unusual, the
second if you analyse deliberately defocused sweeps. Both are in the `profiles`
section of the config.
