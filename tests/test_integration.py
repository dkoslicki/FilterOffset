"""End-to-end: synthesise a night of FITS frames and run the whole pipeline.

The synthetic night is built with *known* filter offsets and a known
temperature coefficient, and the stars are rendered at a width that follows the
real defocus geometry, so this exercises header reading, block segmentation,
both measurement backends, quality control, the models, the plots and the
report in one pass - and checks the answer.
"""

import os

import numpy as np
import pytest

from filteroffset.config import Config

pytest.importorskip("astropy")

TRUE_OFFSETS = {"B": 0.0, "R": -6.0, "G": -5.5}
TRUE_K = -6.5
BASE = 2000.0


def _render(hfd_px: float, size: int = 400, n_stars: int = 140, seed: int = 0,
            background: float = 300.0) -> np.ndarray:
    """A small star field whose stars have the requested half-flux diameter."""
    rng = np.random.default_rng(seed)
    sigma = max(hfd_px, 0.6) / 2.3548  # HFD == FWHM for a Gaussian
    img = np.full((size, size), background, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_stars):
        cx, cy = rng.uniform(12, size - 12, 2)
        amp = rng.lognormal(7.4, 0.8)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        near = r2 < (7 * sigma) ** 2
        img[near] += (amp * np.exp(-r2[near] / (2 * sigma**2))).astype(np.float32)
    img += rng.normal(0, 8.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 65535)


def build_night(tmpdir: str, n_cycles: int = 5, n_frames: int = 3,
                seeing_hfd: float = 3.0) -> str:
    from astropy.io import fits

    os.makedirs(tmpdir, exist_ok=True)
    filters = ["B", "R", "G"]
    order = filters * n_cycles
    temps = np.linspace(17.0, 13.5, len(order))
    tmean = float(np.mean(temps))
    t = np.datetime64("2026-09-16T01:30:00")
    seq = 0
    for i, (filt, temp) in enumerate(zip(order, temps)):
        pos = round(BASE + TRUE_OFFSETS[filt] + TRUE_K * (temp - tmean))
        for j in range(n_frames):
            # Autofocus pause before the first frame of each block.
            t = t + np.timedelta64(600 + (16 if j else 440), "s")
            data = _render(seeing_hfd, seed=seq, background=300.0 + 40 * i)
            hdu = fits.PrimaryHDU(data.astype(np.uint16))
            h = hdu.header
            h["DATE-OBS"] = str(t)
            h["EXPTIME"] = 600
            h["FILTER"] = filt
            h["FOCUSPOS"] = int(pos)
            h["AMB-TEMP"] = float(round(temp, 2))
            h["IMAGETYP"] = "Light Frame"
            h["INSTRUME"] = "SynthCam"
            h["TELESCOP"] = "SynthScope"
            h["FOCALLEN"] = 530
            h["APTDIA"] = 160
            h["XPIXSZ"] = 3.76
            h["YPIXSZ"] = 3.76
            h["XBINNING"] = 1
            h["YBINNING"] = 1
            h["OBJCTALT"] = 70.0
            h["AIRMASS"] = 1.06
            h["OBJECT"] = "Synthetic"
            hdu.writeto(os.path.join(tmpdir, f"L_{9000+seq}_{filt}.fit"), overwrite=True)
            seq += 1
    return tmpdir


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    return build_night(str(tmp_path_factory.mktemp("night")))


def test_pipeline_recovers_known_offsets(night, tmp_path):
    from filteroffset.pipeline import run

    cfg = Config(
        inputs=[night], outdir=str(tmp_path / "out"),
        backends=["internal"], progress=False, step_microns=2.0,
    )
    cfg.profiles.crop = None
    cfg.profiles.min_stars = 10
    cfg.model.bootstrap = 0
    cfg.model.reference_filter = "B"
    res = run(cfg)

    assert len(res.blocks_qc) == 15
    assert res.frames_qc["af_likely"].fillna(True).all()
    for f, want in TRUE_OFFSETS.items():
        assert res.fit.offsets[f] == pytest.approx(want, abs=1.0), f
    assert res.fit.temp_coeff == pytest.approx(TRUE_K, abs=0.5)
    # No filter was actually defocused, so no star-size excess should appear.
    for label, ex in res.hfd_excess.items():
        for filt, val in ex.excess_frac.items():
            assert abs(val) < 0.06, f"{label} {filt} spurious excess {val:.3f}"


def test_pipeline_writes_all_products(night, tmp_path):
    from filteroffset.pipeline import run

    out = tmp_path / "out2"
    cfg = Config(
        inputs=[night], outdir=str(out), backends=["internal"], progress=False,
    )
    cfg.profiles.crop = None
    cfg.profiles.min_stars = 10
    cfg.model.bootstrap = 0
    res = run(cfg)

    for name in ("report.md", "report.html", "offsets.json", "offsets.csv",
                 "blocks.csv", "frames.csv", "config_used.yaml"):
        assert (out / name).exists(), name
    assert len(res.plots) >= 7
    for p in res.plots:
        assert os.path.getsize(p) > 5000
        assert os.path.exists(p.replace(".png", ".csv"))
    html = (out / "report.html").read_text()
    assert "<table>" in html and "Filter offset" in html
    assert res.warnings == [] or all("plot" not in w for w in res.warnings), res.warnings


def test_detects_a_defocused_filter(tmp_path):
    """A filter rendered genuinely out of focus must raise the star-size flag."""
    from astropy.io import fits

    from filteroffset.pipeline import run

    night = build_night(str(tmp_path / "bad"), n_cycles=4)
    # Re-render every R frame 25% broader, as a constant autofocus error would.
    for name in sorted(os.listdir(night)):
        if "_R" not in name:
            continue
        path = os.path.join(night, name)
        with fits.open(path) as hdul:
            hdr = hdul[0].header.copy()
        seed = int(name.split("_")[1])
        data = _render(3.0 * 1.25, seed=seed, background=300.0)
        fits.PrimaryHDU(data.astype(np.uint16), header=hdr).writeto(path, overwrite=True)

    cfg = Config(
        inputs=[night], outdir=str(tmp_path / "out3"),
        backends=["internal"], progress=False,
    )
    cfg.profiles.crop = None
    cfg.profiles.min_stars = 10
    cfg.model.bootstrap = 0
    cfg.model.reference_filter = "B"
    res = run(cfg)
    assert res.hfd_excess, "star-size model should have run"
    excess = [ex.excess_frac.get("R", 0.0) for ex in res.hfd_excess.values()]
    assert max(excess) > 0.10, f"defocused R not detected: {excess}"
    codes = {f.code for f in res.recommendations.findings}
    assert "star_size_excess_R" in codes


def test_cli_runs(night, tmp_path):
    from click.testing import CliRunner

    from filteroffset.cli import main

    r = CliRunner().invoke(
        main,
        ["run", night, "-o", str(tmp_path / "cli"), "--backend", "internal",
         "--crop", "0", "--no-progress", "--reference", "B"],
    )
    assert r.exit_code in (0, 2), r.output
    assert "FILTER OFFSETS" in r.output


def test_cli_inspect(night):
    from click.testing import CliRunner

    from filteroffset.cli import main

    r = CliRunner().invoke(main, ["inspect", night, "--keywords"])
    assert r.exit_code == 0, r.output
    assert "focus_pos" in r.output and "FOCUSPOS" in r.output


def _have_astap() -> bool:
    from filteroffset.profiles.astap import astap_path

    return astap_path() is not None


@pytest.mark.skipif(not _have_astap(), reason="astap_cli not installed")
def test_astap_backend_agrees_with_internal_on_the_pattern(night, tmp_path):
    """Both backends must rank frames the same way, even at different scales."""

    from filteroffset.blocks import segment_blocks
    from filteroffset.ingest import scan_directory
    from filteroffset.measure import measure_frames
    from filteroffset.profiles import ProfileParams

    frames = segment_blocks(scan_directory([night]).frames)
    params = ProfileParams(crop=None, min_stars=10)
    res = measure_frames(frames, backends=("astap", "internal"), params=params)
    assert not res.profiles.empty
    ok = res.profiles.groupby("backend")["ok"].apply(lambda s: s.fillna(False).mean())
    assert (ok > 0.8).all(), f"backend failure rate too high:\n{ok}"

    wide = res.profiles.pivot_table(
        index="path", columns="backend", values="hfd_median"
    ).dropna()
    assert len(wide) >= 10
    # Absolute scales differ by design; the frame-to-frame pattern must not.
    rho = wide["astap"].corr(wide["internal"], method="spearman")
    assert rho > 0.5, f"backends disagree on the pattern (rho={rho:.2f})"


@pytest.mark.skipif(not _have_astap(), reason="astap_cli not installed")
def test_astap_leaves_the_input_directory_untouched(night, tmp_path):
    """ASTAP writes its CSV beside the file it reads, so we run it on symlinks."""
    from filteroffset.profiles import ProfileParams
    from filteroffset.profiles.astap import measure_frame

    before = set(os.listdir(night))
    target = os.path.join(night, sorted(before)[0])
    prof = measure_frame(target, ProfileParams(min_stars=5), workdir=str(tmp_path))
    assert prof.ok, prof.error
    assert set(os.listdir(night)) == before, "ASTAP polluted the data directory"


def build_campaign(tmpdir: str, layout, seeing_hfd: float = 3.0,
                   offsets=None, k: float = -4.0,
                   night_jump_sd: float = 0.0) -> str:
    """Synthesise several nights, days apart, from {night_index: [filters]}."""
    import numpy as np
    from astropy.io import fits

    offsets = offsets or {"B": 0.0, "R": -5.0, "G": -6.0, "Ha": 18.0, "OIII": 2.0}
    os.makedirs(tmpdir, exist_ok=True)
    rng = np.random.default_rng(3)
    seq = 0
    for night, filters in sorted(layout.items()):
        # Nights are days apart, as they are when waiting on weather.
        start = np.datetime64("2026-07-22T02:00:00") + np.timedelta64(
            int(3 * night), "D"
        )
        # Default 0: the documented assumption is an undisturbed imaging train.
        night_shift = rng.normal(0.0, night_jump_sd) if night_jump_sd else 0.0
        t = start
        for cycle in range(3):
            for filt in filters:
                temp = 18.0 - 1.1 * cycle + rng.normal(0, 0.05)
                pos = round(
                    2000.0 + offsets[filt] + k * (temp - 15.0) + night_shift
                )
                for j in range(2):
                    t = t + np.timedelta64(300 + (16 if j else 440), "s")
                    data = _render(seeing_hfd, seed=seq, background=300.0)
                    hdu = fits.PrimaryHDU(data.astype(np.uint16))
                    h = hdu.header
                    h["DATE-OBS"] = str(t)
                    h["EXPTIME"] = 300
                    h["FILTER"] = filt
                    h["FOCUSPOS"] = int(pos)
                    h["AMB-TEMP"] = float(round(temp, 2))
                    h["IMAGETYP"] = "Light Frame"
                    h["INSTRUME"] = "SynthCam"
                    h["TELESCOP"] = "SynthScope"
                    h["FOCALLEN"] = 530
                    h["APTDIA"] = 160
                    h["XPIXSZ"] = 3.76
                    h["YPIXSZ"] = 3.76
                    h["XBINNING"] = 1
                    h["YBINNING"] = 1
                    h["OBJCTALT"] = 70.0
                    h["AIRMASS"] = 1.06
                    h["OBJECT"] = "Synthetic"
                    hdu.writeto(
                        os.path.join(tmpdir, f"L_{7000 + seq}_{filt}.fit"),
                        overwrite=True,
                    )
                    seq += 1
    return tmpdir


#: Shaped like the real archive: narrowband mostly alone, one mixed night.
CAMPAIGN_LAYOUT = {
    0: ["OIII"], 1: ["OIII"], 2: ["OIII"], 3: ["Ha"], 4: ["B", "G", "R", "Ha"],
}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Five nights, undisturbed imaging train."""
    return build_campaign(
        str(tmp_path_factory.mktemp("campaign")), CAMPAIGN_LAYOUT
    )


@pytest.fixture(scope="module")
def jumpy_campaign(tmp_path_factory):
    """The same nights, but with the focuser zero point moving between them."""
    return build_campaign(
        str(tmp_path_factory.mktemp("jumpy")), CAMPAIGN_LAYOUT, night_jump_sd=25.0
    )


def _run(inputs, outdir):
    from filteroffset.config import Config
    from filteroffset.pipeline import run

    cfg = Config(inputs=[inputs], outdir=str(outdir), backends=["internal"],
                 progress=False, step_microns=2.0)
    cfg.profiles.crop = None
    cfg.profiles.min_stars = 10
    cfg.model.bootstrap = 0
    cfg.model.reference_filter = "B"
    return run(cfg)


def test_multi_night_campaign_separates_nights(campaign, tmp_path):
    """Nights are recovered, and a stable focuser keeps every offset measurable."""
    res = _run(campaign, tmp_path / "camp")
    assert res.blocks_qc["session"].nunique() == 5
    # The synthetic train is undisturbed, so between-night information is valid
    # and even the filter that never shares a night gets an offset.
    assert res.fit.not_identified == []
    for f in ("G", "R", "Ha", "OIII"):
        assert np.isfinite(res.fit.offsets[f])
    codes = {f.code for f in res.recommendations.findings}
    assert "zero_point_stable" in codes or "zero_point_untested" in codes


def test_jumpy_campaign_is_detected_and_protected(jumpy_campaign, tmp_path):
    """A disturbed train must be caught, not silently trusted."""
    res = _run(jumpy_campaign, tmp_path / "jumpy")
    assert res.fit.drift_test.verdict == "jumps"
    assert res.fit.diagnostics["night_effects"] is True
    assert res.fit.not_identified == ["OIII"]
    codes = {f.code for f in res.recommendations.findings}
    assert "zero_point_jumps" in codes


def test_forcing_night_effects_flags_the_orphan(campaign, tmp_path):
    """The conservative path is still available and still reports honestly."""
    from filteroffset.config import Config
    from filteroffset.pipeline import run

    cfg = Config(inputs=[campaign], outdir=str(tmp_path / "camp_on"),
                 backends=["internal"], progress=False)
    cfg.profiles.crop = None
    cfg.profiles.min_stars = 10
    cfg.model.bootstrap = 0
    cfg.model.reference_filter = "B"
    cfg.model.night_effects = "on"
    res = run(cfg)
    assert res.fit.not_identified == ["OIII"]
    assert not np.isfinite(res.fit.offsets["OIII"])
    codes = {f.code for f in res.recommendations.findings}
    assert "offsets_not_identified" in codes


def test_multi_night_campaign_recovers_the_linked_offsets(campaign, tmp_path):
    res = _run(campaign, tmp_path / "camp2")
    truth = {"G": -6.0, "R": -5.0, "Ha": 18.0, "OIII": 2.0}
    for f, want in truth.items():
        assert res.fit.offsets[f] == pytest.approx(want, abs=2.5), f


def test_multi_night_plots_and_report_are_written(campaign, tmp_path):
    out = tmp_path / "camp3"
    res = _run(campaign, out)
    assert len(res.plots) >= 7
    for p in res.plots:
        assert os.path.getsize(p) > 5000
    # Every figure link must resolve relative to the report itself.
    md = (out / "report.md").read_text()
    import re

    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    assert links, "report should embed figures"
    for link in links:
        assert (out / link).exists(), f"broken figure link: {link}"
    html = (out / "report.html").read_text()
    for src in re.findall(r'src="([^"]+\.png)"', html):
        assert (out / src).exists(), f"broken html image: {src}"


def test_report_defines_its_symbols(campaign, tmp_path):
    out = tmp_path / "camp4"
    _run(campaign, out)
    md = (out / "report.md").read_text()
    assert "## What the symbols mean" in md
    for symbol in ("`k`", "`VIF`", "`F`", "`p`", "`R^2`", "`HFD`", "`SE`",
                   "`CI`", "`Cook's distance`", "identified"):
        assert symbol in md, f"{symbol} is used but never defined"
