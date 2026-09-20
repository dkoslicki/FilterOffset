"""Configuration round-tripping and validation."""

import pytest

from filteroffset.config import Config


def test_round_trip_through_yaml(tmp_path):
    cfg = Config(inputs=["lights"], outdir="out", step_microns=2.63)
    cfg.profiles.crop = 1024
    cfg.qc.ecc_fail = 0.7
    cfg.model.reference_filter = "L"
    cfg.blocks.block_split_dead_time_s = 300.0
    path = cfg.dump(str(tmp_path / "c.yaml"))
    back = Config.load(path)
    assert back.inputs == ["lights"]
    assert back.step_microns == pytest.approx(2.63)
    assert back.profiles.crop == 1024
    assert back.qc.ecc_fail == pytest.approx(0.7)
    assert back.model.reference_filter == "L"
    assert back.blocks.block_split_dead_time_s == pytest.approx(300.0)


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError, match="unknown configuration key"):
        Config.from_dict({"not_a_key": 1})


def test_unknown_nested_key_is_rejected():
    """A typo in a nested section must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="unknown keys for QcParams"):
        Config.from_dict({"qc": {"ecc_fale": 0.7}})


def test_tuple_fields_survive_yaml_lists(tmp_path):
    cfg = Config(inputs=["x"])
    path = cfg.dump(str(tmp_path / "c.yaml"))
    back = Config.load(path)
    assert isinstance(back.blocks.config_columns, tuple)


def test_defaults_are_sensible():
    cfg = Config()
    assert cfg.backends == ["astap", "internal"]
    assert cfg.primary_backend == "internal"
    assert cfg.metric == "hfd_median_bright"
    assert cfg.model.night_effects == "auto"
