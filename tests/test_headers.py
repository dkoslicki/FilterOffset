"""Header parsing and vendor-agnostic keyword mapping."""

import numpy as np
import pytest

from filteroffset.headers import (
    _parse_card,
    normalise_filter_name,
    normalise_header,
    parse_date_obs,
)


@pytest.mark.parametrize(
    "card,key,value",
    [
        ("FOCUSPOS=                 1996 /  Focuser position in steps       ", "FOCUSPOS", 1996),
        ("AMB-TEMP=                16.61 / The Ambient Temperature          ", "AMB-TEMP", 16.61),
        ("FILTER  = 'B'                  / Filter used                      ", "FILTER", "B"),
        ("SIMPLE  =                    T / conforms                         ", "SIMPLE", True),
        ("OBJECT  = 'NGC 7000'           / name                             ", "OBJECT", "NGC 7000"),
        ("COMMENT this is a comment                                         ", None, None),
    ],
)
def test_parse_card(card, key, value):
    k, v = _parse_card(card.ljust(80))
    assert k == key
    assert v == value


def test_parse_card_handles_slash_inside_string():
    k, v = _parse_card("OBJECT  = 'M42 / M43'          / two objects".ljust(80))
    assert (k, v) == ("OBJECT", "M42 / M43")


def test_parse_date_obs_variants():
    for text in ("2026-09-16T01:30:47", "2026-09-16T01:30:47.500", "2026-09-16 01:30:47"):
        dt = parse_date_obs(text)
        assert dt is not None and dt.year == 2026 and dt.hour == 1
    assert parse_date_obs("not a date") is None
    assert parse_date_obs(None) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ha", "Ha"), ("H-alpha", "Ha"), ("Halpha", "Ha"), ("h_alpha", "Ha"),
        ("OIII", "OIII"), ("O3", "OIII"), ("o-iii", "OIII"),
        ("SII", "SII"), ("S2", "SII"),
        ("Red", "R"), ("blue", "B"), ("Luminance", "L"),
        ("MyCustomFilter", "MyCustomFilter"),
    ],
)
def test_filter_normalisation(raw, expected):
    assert normalise_filter_name(raw) == expected


def test_alias_mapping_across_capture_software():
    """APT, N.I.N.A. and SGP spellings must all reach the same columns."""
    apt = {"FOCUSPOS": 1996, "AMB-TEMP": 16.6, "FILTER": "B", "IMAGETYP": "Light Frame"}
    nina = {"FOCPOS": 1996, "FOCUSTEM": 16.6, "FILTER": "B", "IMAGETYP": "LIGHT"}
    sgp = {"FOCUSER": 1996, "TEMPERAT": 16.6, "FILTER": "B", "IMAGETYP": "Light"}
    for header in (apt, nina, sgp):
        rec = normalise_header("x.fit", header)
        assert rec.values["focus_pos"] == 1996.0
        assert rec.values["filter"] == "B"
        temp = rec.values.get("focus_temp", rec.values.get("ambient_temp"))
        assert temp == pytest.approx(16.6)


def test_extra_aliases_take_priority():
    header = {"MYFOCUS": 1234, "FOCUSPOS": 9999, "FILTER": "R"}
    rec = normalise_header("x.fit", header, extra_aliases={"focus_pos": ["MYFOCUS"]})
    assert rec.values["focus_pos"] == 1234.0
    assert rec.provenance["focus_pos"] == "MYFOCUS"


def test_derived_quantities():
    rec = normalise_header(
        "x.fit",
        {"FOCALLEN": 530, "APTDIA": 160, "XPIXSZ": 3.76, "OBJCTALT": 30.0, "FILTER": "G"},
    )
    assert rec.values["focal_ratio"] == pytest.approx(530 / 160)
    assert rec.values["pixel_scale"] == pytest.approx(206.265 * 3.76 / 530)
    # airmass derived from altitude via sec(z)
    assert rec.values["airmass"] == pytest.approx(1 / np.cos(np.radians(60.0)), rel=1e-6)


def test_units_suffix_is_stripped():
    rec = normalise_header("x.fit", {"AMB-TEMP": "16.61 C", "FILTER": "B"})
    assert rec.values["ambient_temp"] == pytest.approx(16.61)
