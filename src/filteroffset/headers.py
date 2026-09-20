"""Fast FITS header reading and vendor-agnostic keyword normalisation.

Two problems are solved here.

**Speed.**  A single QHY600 frame is ~123 MB.  Opening a hundred thousand of
them with a general purpose FITS reader is dominated by per-file overhead, so
:func:`read_primary_header` reads only the 2880-byte header blocks at the front
of the file and stops at ``END``.  That is typically one or two disk reads per
file regardless of image size.

**Vocabulary.**  Every capture program spells the interesting keywords
differently (APT writes ``FOCUSPOS`` and ``AMB-TEMP``, N.I.N.A. writes
``FOCPOS``/``FOCUSTEM``, SGP writes ``FOCUSER``...).  :func:`normalise_header`
maps whatever is present onto a fixed internal schema, and records which source
keyword each value came from so the report can be audited.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

BLOCK = 2880
CARD = 80

# --- canonical schema -------------------------------------------------------
# Each entry maps an internal column name to the FITS keywords that may carry
# it, in order of preference.  Keys are matched case-insensitively.
KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    # identity / sequencing
    "date_obs": ("DATE-OBS", "DATE_OBS", "DATEOBS", "DATE-BEG"),
    "filter": ("FILTER", "FILTER1", "FILT", "FILTNAME", "INSFLNAM"),
    "image_type": ("IMAGETYP", "IMAGETYPE", "FRAME", "OBSTYPE"),
    "object": ("OBJECT", "OBJNAME", "TARGET"),
    "exptime": ("EXPTIME", "EXPOSURE", "EXPTIMEs", "ITIME"),
    # focuser
    "focus_pos": (
        "FOCUSPOS", "FOCPOS", "FOCUSER", "FOCUSERPOS", "FOCUSERPOSITION",
        "FOCPOSIT", "POSITION", "FOCUSSTEP", "ABSPOS", "FOC-POS",
    ),
    "focus_temp": (
        # Temperature reported *by the focuser* (usually closest to the OTA).
        "FOCUSTEM", "FOCTEMP", "FOCUSTEMP", "FOCTEM", "FOCUSERTEMP", "FOC-TEMP",
    ),
    "ambient_temp": (
        # Ambient / sky-quality / weather-station probe.
        "AMB-TEMP", "AMBTEMP", "AMBIENT", "TEMPERAT", "EXTTEMP", "AOCAMBT",
        "OUTTEMP", "SKYTEMP", "ENVTEMP", "WXTEMP",
    ),
    "focus_step_size": ("FOCSTEP", "FOCUSSTEPSIZE", "STEPSIZE"),
    # camera
    "ccd_temp": ("CCD-TEMP", "CCDTEMP", "SENSORTEMP", "CAMTEMP"),
    "set_temp": ("SET-TEMP", "SETTEMP", "CCDSETT"),
    "gain": ("GAIN", "EGAIN_SET", "ISOSPEED"),
    "offset": ("OFFSET", "BLKLEVEL", "BLACKLEV"),
    "xbinning": ("XBINNING", "BINX", "XBIN"),
    "ybinning": ("YBINNING", "BINY", "YBIN"),
    "xpixsz": ("XPIXSZ", "PIXSIZE1", "PIXSZ", "XPIXELSZ"),
    "ypixsz": ("YPIXSZ", "PIXSIZE2", "YPIXELSZ"),
    "naxis1": ("NAXIS1",),
    "naxis2": ("NAXIS2",),
    "instrument": ("INSTRUME", "CAMERA", "DETECTOR"),
    # telescope / pointing
    "telescope": ("TELESCOP", "TELESCOPE", "OTA"),
    "focal_len": ("FOCALLEN", "FOCLEN", "TELFOC"),
    "aperture": ("APTDIA", "APERTURE", "TELAPER"),
    "focal_ratio": ("FOCRATIO", "APTAREA_F", "FRATIO"),
    "airmass": ("AIRMASS", "SECZ"),
    "altitude": ("OBJCTALT", "CENTALT", "ALTITUDE", "ELEVATION_OBJ", "ALT-OBJ"),
    "azimuth": ("OBJCTAZ", "CENTAZ", "AZIMUTH"),
    "ra": ("RA", "OBJCTRA", "CRVAL1"),
    "dec": ("DEC", "OBJCTDEC", "CRVAL2"),
    "rotator": ("ROTATANG", "ROTATOR", "ROTPOSN", "DEROTANG"),
    # environment / guiding (used for QC only)
    "humidity": ("HUMIDITY", "AOCHUM", "OUTHUM", "WXHUMID"),
    "dewpoint": ("DEWPOINT", "AOCDEW"),
    "pressure": ("PRESSURE", "AOCBAROM", "WXPRES"),
    "software": ("SWCREATE", "CREATOR", "PROGRAM"),
    "guide_rms": ("GUIDERMS", "RMSERROR", "GUIDE_RMS"),
    "hfr_header": ("HFR", "HFD", "FWHM", "MEDHFR"),
}

# Columns that must be numeric in the tidy table.
NUMERIC_COLUMNS = (
    "exptime", "focus_pos", "focus_temp", "ambient_temp", "focus_step_size",
    "ccd_temp", "set_temp", "gain", "offset", "xbinning", "ybinning",
    "xpixsz", "ypixsz", "naxis1", "naxis2", "focal_len", "aperture",
    "focal_ratio", "airmass", "altitude", "azimuth", "rotator", "humidity",
    "dewpoint", "pressure", "guide_rms", "hfr_header",
)

_LOOKUP: dict[str, str] = {}
for _canon, _aliases in KEYWORD_ALIASES.items():
    for _a in _aliases:
        _LOOKUP.setdefault(_a.upper(), _canon)

FITS_EXTENSIONS = (".fit", ".fits", ".fts", ".fit.fz", ".fits.fz", ".fts.fz")


class HeaderReadError(RuntimeError):
    """Raised when a file cannot be parsed as FITS."""


@dataclass
class HeaderRecord:
    """A normalised header plus provenance for every value that was used."""

    path: str
    values: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"path": self.path}
        row.update(self.values)
        return row


def _parse_card(card: str) -> tuple[str | None, Any]:
    """Parse one 80-character FITS card into (keyword, value)."""
    key = card[:8].strip()
    if not key or key in ("COMMENT", "HISTORY", "END"):
        return None, None
    if card[8:10] != "= ":
        return None, None
    body = card[10:]

    # Strip an inline comment, honouring single-quoted strings.
    if body.lstrip().startswith("'"):
        start = body.index("'")
        i = start + 1
        chunks: list[str] = []
        while i < len(body):
            if body[i] == "'":
                if i + 1 < len(body) and body[i + 1] == "'":
                    chunks.append("'")
                    i += 2
                    continue
                break
            chunks.append(body[i])
            i += 1
        return key, "".join(chunks).strip()

    slash = body.find("/")
    token = (body if slash < 0 else body[:slash]).strip()
    if not token:
        return key, None
    if token in ("T", "F"):
        return key, token == "T"
    try:
        if re.fullmatch(r"[+-]?\d+", token):
            return key, int(token)
        return key, float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return key, token


def read_primary_header(path: str | os.PathLike[str], max_blocks: int = 64) -> dict[str, Any]:
    """Read only the primary HDU header of *path*.

    Stops at the ``END`` card, so cost is independent of image size.  Falls back
    to :mod:`astropy` for anything unusual (compressed or non-conforming files).
    """
    path = os.fspath(path)
    header: dict[str, Any] = {}
    try:
        with open(path, "rb") as fh:
            first = fh.read(BLOCK)
            if not first.startswith(b"SIMPLE") and not first.startswith(b"XTENSION"):
                return _read_header_astropy(path)
            block = first
            for _ in range(max_blocks):
                if not block:
                    break
                text = block.decode("ascii", errors="replace")
                done = False
                for i in range(0, len(text), CARD):
                    card = text[i:i + CARD]
                    if card.startswith("END") and card[3:].strip() == "":
                        done = True
                        break
                    key, value = _parse_card(card)
                    if key is not None and key not in header:
                        header[key] = value
                if done:
                    break
                block = fh.read(BLOCK)
    except OSError as exc:  # unreadable file
        raise HeaderReadError(f"{path}: {exc}") from exc

    if not header:
        return _read_header_astropy(path)
    # A tile-compressed image keeps the real keywords in the first extension.
    if header.get("NAXIS", 0) == 0 or str(path).endswith(".fz"):
        try:
            merged = _read_header_astropy(path)
            merged.update({k: v for k, v in header.items() if k not in merged})
            return merged
        except HeaderReadError:
            pass
    return header


def _read_header_astropy(path: str) -> dict[str, Any]:
    try:
        import warnings

        from astropy.io import fits

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=True) as hdul:
                for hdu in hdul:
                    if getattr(hdu, "data", None) is not None or hdu.header.get("NAXIS", 0) > 0:
                        return {k: hdu.header[k] for k in hdu.header if k}
                return {k: hdul[0].header[k] for k in hdul[0].header if k}
    except Exception as exc:  # pragma: no cover - depends on corrupt input
        raise HeaderReadError(f"{path}: {exc}") from exc


def _coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().rstrip("Cc").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        return float(m.group()) if m else None


def parse_date_obs(value: Any) -> datetime | None:
    """Parse a FITS ``DATE-OBS`` string into a timezone-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "")
        if not text:
            return None
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def normalise_filter_name(value: Any) -> str | None:
    """Canonicalise a filter name so ``Ha``/``H-alpha``/``Halpha`` agree."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    squashed = re.sub(r"[\s_\-.]+", "", text).lower()
    canonical = {
        "ha": "Ha", "halpha": "Ha", "hal": "Ha", "h": "Ha", "ha3nm": "Ha",
        "hb": "Hb", "hbeta": "Hb",
        "o3": "OIII", "oiii": "OIII", "oxygen3": "OIII", "o": "OIII",
        "s2": "SII", "sii": "SII", "sulphur2": "SII", "s": "SII",
        "n2": "NII", "nii": "NII",
        "l": "L", "lum": "L", "luminance": "L", "clear": "L", "cls": "L",
        "r": "R", "red": "R", "g": "G", "green": "G", "b": "B", "blue": "B",
        "lp": "LP", "uvir": "UVIR", "uvircut": "UVIR", "lenhance": "LeNhance",
        "ldn": "LDN", "lextreme": "LeXtreme", "duoband": "Duo", "dualband": "Duo",
    }
    return canonical.get(squashed, text)


def normalise_header(
    path: str | os.PathLike[str],
    header: Mapping[str, Any] | None = None,
    extra_aliases: Mapping[str, Sequence[str]] | None = None,
) -> HeaderRecord:
    """Map a raw FITS header onto the internal schema.

    *extra_aliases* lets a site-specific config add keywords without editing
    this module; entries are tried *before* the built-in ones.
    """
    path = os.fspath(path)
    if header is None:
        header = read_primary_header(path)

    upper = {str(k).upper(): v for k, v in header.items()}
    rec = HeaderRecord(path=path, raw=dict(header))

    lookup_order: dict[str, list[str]] = {
        canon: list(aliases) for canon, aliases in KEYWORD_ALIASES.items()
    }
    for canon, aliases in (extra_aliases or {}).items():
        lookup_order.setdefault(canon, [])
        lookup_order[canon] = [str(a) for a in aliases] + lookup_order[canon]

    for canon, aliases in lookup_order.items():
        for alias in aliases:
            key = alias.upper()
            if key in upper and upper[key] is not None and upper[key] != "":
                rec.values[canon] = upper[key]
                rec.provenance[canon] = key
                break

    for col in NUMERIC_COLUMNS:
        if col in rec.values:
            num = _coerce_number(rec.values[col])
            if num is None:
                rec.values.pop(col, None)
                rec.provenance.pop(col, None)
            else:
                rec.values[col] = num

    rec.values["filter"] = normalise_filter_name(rec.values.get("filter"))
    dt = parse_date_obs(rec.values.get("date_obs"))
    rec.values["date_obs"] = dt

    # Derived quantities that are cheap and widely useful.
    fl, ap = rec.values.get("focal_len"), rec.values.get("aperture")
    if rec.values.get("focal_ratio") is None and fl and ap:
        rec.values["focal_ratio"] = fl / ap
    px = rec.values.get("xpixsz")
    if fl and px:
        # xpixsz is already post-binning in the SBIG convention APT follows.
        rec.values["pixel_scale"] = 206.265 * px / fl
    if rec.values.get("airmass") is None and rec.values.get("altitude") is not None:
        alt = float(rec.values["altitude"])
        if alt > 3.0:
            import math

            rec.values["airmass"] = 1.0 / math.cos(math.radians(90.0 - alt))
    rec.values["image_type"] = (
        str(rec.values["image_type"]).strip() if rec.values.get("image_type") else None
    )
    return rec


def is_fits_path(path: str | os.PathLike[str]) -> bool:
    name = os.fspath(path).lower()
    return name.endswith(FITS_EXTENSIONS)


def iter_fits_files(
    roots: Iterable[str | os.PathLike[str]],
    recursive: bool = True,
    follow_symlinks: bool = False,
) -> Iterable[str]:
    """Yield FITS paths under *roots*, sorted for reproducible runs."""
    for root in roots:
        root = os.fspath(root)
        if os.path.isfile(root):
            if is_fits_path(root):
                yield root
            continue
        if not recursive:
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if os.path.isfile(full) and is_fits_path(full):
                    yield full
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            dirnames.sort()
            for name in sorted(filenames):
                if is_fits_path(name):
                    yield os.path.join(dirpath, name)
