"""Export offsets in the form capture software wants.

Conventions differ in a way that matters and is easy to get backwards:

* **Offsets are relative to a reference filter.**  Whichever filter you nominate
  gets 0 and the rest are expressed against it.  Changing the reference shifts
  every number by a constant; it does not change the focus positions the
  software will command.
* **Sign.**  This package reports an offset as *"add this to the reference
  filter's in-focus position to get this filter's in-focus position"*, which is
  what APT, N.I.N.A. and SGP all mean by a filter offset.  A negative value
  means the filter comes to focus at a lower step count.
* **Temperature compensation sign.**  APT and most focuser drivers ask for
  steps per degree C to *apply*, matching the fitted slope dposition/dT
  directly.  A negative coefficient means the focuser must move inward as
  temperature falls.  Drivers do occasionally invert this; the report says how
  to verify it in one move rather than trusting the convention.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def offsets_for_capture_software(fit, integer: bool = True) -> pd.DataFrame:
    """Offsets as a capture-software-ready table.

    Filters whose offset the design cannot identify are carried through with a
    null rather than a rounded guess; see :mod:`filteroffset.model` for why a
    number would be meaningless.
    """
    tbl = fit.offset_table().copy()
    vals = pd.to_numeric(tbl["offset_steps"], errors="coerce")
    if integer:
        tbl["offset_to_enter"] = vals.round().astype("Int64")
    else:
        tbl["offset_to_enter"] = vals.round(2)
    tbl["reference"] = fit.reference
    cols = [
        "filter", "identified", "offset_to_enter", "offset_steps", "se_steps",
        "ci_lo", "ci_hi", "n_blocks", "is_reference", "reference",
    ]
    if "offset_pooled_steps" in tbl.columns:
        cols.append("offset_pooled_steps")
    return tbl[[c for c in cols if c in tbl.columns]]


def rebase(fit, new_reference: str) -> dict[str, float]:
    """Re-express the offsets against a different reference filter."""
    if new_reference not in fit.offsets:
        raise ValueError(
            f"{new_reference!r} is not one of the fitted filters: "
            f"{sorted(fit.offsets)}"
        )
    shift = fit.offsets[new_reference]
    if not np.isfinite(shift):
        raise ValueError(
            f"{new_reference!r} has no identified offset, so it cannot serve as "
            "a reference. Pick a filter that shares a night with the others."
        )
    return {f: v - shift for f, v in fit.offsets.items()}


def to_json(fit, recommendations=None, extra: dict[str, Any] | None = None) -> str:
    """Machine-readable summary, for scripting or version control."""
    payload: dict[str, Any] = {
        "schema": "filteroffset/v1",
        "reference_filter": fit.reference,
        "offsets_steps": {
            f: (round(float(v), 3) if np.isfinite(v) else None)
            for f, v in fit.offsets.items()
        },
        "offsets_to_enter": {
            f: (int(round(float(v))) if np.isfinite(v) else None)
            for f, v in fit.offsets.items()
        },
        "offsets_not_identified": list(getattr(fit, "not_identified", []) or []),
        "filter_groups": [list(c) for c in (getattr(fit, "components", []) or [])],
        "offsets_pooled_steps": {
            f: (round(float(v), 3) if np.isfinite(v) else None)
            for f, v in (getattr(fit, "offsets_pooled", {}) or {}).items()
        },
        "offsets_se_steps": {f: round(float(v), 3) for f, v in fit.offsets_se.items()},
        "offsets_ci95": {
            f: [round(float(a), 3), round(float(b), 3)]
            for f, (a, b) in fit.offsets_ci.items()
        },
        "temperature_coefficient_steps_per_C": _num_or_none(fit.temp_coeff),
        "temperature_coefficient_se": _num_or_none(fit.temp_coeff_se),
        "temperature_coefficient_ci95": [
            _num_or_none(fit.temp_coeff_ci[0]),
            _num_or_none(fit.temp_coeff_ci[1]),
        ],
        "temperature_column": fit.temp_column,
        "thermal_lag_minutes": fit.thermal_lag_minutes,
        "reference_temperature_C": round(float(fit.temp_ref), 3),
        "model": {
            "method": fit.method,
            "n_blocks": int(fit.n_blocks),
            "n_parameters": int(fit.n_params),
            "residual_scale_steps": round(float(fit.resid_scale), 4),
            "r_squared": round(float(fit.r2), 5),
            "ci_method": fit.ci_method,
        },
    }
    if recommendations is not None:
        payload["verdict"] = recommendations.verdict
        payload["trust_offsets"] = bool(recommendations.trust_offsets)
        payload["trust_temperature_coefficient"] = bool(
            recommendations.trust_temp_coeff
        )
        payload["findings"] = [f.as_row() for f in recommendations.findings]
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2, default=_default)


def _num_or_none(value, digits: int = 4) -> float | None:
    """Round for output, or None when undefined - JSON has no NaN."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def _default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def instructions(fit, step_size_um: float | None = None) -> list[str]:
    """Step-by-step application notes, including how to check the sign."""
    ref = fit.reference
    not_id = set(getattr(fit, "not_identified", []) or [])
    pooled = getattr(fit, "offsets_pooled", {}) or {}
    lines = [
        f"Set **{ref}** as the reference filter with offset **0**.",
    ]
    for f in sorted(fit.offsets):
        if f == ref:
            continue
        v = fit.offsets[f]
        if f in not_id or not np.isfinite(v):
            pv = pooled.get(f)
            extra = (
                f" A provisional value of **{int(round(pv)):+d} steps** comes from "
                "ignoring night-to-night zero-point shifts; use it only until you "
                "can observe the link night described in the findings."
                if pv is not None and np.isfinite(pv) else ""
            )
            lines.append(
                f"**{f}**: no offset can be measured from this data, because it "
                f"never shared a night with {ref}." + extra
            )
            continue
        lines.append(
            f"Set **{f}** to **{int(round(v)):+d} steps** "
            f"(fitted {v:+.2f} +/- {fit.offsets_se[f]:.2f})."
        )
    k = fit.temp_coeff
    if not np.isfinite(k):
        lines.append(
            "Temperature compensation: **leave it off** - this data contains no "
            "usable temperature sensor, so no coefficient could be measured."
        )
        return lines
    lines.append(
        f"Temperature compensation: **{k:+.2f} steps per C** "
        f"(i.e. move {'inward' if k < 0 else 'outward'} as temperature falls)."
    )
    lines.append(
        "Verify the compensation sign before trusting it overnight: note the "
        "focus position after an autofocus run, wait for the temperature to fall "
        "by about a degree, and run autofocus again. If the position moves in the "
        "same direction the coefficient predicts, the sign is right; if it moves "
        "the other way, negate it in the driver."
    )
    lines.append(
        "Keep the offsets and the compensation coefficient together: they were "
        "fitted jointly, and applying one without the other will drift."
    )
    return lines
