"""Star profile measurement backends.

Two independent implementations measure the same quantity so that they can be
cross-checked against each other:

* :mod:`filteroffset.profiles.astap` shells out to ``astap_cli -extract``,
  which is Han Kleijn's independent C implementation of half-flux diameter;
* :mod:`filteroffset.profiles.internal` uses :mod:`sep` / :mod:`photutils` to
  do source extraction and a curve-of-growth half-flux radius in Python.

Both return a star table with the same columns, and both are summarised by the
*same* :func:`filteroffset.profiles.base.summarise_stars` function, so any
disagreement is attributable to detection and measurement rather than to
differing statistics.
"""

from __future__ import annotations

from .base import (
    STAR_COLUMNS,
    FrameProfile,
    ProfileBackendError,
    ProfileParams,
    summarise_stars,
)

__all__ = [
    "STAR_COLUMNS",
    "FrameProfile",
    "ProfileBackendError",
    "ProfileParams",
    "summarise_stars",
    "get_backend",
    "available_backends",
]


def get_backend(name: str):
    """Return the measurement callable for backend *name*."""
    key = str(name).strip().lower()
    if key == "astap":
        from .astap import measure_frame as fn
    elif key in ("internal", "sep", "python"):
        from .internal import measure_frame as fn
    else:
        raise ProfileBackendError(
            f"unknown star-profile backend {name!r}; expected 'astap' or 'internal'"
        )
    return fn


def available_backends() -> dict[str, str]:
    """Report which backends can actually run here, and why not if they cannot."""
    status: dict[str, str] = {}
    from .astap import astap_status

    status["astap"] = astap_status()
    try:
        import sep  # noqa: F401

        status["internal"] = "ok (sep)"
    except Exception:
        try:
            import photutils  # noqa: F401

            status["internal"] = "ok (photutils)"
        except Exception as exc:
            status["internal"] = f"unavailable: {exc}"
    return status
