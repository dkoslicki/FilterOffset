"""FilterOffset: data-driven filter offsets and temperature compensation for autofocus.

The package turns a directory of light frames into:

* per-filter focus offsets (in focuser steps) suitable for pasting into APT,
  N.I.N.A., SGP, etc;
* a temperature compensation coefficient (steps per degree C);
* diagnostic plots and a written report that says whether the underlying data
  actually supports those numbers.

See :mod:`filteroffset.cli` for the command line entry point.
"""

__version__ = "0.6.0"

__all__ = ["__version__"]
