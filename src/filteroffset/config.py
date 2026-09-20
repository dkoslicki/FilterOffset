"""Run configuration, loadable from YAML so a run is reproducible."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .blocks import BlockParams
from .model import ModelSpec
from .profiles import ProfileParams
from .qc import QcParams


@dataclass
class Config:
    """Everything a run needs, in one serialisable object."""

    inputs: list[str] = field(default_factory=list)
    outdir: str = "filteroffset_out"
    recursive: bool = True
    cache: str | None = None
    backends: list[str] = field(default_factory=lambda: ["astap", "internal"])
    primary_backend: str = "internal"
    metric: str = "hfd_median_bright"
    image_types: list[str] | None = field(default_factory=lambda: ["light"])
    workers: int | None = None
    theme: str = "light"
    #: "filter" draws each filter in the colour of the light it passes
    #: (narrowband dashed); "accessible" uses the CVD-validated palette.
    palette: str = "filter"
    step_microns: float | None = None
    skip_measure: bool = False
    progress: bool = True
    #: Additional FITS keyword aliases, canonical_name -> [KEYWORDS...].
    extra_aliases: dict[str, list[str]] = field(default_factory=dict)

    profiles: ProfileParams = field(default_factory=ProfileParams)
    blocks: BlockParams = field(default_factory=BlockParams)
    qc: QcParams = field(default_factory=QcParams)
    model: ModelSpec = field(default_factory=ModelSpec)

    # ------------------------------------------------------------------ YAML
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out

    def dump(self, path: str) -> str:
        import yaml

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def load(cls, path: str) -> Config:
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = dict(raw or {})
        nested = {
            "profiles": ProfileParams,
            "blocks": BlockParams,
            "qc": QcParams,
            "model": ModelSpec,
        }
        kwargs: dict[str, Any] = {}
        known = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in known:
                raise ValueError(f"unknown configuration key: {key!r}")
            if key in nested and isinstance(value, dict):
                kwargs[key] = _build(nested[key], value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def _build(kls, values: dict[str, Any]):
    allowed = {f.name for f in fields(kls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"unknown keys for {kls.__name__}: {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )
    coerced = dict(values)
    # dataclass fields declared as tuples must not silently become lists.
    for f in fields(kls):
        if (
            f.name in coerced
            and isinstance(coerced[f.name], list)
            and "tuple" in str(f.type)
        ):
            coerced[f.name] = tuple(coerced[f.name])
    return kls(**coerced)


def default_config(inputs: Sequence[str] | None = None) -> Config:
    cfg = Config(inputs=list(inputs or []))
    return cfg
