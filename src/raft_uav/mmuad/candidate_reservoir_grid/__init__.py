"""Compatibility package with validated reservoir offset-grid specifications.

The maintained implementation lives in the sibling
``candidate_reservoir_grid.py`` module. This package preserves the public import
path while rejecting ambiguous or non-finite offset-grid specifications and
removing repeated values that would otherwise rerun identical configurations.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate reservoir grid from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _parse_offset_specs(specs: Sequence[str]) -> list[tuple[str, tuple[float, ...]]]:
    """Parse unique finite offset grids without redundant configurations."""

    parsed: list[tuple[str, tuple[float, ...]]] = []
    seen_names: set[str] = set()
    for spec in specs:
        text = str(spec)
        if "=" not in text:
            raise ValueError(f"offset grid spec must be NAME=v1,v2,...; got {spec!r}")
        name, values_text = text.split("=", 1)
        name = name.strip()
        value_tokens = [token.strip() for token in values_text.split(",")]
        if not name or not value_tokens or any(not token for token in value_tokens):
            raise ValueError(f"invalid offset grid spec {spec!r}")
        if name in seen_names:
            raise ValueError(f"duplicate offset grid name {name!r}")

        values: list[float] = []
        for token in value_tokens:
            try:
                value = float(token)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid offset value {token!r} for {name!r}") from exc
            if not np.isfinite(value):
                raise ValueError(f"offset values for {name!r} must be finite")
            if value not in values:
                values.append(value)

        seen_names.add(name)
        parsed.append((name, tuple(values)))
    return parsed


_IMPL._parse_offset_specs = _parse_offset_specs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_parse_offset_specs"] = _parse_offset_specs

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
