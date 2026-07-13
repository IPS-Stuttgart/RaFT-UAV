"""Compatibility wrapper rejecting ambiguous reservoir offset grids.

The maintained implementation lives in the sibling
``candidate_reservoir_grid.py`` module. This package preserves the public import
path while rejecting duplicate branch or source offset names instead of silently
letting later specifications overwrite earlier values.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate reservoir grid implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_PARSE_OFFSET_SPECS = _IMPL._parse_offset_specs


def _parse_offset_specs(
    specs: Sequence[str],
) -> list[tuple[str, tuple[float, ...]]]:
    """Parse offset specifications while rejecting repeated parameter names."""

    parsed = _ORIGINAL_PARSE_OFFSET_SPECS(specs)
    seen: set[str] = set()
    duplicates: list[str] = []
    for name, _ in parsed:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        joined = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"offset grid specs contain duplicate names: {joined}")
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
