"""Compatibility package with validated reservoir offset-grid specifications.

The maintained implementation lives in the sibling
``candidate_reservoir_grid.py`` module. This package preserves the public import
path while rejecting ambiguous or non-finite offset-grid specifications, removing
repeated values that would otherwise rerun identical configurations, keeping
distinct floating-point offsets distinct in per-configuration labels, and
rejecting lossy top-K coercions before oracle diagnostics are selected.
"""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from raft_uav.numeric import optional_int

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
_ORIGINAL_RUN_CANDIDATE_RESERVOIR_OFFSET_GRID = (
    _IMPL.run_candidate_reservoir_offset_grid
)


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


def _format_float(value: float) -> str:
    """Return a filename-safe shortest round-trip floating-point label."""

    text = repr(float(value))
    if text.endswith(".0") and "e" not in text.lower():
        text = text[:-2]
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _validated_top_k_values(values: object) -> tuple[int, ...]:
    """Normalize exact positive top-K values without lossy integer coercion."""

    if isinstance(values, (str, bytes)):
        raise ValueError("top_k_values must be a sequence of positive exact integers")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "top_k_values must be a sequence of positive exact integers"
        ) from exc

    normalized: list[int] = []
    for value in items:
        integer = optional_int(value)
        if integer is None or integer <= 0:
            raise ValueError("top_k_values must contain positive exact integers")
        normalized.append(integer)
    return tuple(sorted(set(normalized)))


@wraps(_ORIGINAL_RUN_CANDIDATE_RESERVOIR_OFFSET_GRID)
def run_candidate_reservoir_offset_grid(*args, **kwargs):
    """Run the grid after losslessly normalizing explicit oracle top-K values."""

    if "top_k_values" in kwargs:
        kwargs = dict(kwargs)
        kwargs["top_k_values"] = _validated_top_k_values(kwargs["top_k_values"])
    return _ORIGINAL_RUN_CANDIDATE_RESERVOIR_OFFSET_GRID(*args, **kwargs)


_IMPL._parse_offset_specs = _parse_offset_specs
_IMPL._format_float = _format_float
_IMPL.run_candidate_reservoir_offset_grid = run_candidate_reservoir_offset_grid

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_parse_offset_specs"] = _parse_offset_specs
globals()["_format_float"] = _format_float
globals()["_validated_top_k_values"] = _validated_top_k_values
globals()["run_candidate_reservoir_offset_grid"] = run_candidate_reservoir_offset_grid

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
