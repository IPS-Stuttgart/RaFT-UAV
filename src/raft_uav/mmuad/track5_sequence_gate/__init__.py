"""Compatibility package validating Track 5 sequence-gate weights.

The maintained implementation lives in the sibling ``track5_sequence_gate.py``
module. This package preserves the public import path while validating every
configured weight before duplicate sequence rows are averaged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_sequence_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_sequence_gate_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load Track 5 sequence-gate implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _validate_weight(value: Any, *, name: str) -> float:
    """Return a finite non-Boolean real scalar weight in the unit interval."""

    message = f"{name} must be finite and in [0, 1], got {value!r}"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or scalar.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        weight = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError(message)
    return weight


def _sequence_weight_map(weights: pd.DataFrame) -> dict[str, float]:
    """Validate each weight row before averaging duplicate sequence entries."""

    rows = pd.DataFrame(weights).copy()
    if rows.empty:
        return {}
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    weight_column = _IMPL._first_present(rows, _IMPL.WEIGHT_ALIASES)
    if sequence_column is None:
        raise ValueError(
            f"sequence weights missing one of columns: {_IMPL.SEQUENCE_ALIASES}"
        )
    if weight_column is None:
        raise ValueError(f"sequence weights missing one of columns: {_IMPL.WEIGHT_ALIASES}")

    rows["__sequence_id"] = rows[sequence_column].map(_IMPL._sequence_weight_key)
    rows = rows.loc[rows["__sequence_id"].notna()].copy()
    if rows.empty:
        return {}

    out: dict[str, float] = {}
    for sequence, group in rows.groupby("__sequence_id", sort=True):
        validated = [
            _validate_weight(value, name=f"sequence weight for {sequence}")
            for value in group[weight_column].tolist()
        ]
        out[str(sequence)] = float(np.mean(validated))
    return out


_IMPL._validate_weight = _validate_weight
_IMPL._sequence_weight_map = _sequence_weight_map

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_weight"] = _validate_weight
globals()["_sequence_weight_map"] = _sequence_weight_map

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
