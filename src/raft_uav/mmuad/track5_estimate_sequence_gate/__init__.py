"""Compatibility validation for Track 5 estimate sequence-gate configuration.

The maintained implementation lives in the sibling
``track5_estimate_sequence_gate.py`` module. This package preserves the public
import path while rejecting malformed blend weights, invalid sequence
identifiers, and duplicate normalized identifiers before they can silently
select a trajectory or fall back to the default blend weight.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_sequence_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_sequence_gate_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load estimate sequence-gate implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _validate_weight(value: Any, *, name: str) -> float:
    """Return one finite real scalar blend weight in the closed unit interval."""

    message = f"{name} must be a finite real scalar in [0, 1]"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(f"{message}: {value!r}")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{message}: {value!r}") from exc
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError(f"{message}: {value!r}")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{message}: {value!r}")
    try:
        weight = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{message}: {value!r}") from exc
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"{message}: {value!r}")
    return weight


def _sequence_weight_map(rows: Any) -> dict[str, float]:
    """Build an unambiguous weight map without discarding malformed rows."""

    frame = pd.DataFrame(rows).copy()
    sequence_column = _IMPL._first_present(frame, _IMPL.SEQUENCE_ALIASES)
    weight_column = _IMPL._first_present(frame, _IMPL.WEIGHT_ALIASES)
    if sequence_column is None or weight_column is None:
        raise ValueError(
            "sequence weight table must contain sequence_id and weight columns"
        )

    result: dict[str, float] = {}
    for index, row in frame.iterrows():
        raw_sequence_id = row[sequence_column]
        try:
            sequence_id = _IMPL.parse_official_sequence_cell(raw_sequence_id)
        except ValueError as exc:
            raise ValueError(
                "sequence weight table contains an invalid sequence identifier "
                f"at row {index}: {raw_sequence_id!r}"
            ) from exc
        if sequence_id in result:
            raise ValueError(
                "sequence weight table contains duplicate normalized sequence_id "
                f"at row {index}: {sequence_id}"
            )
        result[sequence_id] = _validate_weight(
            row[weight_column],
            name="sequence_weight",
        )
    return result


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
