"""Compatibility validation for MMUAD timestamp assignment inputs.

The maintained implementation lives in the sibling ``timestamp_assignment.py``
module. This package preserves the public import path while preventing Boolean
request or prediction values from being interpreted as numeric timestamps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "timestamp_assignment.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._timestamp_assignment_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load MMUAD timestamp assignment implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_OPTIMAL_TIMESTAMP_ASSIGNMENT = _IMPL.optimal_timestamp_assignment


def _materialize_timestamp_values(
    values: Iterable[float],
    *,
    argument_name: str,
) -> list[Any]:
    """Materialize one timestamp iterable and reject Boolean scalar values."""

    try:
        materialized = list(values)
    except TypeError as exc:
        raise ValueError("timestamp arrays must be one-dimensional") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in materialized):
        raise ValueError(f"{argument_name} must not contain Boolean timestamp values")
    return materialized


def optimal_timestamp_assignment(
    requested_times: Iterable[float],
    prediction_times: Iterable[float],
    *,
    tolerance_s: float,
) -> dict[int, int]:
    """Match timestamps after rejecting semantically invalid Boolean values."""

    requests = _materialize_timestamp_values(
        requested_times,
        argument_name="requested_times",
    )
    predictions = _materialize_timestamp_values(
        prediction_times,
        argument_name="prediction_times",
    )
    return _ORIGINAL_OPTIMAL_TIMESTAMP_ASSIGNMENT(
        requests,
        predictions,
        tolerance_s=tolerance_s,
    )


_IMPL.optimal_timestamp_assignment = optimal_timestamp_assignment

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_materialize_timestamp_values"] = _materialize_timestamp_values
globals()["optimal_timestamp_assignment"] = optimal_timestamp_assignment

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
