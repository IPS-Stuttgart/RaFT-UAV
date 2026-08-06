"""Compatibility validation for MMUAD timestamp assignment inputs.

The maintained implementation lives in the sibling ``timestamp_assignment.py``
module. This package preserves the public import path while preventing Boolean
or nested non-scalar request, prediction, and tolerance values from being
interpreted as numeric timestamps.
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


def _unwrap_zero_dimensional_arrays(
    value: Any,
    *,
    non_scalar_error: str,
    invalid_error: str,
) -> Any:
    """Recursively unwrap scalar arrays without coercing nested vectors."""

    seen_array_ids: set[int] = set()
    while isinstance(value, np.ndarray):
        if np.ma.is_masked(value):
            raise ValueError(invalid_error)
        if value.ndim != 0:
            raise ValueError(non_scalar_error)
        array_id = id(value)
        if array_id in seen_array_ids:
            raise ValueError(invalid_error)
        seen_array_ids.add(array_id)
        try:
            value = value.item()
        except ValueError as exc:
            raise ValueError(invalid_error) from exc
    return value


def _materialize_timestamp_values(
    values: Iterable[float],
    *,
    argument_name: str,
) -> list[Any]:
    """Materialize timestamps and reject scalar text or nested non-scalars."""

    shape_error = f"{argument_name} must be one-dimensional"
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(shape_error)
    try:
        materialized = list(values)
    except TypeError as exc:
        raise ValueError(shape_error) from exc

    value_error = (
        f"{argument_name} must contain only finite real scalar timestamp values"
    )
    normalized: list[Any] = []
    for value in materialized:
        if np.ma.is_masked(value):
            raise ValueError(value_error)
        value = _unwrap_zero_dimensional_arrays(
            value,
            non_scalar_error=shape_error,
            invalid_error=value_error,
        )
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{argument_name} must not contain Boolean timestamp values")
        normalized.append(value)
    return normalized


def _materialize_tolerance(value: Any) -> Any:
    """Reject nested non-scalar tolerance arrays before NumPy float coercion."""

    error = "tolerance_s must be a finite nonnegative real scalar"
    if np.ma.is_masked(value):
        raise ValueError(error)
    return _unwrap_zero_dimensional_arrays(
        value,
        non_scalar_error=error,
        invalid_error=error,
    )


def optimal_timestamp_assignment(
    requested_times: Iterable[float],
    prediction_times: Iterable[float],
    *,
    tolerance_s: float,
) -> dict[int, int]:
    """Match timestamps after validating nested scalar representations."""

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
        tolerance_s=_materialize_tolerance(tolerance_s),
    )


_IMPL.optimal_timestamp_assignment = optimal_timestamp_assignment

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_unwrap_zero_dimensional_arrays"] = _unwrap_zero_dimensional_arrays
globals()["_materialize_timestamp_values"] = _materialize_timestamp_values
globals()["_materialize_tolerance"] = _materialize_tolerance
globals()["optimal_timestamp_assignment"] = optimal_timestamp_assignment

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
