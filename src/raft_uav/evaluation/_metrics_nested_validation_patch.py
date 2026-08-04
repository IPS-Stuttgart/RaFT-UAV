"""Reject recursively boxed Boolean and complex metric inputs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import metrics as _metrics

_ArrayPredicate = Callable[[np.ndarray], bool]
_ORIGINAL_REJECT_COMPLEX_VALUES = _metrics._reject_complex_values
_ORIGINAL_REJECT_BOOLEAN_VALUES = _metrics._reject_boolean_values


def _nested_payload_matches(
    value: object,
    *,
    predicate: _ArrayPredicate,
    seen: set[int],
) -> bool:
    """Return whether a nested scalar container matches ``predicate``."""

    if np.ma.is_masked(value):
        return False
    array = np.asanyarray(value)
    if bool(predicate(array)):
        return True
    if array.dtype != object or not isinstance(value, (np.ndarray, list, tuple)):
        return False

    identity = id(value)
    if identity in seen:
        raise ValueError("metric inputs must not contain cyclic object containers")
    seen.add(identity)
    try:
        if np.ma.isMaskedArray(array):
            items = array.compressed().reshape(-1)
        else:
            items = array.reshape(-1)
        return any(
            _nested_payload_matches(item, predicate=predicate, seen=seen)
            for item in items
        )
    finally:
        seen.remove(identity)


def _reject_complex_values(value: object, *, name: str) -> None:
    """Reject complex values at any object-container nesting depth."""

    _ORIGINAL_REJECT_COMPLEX_VALUES(value, name=name)
    if _nested_payload_matches(value, predicate=np.iscomplexobj, seen=set()):
        raise ValueError(f"{name} must contain only real values")


def _is_boolean_array(array: np.ndarray) -> bool:
    return bool(np.issubdtype(array.dtype, np.bool_))


def _reject_boolean_values(value: object, *, name: str) -> None:
    """Reject Boolean pseudo-numbers at any object-container nesting depth."""

    _ORIGINAL_REJECT_BOOLEAN_VALUES(value, name=name)
    if _nested_payload_matches(value, predicate=_is_boolean_array, seen=set()):
        raise ValueError(f"{name} must not contain Boolean values")


_metrics._reject_complex_values = _reject_complex_values
_metrics._reject_boolean_values = _reject_boolean_values
