"""Shared helpers for tracking-record based smoothers."""

from __future__ import annotations

import numpy as np

from raft_uav.numeric import optional_float


def _reject_complex_values(value: object, *, name: str) -> None:
    """Reject complex payloads before float conversion can discard information."""

    masked = np.ma.asarray(value)
    if np.iscomplexobj(masked):
        raise ValueError(f"{name} must contain only real values")
    if masked.dtype != object:
        return
    for item in masked.compressed().reshape(-1):
        if np.iscomplexobj(np.asanyarray(item)):
            raise ValueError(f"{name} must contain only real values")


def _real_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return a finite, unmasked real-valued array with the requested shape."""

    _reject_complex_values(value, name=name)
    masked = np.ma.asarray(value)
    if bool(np.ma.getmaskarray(masked).any()):
        raise ValueError(f"{name} must not contain masked values")
    try:
        cells = np.asarray(value, dtype=object).reshape(shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must have shape {shape} and contain finite real values"
        ) from exc
    parsed = np.empty(shape, dtype=float)
    for index, item in np.ndenumerate(cells):
        number = optional_float(item)
        if number is None:
            raise ValueError(
                f"{name} must have shape {shape} and contain finite real values"
            )
        parsed[index] = number
    return parsed


def _real_scalar(value: object, *, name: str) -> float:
    """Return a finite, unmasked real scalar without discarding information."""

    _reject_complex_values(value, name=name)
    masked = np.ma.asarray(value)
    if masked.ndim != 0 or np.ma.is_masked(masked):
        raise ValueError(f"{name} must be an unmasked real scalar")
    parsed = optional_float(masked.item())
    if parsed is None:
        raise ValueError(f"{name} must be a finite unmasked real scalar")
    return parsed


def record_arrays(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, 6D states, and 6x6 covariances from tracking records."""

    times = np.asarray(
        [_real_scalar(record["time_s"], name="time_s") for record in records],
        dtype=float,
    )
    states = np.stack(
        [_real_array(record["state"], name="state", shape=(6,)) for record in records]
    )
    covariances = np.stack(
        [
            _real_array(record["covariance"], name="covariance", shape=(6, 6))
            for record in records
        ]
    )
    return times, states, covariances


def copy_record(record: dict[str, object]) -> dict[str, object]:
    """Return a shallow record copy that copies NumPy arrays by value."""

    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in record.items()
    }


def symmetrized(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    return 0.5 * (matrix + matrix.T)
