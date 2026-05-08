"""Shared helpers for tracking-record based smoothers."""

from __future__ import annotations

import numpy as np


def record_arrays(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, 6D states, and 6x6 covariances from tracking records."""

    times = np.asarray([float(record["time_s"]) for record in records], dtype=float)
    states = np.stack([np.asarray(record["state"], dtype=float).reshape(6) for record in records])
    covariances = np.stack(
        [np.asarray(record["covariance"], dtype=float).reshape(6, 6) for record in records]
    )
    return times, states, covariances


def copy_record(record: dict[str, object]) -> dict[str, object]:
    """Return a shallow record copy that copies NumPy arrays by value."""

    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in record.items()}


def symmetrized(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    return 0.5 * (matrix + matrix.T)
