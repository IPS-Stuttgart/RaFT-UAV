"""Reject unsupported RF vector dimensions during delayed initialization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

_SUPPORTED_RF_VECTOR_DIMENSIONS = frozenset({2, 3, 6})
_PATCH_MARKER = "_raft_uav_delayed_initialization_rf_dimension_patch_applied"


def apply_delayed_initialization_rf_dimension_patch(delayed_module: Any) -> None:
    """Install RF-window validation for supported position/state dimensions."""

    if getattr(delayed_module, _PATCH_MARKER, False):
        return

    def _first_rf_window(
        rf_measurements: Iterable[Any],
        *,
        window_s: float,
    ) -> list[tuple[float, np.ndarray]]:
        valid_measurements: list[tuple[float, np.ndarray]] = []
        for measurement in rf_measurements:
            try:
                vector = np.asarray(
                    delayed_module._measurement_field(measurement, "vector", []),
                    dtype=float,
                ).reshape(-1)
                time_s = float(
                    delayed_module._measurement_field(measurement, "time_s")
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                vector.size not in _SUPPORTED_RF_VECTOR_DIMENSIONS
                or not np.isfinite(time_s)
                or not np.isfinite(vector).all()
            ):
                continue
            valid_measurements.append((time_s, vector))

        if not valid_measurements:
            return []
        start = min(time_s for time_s, _ in valid_measurements)
        return [
            (time_s, vector)
            for time_s, vector in valid_measurements
            if time_s <= start + window_s
        ]

    delayed_module._first_rf_window = _first_rf_window
    setattr(delayed_module, _PATCH_MARKER, True)
