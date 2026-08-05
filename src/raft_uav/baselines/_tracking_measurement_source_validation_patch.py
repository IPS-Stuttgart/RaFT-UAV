"""Validate source labels at the public tracking-measurement boundary."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_tracking_measurement_source_validation_patch_applied"


def _normalized_source_name(value: Any) -> str:
    """Return a trimmed non-missing string source label."""

    error = "measurement source must be a non-empty, non-missing string scalar"
    if np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0:
        raise ValueError(error)

    raw = scalar.item()
    try:
        missing = pd.isna(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if not isinstance(missing, bool | np.bool_) or bool(missing):
        raise ValueError(error)
    if not isinstance(raw, str):
        raise ValueError(error)

    source = raw.strip()
    if not source:
        raise ValueError(error)
    return source


def apply_tracking_measurement_source_validation_patch(kalman_module: Any) -> None:
    """Reject missing source provenance before calibration or gating consumes it."""

    if getattr(kalman_module, _PATCH_MARKER, False):
        return

    original_post_init = kalman_module.TrackingMeasurement.__post_init__

    @wraps(original_post_init)
    def tracking_measurement_post_init(
        self: Any,
        _apply_runtime_calibration: bool,
    ) -> None:
        source = _normalized_source_name(self.source)
        object.__setattr__(self, "source", source)
        original_post_init(self, _apply_runtime_calibration)
        object.__setattr__(self, "source", source)

    kalman_module.TrackingMeasurement.__post_init__ = tracking_measurement_post_init
    setattr(kalman_module, _PATCH_MARKER, True)
