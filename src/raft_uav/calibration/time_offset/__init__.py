"""Compatibility fixes for pooled radar time-offset calibration.

The maintained implementation lives in the sibling ``time_offset.py`` module.
This package preserves the public import path while pooling per-frame radar
oracle errors exactly and rejecting malformed time-offset sweep inputs before
they can produce invalid timestamps or misleading calibration rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import nearest_candidate_oracle

_IMPL_PATH = Path(__file__).resolve().parent.parent / "time_offset.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._time_offset_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load time-offset calibration implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_APPLY_TIME_OFFSET = _IMPL.apply_time_offset
_ORIGINAL_AGGREGATE_MEASUREMENT_TIME_OFFSET_SWEEP = (
    _IMPL.aggregate_measurement_time_offset_sweep
)


def _finite_offset_seconds(value: Any) -> float:
    """Return a finite non-Boolean scalar time offset in seconds."""

    message = "offset_s must be a finite real scalar"
    scalar = value
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        scalar = value.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, (complex, np.complexfloating)):
        raise ValueError(message)
    try:
        offset = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(offset):
        raise ValueError(message)
    return offset


def _validated_offsets(offsets_s: Iterable[Any]) -> list[float]:
    """Materialize and validate every requested sweep offset."""

    return [_finite_offset_seconds(offset) for offset in offsets_s]


def _measurement_dimensions(value: Any) -> int:
    """Return a supported measurement dimension without lossy coercion."""

    message = "dimensions must be 2 or 3"
    scalar = value
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        scalar = value.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(message)
    dimensions = int(numeric)
    if dimensions not in (2, 3):
        raise ValueError(message)
    return dimensions


def apply_time_offset(
    frame: pd.DataFrame,
    offset_s: float | None,
    *,
    time_column: str = "time_s",
    copy_uncorrected: bool = True,
) -> pd.DataFrame:
    """Shift timestamps after validating the requested correction."""

    validated_offset = (
        None if offset_s is None else _finite_offset_seconds(offset_s)
    )
    return _ORIGINAL_APPLY_TIME_OFFSET(
        frame,
        validated_offset,
        time_column=time_column,
        copy_uncorrected=copy_uncorrected,
    )


def aggregate_radar_time_offset_sweep(
    training_pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    offsets_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Aggregate radar oracle errors exactly across all training flights."""

    offsets = _validated_offsets(offsets_s)
    rows: list[dict[str, float]] = []
    for offset in offsets:
        errors_3d: list[np.ndarray] = []
        errors_2d: list[np.ndarray] = []
        frame_count = 0
        for radar, truth in training_pairs:
            if radar.empty or truth.empty:
                continue
            frame_count += _IMPL._radar_frame_count(radar)
            selected = nearest_candidate_oracle(
                radar,
                truth,
                time_offset_s=offset,
                max_time_delta_s=max_time_delta_s,
            )
            if selected.empty:
                continue
            selected_3d = pd.to_numeric(
                selected["oracle_error_3d_m"],
                errors="coerce",
            ).to_numpy(dtype=float)
            selected_2d = pd.to_numeric(
                selected["oracle_error_2d_m"],
                errors="coerce",
            ).to_numpy(dtype=float)
            errors_3d.append(selected_3d[np.isfinite(selected_3d)])
            errors_2d.append(selected_2d[np.isfinite(selected_2d)])

        pooled_3d = _IMPL._concat(errors_3d)
        pooled_2d = _IMPL._concat(errors_2d)
        row = {
            "time_offset_s": offset,
            "count": float(pooled_3d.size),
            "coverage": _IMPL._coverage(pooled_3d.size, frame_count),
        }
        row.update(_IMPL._stats(pooled_3d, "3d"))
        row.update(_IMPL._stats(pooled_2d, "2d"))
        rows.append(row)
    return pd.DataFrame.from_records(
        rows,
        columns=["time_offset_s", *_IMPL.PAPER_METRIC_COLUMNS],
    )


def aggregate_measurement_time_offset_sweep(
    training_pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    offsets_s: Iterable[float],
    *,
    dimensions: int,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Aggregate measurement errors after validating sweep controls."""

    offsets = _validated_offsets(offsets_s)
    validated_dimensions = _measurement_dimensions(dimensions)
    return _ORIGINAL_AGGREGATE_MEASUREMENT_TIME_OFFSET_SWEEP(
        training_pairs,
        offsets,
        dimensions=validated_dimensions,
        max_time_delta_s=max_time_delta_s,
    )


_IMPL.apply_time_offset = apply_time_offset
_IMPL.aggregate_radar_time_offset_sweep = aggregate_radar_time_offset_sweep
_IMPL.aggregate_measurement_time_offset_sweep = (
    aggregate_measurement_time_offset_sweep
)


globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_offset_seconds"] = _finite_offset_seconds
globals()["_validated_offsets"] = _validated_offsets
globals()["_measurement_dimensions"] = _measurement_dimensions
globals()["apply_time_offset"] = apply_time_offset
globals()["aggregate_radar_time_offset_sweep"] = (
    aggregate_radar_time_offset_sweep
)
globals()["aggregate_measurement_time_offset_sweep"] = (
    aggregate_measurement_time_offset_sweep
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
