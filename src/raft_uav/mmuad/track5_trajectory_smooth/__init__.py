"""Compatibility guards for Track 5 trajectory smoothing.

The maintained implementation lives in the sibling ``track5_trajectory_smooth.py``
module. This package keeps the public import path while rejecting malformed
smoothing controls, non-finite fixed-grid rows, and duplicate
``(sequence_id, time_s)`` keys before they can silently disable smoothing,
truncate configuration values, or remove submission rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_trajectory_smooth.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_trajectory_smooth_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 trajectory smoother from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_NORMALIZE = _IMPL._normalized_estimate_rows
_ORIGINAL_SMOOTH = _IMPL.smooth_track5_submission_rows
_ORIGINAL_WRITE_OUTPUTS = _IMPL.write_track5_trajectory_smooth_outputs
_REQUIRED_COLUMNS = (
    "sequence_id",
    "time_s",
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "Classification",
)
_NUMERIC_COLUMNS = ("time_s", "state_x_m", "state_y_m", "state_z_m")


def _finite_real_scalar(value: object, *, message: str) -> float:
    """Return one finite non-Boolean real scalar."""

    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or scalar.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        normalized = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(normalized):
        raise ValueError(message)
    return normalized


def _positive_finite_scalar(value: object, *, name: str) -> float:
    """Return a strictly positive finite real scalar."""

    message = f"{name} must be a positive finite real scalar"
    normalized = _finite_real_scalar(value, message=message)
    if normalized <= 0.0:
        raise ValueError(message)
    return normalized


def _nonnegative_finite_scalar(value: object, *, name: str) -> float:
    """Return a finite non-negative real scalar."""

    message = f"{name} must be a finite non-negative real scalar"
    normalized = _finite_real_scalar(value, message=message)
    if normalized < 0.0:
        raise ValueError(message)
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without lossy scalar coercion."""

    message = f"{name} must be a positive finite integer"
    normalized = _finite_real_scalar(value, message=message)
    if normalized < 1.0 or not normalized.is_integer():
        raise ValueError(message)
    return int(normalized)


def _validated_controls(
    *,
    window_s: object,
    bandwidth_s: object | None,
    blend: object,
    max_correction_m: object | None,
    min_neighbors: object,
) -> dict[str, float | int | None]:
    """Normalize every smoothing control before scientific computation."""

    normalized_window = _positive_finite_scalar(window_s, name="window_s")
    normalized_bandwidth = (
        None
        if bandwidth_s is None
        else _positive_finite_scalar(bandwidth_s, name="bandwidth_s")
    )
    normalized_blend = _finite_real_scalar(
        blend,
        message="blend must be a finite real scalar in [0, 1]",
    )
    if not 0.0 <= normalized_blend <= 1.0:
        raise ValueError("blend must be a finite real scalar in [0, 1]")
    normalized_max_correction = (
        None
        if max_correction_m is None
        else _nonnegative_finite_scalar(
            max_correction_m,
            name="max_correction_m",
        )
    )
    normalized_min_neighbors = _positive_integer(
        min_neighbors,
        name="min_neighbors",
    )
    return {
        "window_s": normalized_window,
        "bandwidth_s": normalized_bandwidth,
        "blend": normalized_blend,
        "max_correction_m": normalized_max_correction,
        "min_neighbors": normalized_min_neighbors,
    }


def _validated_input_rows(rows: object) -> pd.DataFrame:
    """Reject rows the legacy normalizer would silently drop or coerce."""

    frame = pd.DataFrame(rows).copy()
    missing = sorted(set(_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Track 5 rows missing normalized columns: {missing}")

    for column in _NUMERIC_COLUMNS:
        values = frame[column]
        boolean = values.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).to_numpy(dtype=bool)
        if boolean.any():
            indices = frame.index[boolean].tolist()
            raise ValueError(
                f"Track 5 rows contain Boolean values in {column} at row indices: "
                f"{indices}"
            )
        numeric = pd.to_numeric(values, errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if not finite.all():
            indices = frame.index[~finite].tolist()
            raise ValueError(
                f"Track 5 rows contain non-finite {column} values at row indices: "
                f"{indices}"
            )
    return frame


def _normalized_estimate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize rows without dropping malformed entries or accepting duplicate keys."""

    validated = _validated_input_rows(rows)
    normalized = _ORIGINAL_NORMALIZE(validated)
    duplicate = normalized.duplicated(["sequence_id", "time_s"], keep=False)
    if duplicate.any():
        keys = (
            normalized.loc[duplicate, ["sequence_id", "time_s"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        preview = ", ".join(
            f"({sequence_id!r}, {time_s!r})" for sequence_id, time_s in keys
        )
        raise ValueError(
            f"Track 5 rows contain duplicate sequence/timestamp keys: {preview}"
        )
    return normalized


def smooth_track5_submission_rows(
    rows: pd.DataFrame,
    *,
    window_s: float = 15.0,
    bandwidth_s: float | None = None,
    blend: float = 1.0,
    max_correction_m: float | None = 10.0,
    min_neighbors: int = 3,
):
    """Smooth only after controls have been validated without lossy coercion."""

    controls = _validated_controls(
        window_s=window_s,
        bandwidth_s=bandwidth_s,
        blend=blend,
        max_correction_m=max_correction_m,
        min_neighbors=min_neighbors,
    )
    return _ORIGINAL_SMOOTH(rows, **controls)


def write_track5_trajectory_smooth_outputs(
    *,
    rows: pd.DataFrame,
    output_dir: Path,
    template: pd.DataFrame | None = None,
    window_s: float = 15.0,
    bandwidth_s: float | None = None,
    blend: float = 1.0,
    max_correction_m: float | None = 10.0,
    min_neighbors: int = 3,
):
    """Validate rows and controls before the output directory is created."""

    _normalized_estimate_rows(rows)
    controls = _validated_controls(
        window_s=window_s,
        bandwidth_s=bandwidth_s,
        blend=blend,
        max_correction_m=max_correction_m,
        min_neighbors=min_neighbors,
    )
    return _ORIGINAL_WRITE_OUTPUTS(
        rows=rows,
        output_dir=output_dir,
        template=template,
        **controls,
    )


_IMPL._normalized_estimate_rows = _normalized_estimate_rows
_IMPL.smooth_track5_submission_rows = smooth_track5_submission_rows
_IMPL.write_track5_trajectory_smooth_outputs = write_track5_trajectory_smooth_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_real_scalar"] = _finite_real_scalar
globals()["_positive_finite_scalar"] = _positive_finite_scalar
globals()["_nonnegative_finite_scalar"] = _nonnegative_finite_scalar
globals()["_positive_integer"] = _positive_integer
globals()["_validated_controls"] = _validated_controls
globals()["_validated_input_rows"] = _validated_input_rows
globals()["_normalized_estimate_rows"] = _normalized_estimate_rows
globals()["smooth_track5_submission_rows"] = smooth_track5_submission_rows
globals()["write_track5_trajectory_smooth_outputs"] = (
    write_track5_trajectory_smooth_outputs
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
