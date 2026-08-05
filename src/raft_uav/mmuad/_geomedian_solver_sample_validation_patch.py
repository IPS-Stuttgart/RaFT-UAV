"""Reject lossy scalar coercion in geometric-median solver samples."""

from __future__ import annotations

from typing import Any

import numpy as np


def _coerce_real_cell(value: object, *, field: str) -> float:
    """Return one real scalar while preserving existing missing-value filtering."""

    seen_array_ids: set[int] = set()
    while isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{field} must contain scalar values")
        array_id = id(value)
        if array_id in seen_array_ids:
            raise ValueError(f"{field} contains a cyclic scalar container")
        seen_array_ids.add(array_id)
        value = value.item()

    if np.ma.is_masked(value):
        raise ValueError(f"{field} must not contain masked values")
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{field} must not contain Boolean values")
    if isinstance(value, complex | np.complexfloating):
        raise ValueError(f"{field} must not contain complex values")

    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        try:
            if int(ndim) != 0:
                raise ValueError(f"{field} must contain scalar values")
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain scalar values") from exc

    # The maintained solver filters non-finite rows after conversion. Preserve
    # that behavior for absent values while rejecting ambiguous scalar types.
    if value is None:
        return np.nan
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain real numeric values") from exc


def _object_array(value: object, *, field: str) -> np.ndarray:
    """Materialize an object array without discarding masks or complex parts."""

    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError(f"{field} must not contain masked values")
    try:
        return np.asarray(value, dtype=object)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric array") from exc


def _validated_solver_samples(
    xyz: object,
    weights: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate solver shapes and cells before converting them to floats."""

    point_cells = _object_array(xyz, field="xyz")
    weight_cells = _object_array(weights, field="weights")

    if point_cells.ndim == 1 and point_cells.size == 0:
        point_cells = np.empty((0, 3), dtype=object)
    if point_cells.ndim != 2 or point_cells.shape[1] != 3:
        raise ValueError("xyz must be a numeric 2D array with shape (n, 3)")
    if weight_cells.ndim != 1:
        raise ValueError("weights must be a numeric 1D array")
    if weight_cells.shape[0] != point_cells.shape[0]:
        raise ValueError("xyz and weights must have the same row count")

    points = np.asarray(
        [
            [_coerce_real_cell(value, field="xyz") for value in row]
            for row in point_cells
        ],
        dtype=float,
    ).reshape(point_cells.shape)
    point_weights = np.asarray(
        [_coerce_real_cell(value, field="weights") for value in weight_cells],
        dtype=float,
    )
    return points, point_weights


def install() -> None:
    """Install the lossless sample validator at the public solver boundary."""

    from raft_uav.mmuad import track5_geometric_median_ensemble as geomedian

    current: Any = geomedian._validated_solver_samples
    if getattr(current, "_raft_uav_lossless_samples", False):
        return
    _validated_solver_samples._raft_uav_lossless_samples = True  # type: ignore[attr-defined]
    geomedian._validated_solver_samples = _validated_solver_samples
