"""Conformal uncertainty utilities for honest empirical error radii."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float


@dataclass(frozen=True)
class ConformalRadius:
    """Split-conformal scalar error radius."""

    radius_m: float
    alpha: float
    sample_count: int

    def contains(self, errors_m: Sequence[float]) -> np.ndarray:
        errors = np.asarray(errors_m, dtype=float)
        return errors <= float(self.radius_m)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "radius_m": float(self.radius_m),
            "alpha": float(self.alpha),
            "sample_count": int(self.sample_count),
        }


def _is_missing_or_nonfinite_error(value: object) -> bool:
    """Return whether one malformed numeric parse represents an ignorable gap."""

    seen_array_ids: set[int] = set()
    while isinstance(value, np.ndarray):
        if value.ndim != 0:
            return False
        array_id = id(value)
        if array_id in seen_array_ids:
            return False
        seen_array_ids.add(array_id)
        value = value.item()

    if value is None or np.ma.is_masked(value):
        return True
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        return False

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return True

    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return not np.isfinite(number)


def _finite_calibration_errors(errors_m: Sequence[float]) -> np.ndarray:
    """Normalize real calibration errors without lossy numeric coercion."""

    try:
        masked_errors = np.ma.asarray(errors_m, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("errors_m must contain real scalar values") from exc

    values = np.asarray(masked_errors.data, dtype=object).reshape(-1)
    mask = np.ma.getmaskarray(masked_errors).reshape(-1)
    normalized: list[float] = []
    for value, is_masked in zip(values, mask, strict=True):
        if bool(is_masked):
            continue
        error = optional_float(value)
        if error is None:
            if _is_missing_or_nonfinite_error(value):
                continue
            raise ValueError("errors_m must contain real scalar values")
        if error < 0.0:
            raise ValueError("errors_m must contain only non-negative values")
        normalized.append(error)
    return np.asarray(normalized, dtype=float)


def fit_conformal_radius(
    errors_m: Sequence[float],
    *,
    alpha: float = 0.1,
) -> ConformalRadius:
    """Fit a split-conformal radius from non-negative calibration errors."""

    normalized_alpha = optional_float(alpha)
    if normalized_alpha is None or not 0.0 < normalized_alpha < 1.0:
        raise ValueError("alpha must be a finite real scalar in (0, 1)")

    errors = _finite_calibration_errors(errors_m)
    if errors.size == 0:
        return ConformalRadius(float("nan"), normalized_alpha, 0)
    n = errors.size
    rank = int(np.ceil((n + 1) * (1.0 - normalized_alpha)))
    rank = min(max(rank, 1), n)
    radius = float(np.partition(errors, rank - 1)[rank - 1])
    return ConformalRadius(radius, normalized_alpha, int(n))


def fit_conformal_radii_by_group(
    frame: pd.DataFrame,
    *,
    error_column: str = "error_3d_m",
    group_column: str = "phase",
    alpha: float = 0.1,
) -> dict[str, ConformalRadius]:
    """Fit conformal radii per phase/domain group."""

    if frame.empty:
        return {}
    return {
        str(group): fit_conformal_radius(group_frame[error_column], alpha=alpha)
        for group, group_frame in frame.groupby(group_column, sort=True)
        if error_column in group_frame.columns
    }


def apply_group_conformal_radius(
    frame: pd.DataFrame,
    radii: Mapping[str, ConformalRadius],
    *,
    group_column: str = "phase",
    output_column: str = "conformal_radius_m",
) -> pd.DataFrame:
    """Append a conformal radius selected by group label.

    Rows whose group was not calibrated receive ``NaN`` rather than borrowing an
    arbitrary radius from another group.
    """

    out = frame.copy()
    groups = out.get(group_column, pd.Series("", index=out.index, dtype=str))
    out[output_column] = [
        float(radius.radius_m)
        if (radius := radii.get(str(value))) is not None
        else float("nan")
        for value in groups
    ]
    return out
