"""Conformal uncertainty utilities for honest empirical error radii."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


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


def fit_conformal_radius(errors_m: Sequence[float], *, alpha: float = 0.1) -> ConformalRadius:
    """Fit a split-conformal radius from calibration errors."""

    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if errors.size == 0:
        return ConformalRadius(float("nan"), float(alpha), 0)
    n = errors.size
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), n)
    return ConformalRadius(float(np.partition(errors, rank - 1)[rank - 1]), float(alpha), int(n))


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
    """Append a conformal radius selected by group label."""

    out = frame.copy()
    default = next(iter(radii.values()), ConformalRadius(float("nan"), 0.1, 0)).radius_m
    out[output_column] = [
        radii.get(str(value), ConformalRadius(float(default), 0.1, 0)).radius_m
        for value in out.get(group_column, pd.Series([""] * len(out)))
    ]
    return out
