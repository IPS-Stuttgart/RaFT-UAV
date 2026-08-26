"""Measurement converters that consume learned heteroscedastic covariance columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.numeric import optional_float
from raft_uav.uncertainty import covariance_from_row

_COVARIANCE_LAYOUTS = {
    2: (("ee", "nn"), ((0, 1, "en"),)),
    3: (
        ("ee", "nn", "uu"),
        ((0, 1, "en"), (0, 2, "eu"), (1, 2, "nu")),
    ),
}
_RADAR_VELOCITY_COLUMNS = (
    "velocity_east_mps",
    "velocity_north_mps",
    "velocity_down_mps",
)
_RADAR_VELOCITY_ERROR = (
    "radar velocity components must be all missing or a complete finite "
    "east/north/down triplet"
)


def _finite_covariance_value(value: object) -> float | None:
    """Return one finite real covariance entry, or ``None`` when unavailable."""

    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _positive_covariance_value(value: object) -> float | None:
    """Return one positive finite variance, or ``None`` when unavailable."""

    number = _finite_covariance_value(value)
    return number if number is not None and number > 0.0 else None


def _covariance_series(
    covariance: np.ndarray,
    names: tuple[str, ...],
    cross_terms: tuple[tuple[int, int, str], ...],
    *,
    prefix: str,
) -> pd.Series:
    values = {
        f"{prefix}_{name}": covariance[index, index]
        for index, name in enumerate(names)
    }
    values.update(
        {
            f"{prefix}_{suffix}": covariance[first, second]
            for first, second, suffix in cross_terms
        }
    )
    return pd.Series(values)


def _covariance_with_partial_learned_overrides(
    row: pd.Series,
    dim: int,
    fallback: np.ndarray,
) -> np.ndarray:
    """Overlay available learned axes on association/default covariance.

    Partial uncertainty models are intentionally supported when the training data
    exposes only a subset of position dimensions. Keep learned variances for those
    dimensions instead of discarding the whole learned covariance block.
    """

    try:
        names, cross_terms = _COVARIANCE_LAYOUTS[dim]
    except KeyError as exc:
        raise ValueError("dim must be 2 or 3") from exc

    base = covariance_from_row(
        row,
        dim,
        fallback,
        prefixes=("association_cov",),
    )
    learned_diagonal = tuple(
        _positive_covariance_value(row.get(f"cov_{name}")) for name in names
    )
    if not any(value is not None for value in learned_diagonal):
        return base

    preferred_candidate = base.copy()
    salvage_candidate = base.copy()
    for index, value in enumerate(learned_diagonal):
        if value is not None:
            preferred_candidate[index, index] = value
            salvage_candidate[index, index] = value

    for first, second, suffix in cross_terms:
        value = _finite_covariance_value(row.get(f"cov_{suffix}"))
        learned_cross_term = 0.0 if value is None else value
        if (
            learned_diagonal[first] is not None
            and learned_diagonal[second] is not None
        ):
            preferred_candidate[first, second] = preferred_candidate[second, first] = (
                learned_cross_term
            )
        if (
            learned_diagonal[first] is not None
            or learned_diagonal[second] is not None
        ):
            salvage_candidate[first, second] = salvage_candidate[second, first] = (
                learned_cross_term
            )

    salvage = covariance_from_row(
        _covariance_series(
            salvage_candidate,
            names,
            cross_terms,
            prefix="salvage",
        ),
        dim,
        base,
        prefixes=("salvage",),
    )
    return covariance_from_row(
        _covariance_series(
            preferred_candidate,
            names,
            cross_terms,
            prefix="candidate",
        ),
        dim,
        salvage,
        prefixes=("candidate",),
    )


def rf_measurements_to_enu_with_uncertainty(
    rf: pd.DataFrame,
    *,
    default_std_m: float = 75.0,
) -> list[TrackingMeasurement]:
    """Convert normalized RF rows to measurements using row-wise covariance when present.

    The function is intended for frames that have already been normalized by
    :func:`raft_uav.io.aerpaw.normalize_rf` and optionally augmented with
    ``HeteroscedasticUncertaintyModel.apply_rf``.  It prefers learned
    ``cov_*`` columns, falls back to association covariance columns when those
    are present, and otherwise uses the historical CEP/default-std covariance.
    """

    default_std = _require_positive_float(default_std_m, name="default_std_m")
    measurements: list[TrackingMeasurement] = []
    for position, (_, row) in enumerate(rf.iterrows()):
        std_value = rf["std_m"].iloc[position] if "std_m" in rf.columns else None
        std_m = _positive_float(std_value) or default_std
        fallback = np.diag([std_m**2, std_m**2])
        measurements.append(
            TrackingMeasurement(
                time_s=_finite_real_scalar(
                    rf["time_s"].iloc[position],
                    name="time_s",
                ),
                vector=np.array(
                    [
                        _finite_real_scalar(
                            rf["east_m"].iloc[position],
                            name="east_m",
                        ),
                        _finite_real_scalar(
                            rf["north_m"].iloc[position],
                            name="north_m",
                        ),
                    ]
                ),
                covariance=_covariance_with_partial_learned_overrides(
                    row,
                    2,
                    fallback,
                ),
                source="rf",
            )
        )
    return measurements


def radar_measurements_to_enu_with_uncertainty(
    radar: pd.DataFrame,
    *,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
    default_velocity_std_mps: float = 12.0,
) -> list[TrackingMeasurement]:
    """Convert normalized radar rows to measurements using row-wise covariance.

    If Fortem velocity components are available, the returned measurement is
    six-dimensional.  The learned covariance is applied to the position block
    and the historical fixed velocity covariance is retained for the velocity
    block.
    """

    default_xy_std = _require_positive_float(default_xy_std_m, name="default_xy_std_m")
    default_z_std = _require_positive_float(default_z_std_m, name="default_z_std_m")
    default_velocity_std = _require_positive_float(
        default_velocity_std_mps,
        name="default_velocity_std_mps",
    )
    position_fallback = np.diag([default_xy_std**2, default_xy_std**2, default_z_std**2])
    measurements: list[TrackingMeasurement] = []
    for position_index, (_, row) in enumerate(radar.iterrows()):
        position = np.array(
            [
                _finite_real_scalar(
                    radar["east_m"].iloc[position_index],
                    name="east_m",
                ),
                _finite_real_scalar(
                    radar["north_m"].iloc[position_index],
                    name="north_m",
                ),
                _finite_real_scalar(
                    radar["up_m"].iloc[position_index],
                    name="up_m",
                ),
            ]
        )
        position_covariance = _covariance_with_partial_learned_overrides(
            row,
            3,
            position_fallback,
        )
        velocity = _radar_velocity_vector_enu(row)
        if velocity is None:
            vector = position
            covariance = position_covariance
        else:
            vector = np.concatenate([position, velocity])
            covariance = np.zeros((6, 6), dtype=float)
            covariance[:3, :3] = position_covariance
            covariance[3:, 3:] = np.diag([default_velocity_std**2] * 3)
        measurements.append(
            TrackingMeasurement(
                time_s=_finite_real_scalar(
                    radar["time_s"].iloc[position_index],
                    name="time_s",
                ),
                vector=vector,
                covariance=covariance,
                source="radar",
            )
        )
    return measurements


def _has_velocity_value(value: object) -> bool:
    """Return whether one component contains information rather than a null marker."""

    if value is None or np.ma.is_masked(value):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return True


def _radar_velocity_vector_enu(row: pd.Series) -> np.ndarray | None:
    available = {
        column: _has_velocity_value(row[column])
        for column in _RADAR_VELOCITY_COLUMNS
        if column in row.index
    }
    if not any(available.values()):
        return None
    if not all(available.get(column, False) for column in _RADAR_VELOCITY_COLUMNS):
        raise ValueError(_RADAR_VELOCITY_ERROR)
    try:
        return np.array(
            [
                _finite_real_scalar(
                    row["velocity_east_mps"],
                    name="velocity_east_mps",
                ),
                _finite_real_scalar(
                    row["velocity_north_mps"],
                    name="velocity_north_mps",
                ),
                -_finite_real_scalar(
                    row["velocity_down_mps"],
                    name="velocity_down_mps",
                ),
            ],
            dtype=float,
        )
    except ValueError as exc:
        raise ValueError(_RADAR_VELOCITY_ERROR) from exc


def _finite_real_scalar(value: object, *, name: str) -> float:
    """Return one finite real scalar without lossy NumPy coercion."""

    number = optional_float(value)
    if number is None:
        raise ValueError(f"{name} must be a finite real scalar")
    return number


def _require_positive_float(value: object, *, name: str) -> float:
    number = _finite_real_scalar(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def _positive_float(value: object) -> float | None:
    try:
        number = _finite_real_scalar(value, name="value")
    except ValueError:
        return None
    return number if number > 0.0 else None
