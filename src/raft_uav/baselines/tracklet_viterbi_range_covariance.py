"""Radar covariance models for tracklet-Viterbi association.

The base tracklet-Viterbi implementation uses one Cartesian covariance for all
selected radar rows.  This wrapper keeps the existing retention-aware Viterbi
path but patches the replay and RF-anchor scoring hooks so radar rows can be
down-weighted with either the historical range-adaptive diagonal covariance or
a polar/range-bearing covariance projected into ENU at each candidate row.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines import radar_association as _radar_association
from raft_uav.baselines import tracklet_viterbi as _base
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi_retention import (
    run_async_cv_baseline_with_tracklet_viterbi_association as _run_retention_association,
)

TrackletViterbiAssociationConfig = _base.TrackletViterbiAssociationConfig
DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE = True
DEFAULT_RADAR_COVARIANCE_MODEL = "range-adaptive"
DEFAULT_RADAR_RANGE_XY_FLOOR_STD_M = 20.0
DEFAULT_RADAR_RANGE_Z_FLOOR_STD_M = 30.0
DEFAULT_RADAR_RANGE_XY_SCALE = 0.035
DEFAULT_RADAR_RANGE_Z_SCALE = 0.050
DEFAULT_RADAR_POLAR_RANGE_STD_M = 15.0
DEFAULT_RADAR_POLAR_AZIMUTH_STD_DEG = 2.0
DEFAULT_RADAR_POLAR_ELEVATION_STD_DEG = 3.0


def run_async_cv_baseline_with_tracklet_viterbi_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    candidate_catprob_threshold: float | None = 0.4,
    config: TrackletViterbiAssociationConfig | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run retention-aware Viterbi with per-row radar covariance."""

    cfg = config or TrackletViterbiAssociationConfig()
    with _range_adaptive_covariance_hooks(cfg):
        return _run_retention_association(
            rf_measurements=rf_measurements,
            radar=radar,
            acceleration_std_mps2=acceleration_std_mps2,
            radar_xy_std_m=radar_xy_std_m,
            radar_z_std_m=radar_z_std_m,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
        )


@contextmanager
def _range_adaptive_covariance_hooks(config: TrackletViterbiAssociationConfig):
    original_candidate_cost_terms = _base._candidate_cost_terms
    original_radar_row_to_measurement = _radar_association._radar_row_to_measurement

    def candidate_cost_terms_with_adaptive_covariance(
        *,
        row: pd.Series,
        position: np.ndarray,
        anchor: _base._AnchorState | None,
        covariance: np.ndarray,
        config: TrackletViterbiAssociationConfig,
    ) -> tuple[float, float, float]:
        row_covariance = _radar_row_covariance(row, covariance, config)
        return original_candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=row_covariance,
            config=config,
        )

    def radar_row_to_measurement_with_adaptive_covariance(
        row: pd.Series,
        covariance: np.ndarray,
    ) -> TrackingMeasurement:
        row_covariance = _radar_row_covariance(row, covariance, config)
        _write_radar_covariance_diagnostics(row, row_covariance, covariance, config=config)
        return original_radar_row_to_measurement(row, row_covariance)

    _base._candidate_cost_terms = candidate_cost_terms_with_adaptive_covariance
    _radar_association._radar_row_to_measurement = radar_row_to_measurement_with_adaptive_covariance
    try:
        yield
    finally:
        _base._candidate_cost_terms = original_candidate_cost_terms
        _radar_association._radar_row_to_measurement = original_radar_row_to_measurement


def _radar_row_covariance(
    row: pd.Series,
    default_covariance: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Return the configured ENU radar covariance for one radar row.

    ``range-adaptive`` preserves the historical diagonal range scaling.
    ``polar-projected`` models radar as noisy range, azimuth, and elevation and
    projects that local spherical covariance into ENU.  The supplied Cartesian
    covariance remains a per-axis lower bound in both adaptive modes.
    """

    default_covariance = np.asarray(default_covariance, dtype=float)
    model = _radar_covariance_model(config)
    if model == "fixed" or not bool(
        getattr(
            config,
            "use_range_adaptive_radar_covariance",
            DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE,
        )
    ):
        return default_covariance
    if model == "range-adaptive":
        return _range_adaptive_radar_row_covariance(row, default_covariance, config)
    if model == "polar-projected":
        return _polar_projected_radar_row_covariance(row, default_covariance, config)
    raise ValueError(f"unknown radar covariance model {model!r}")


def _radar_covariance_model(config: Any) -> str:
    raw_model = str(
        getattr(config, "radar_covariance_model", DEFAULT_RADAR_COVARIANCE_MODEL)
    )
    normalized = raw_model.strip().lower().replace("_", "-")
    aliases = {
        "cartesian": "fixed",
        "cartesian-fixed": "fixed",
        "off": "fixed",
        "false": "fixed",
        "range": "range-adaptive",
        "cartesian-range": "range-adaptive",
        "range-covariance": "range-adaptive",
        "polar": "polar-projected",
        "polar-cartesian": "polar-projected",
        "polar-projected-cartesian": "polar-projected",
    }
    return aliases.get(normalized, normalized)


def _range_adaptive_radar_row_covariance(
    row: pd.Series,
    default_covariance: np.ndarray,
    config: Any,
) -> np.ndarray:
    range_m = _base._optional_float(row.get("range_m"))
    if range_m is None or range_m <= 0.0:
        return default_covariance

    default_xy_std_m = float(
        np.sqrt(max(default_covariance[0, 0], default_covariance[1, 1], 0.0))
    )
    default_z_std_m = float(np.sqrt(max(default_covariance[2, 2], 0.0)))
    xy_std_m = max(
        default_xy_std_m,
        float(
            getattr(
                config,
                "radar_range_xy_floor_std_m",
                DEFAULT_RADAR_RANGE_XY_FLOOR_STD_M,
            )
        ),
        float(getattr(config, "radar_range_xy_scale", DEFAULT_RADAR_RANGE_XY_SCALE))
        * float(range_m),
    )
    z_std_m = max(
        default_z_std_m,
        float(getattr(config, "radar_range_z_floor_std_m", DEFAULT_RADAR_RANGE_Z_FLOOR_STD_M)),
        float(getattr(config, "radar_range_z_scale", DEFAULT_RADAR_RANGE_Z_SCALE))
        * float(range_m),
    )
    return np.diag([xy_std_m**2, xy_std_m**2, z_std_m**2])


def _polar_projected_radar_row_covariance(
    row: pd.Series,
    default_covariance: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Project noisy radar range/bearing/elevation into ENU covariance."""

    polar_covariance = _radar_row_polar_projection(row, config)
    if polar_covariance is None:
        return _range_adaptive_radar_row_covariance(row, default_covariance, config)
    return _apply_cartesian_lower_bound(polar_covariance, default_covariance)


def _radar_row_polar_projection(row: pd.Series, config: Any) -> np.ndarray | None:
    position = _radar_position_vector(row)
    if position is None:
        return None
    position_norm_m = float(np.linalg.norm(position))
    if position_norm_m <= 0.0 or not np.isfinite(position_norm_m):
        return None

    range_m = _base._optional_float(row.get("range_m"))
    if range_m is None or range_m <= 0.0:
        range_m = position_norm_m
    scaled_position = position * (float(range_m) / position_norm_m)
    east_m, north_m, up_m = [float(value) for value in scaled_position]
    horizontal_m = float(np.hypot(east_m, north_m))

    range_std_m = float(
        getattr(config, "radar_polar_range_std_m", DEFAULT_RADAR_POLAR_RANGE_STD_M)
    )
    azimuth_std_rad = np.deg2rad(
        float(
            getattr(
                config,
                "radar_polar_azimuth_std_deg",
                DEFAULT_RADAR_POLAR_AZIMUTH_STD_DEG,
            )
        )
    )
    elevation_std_rad = np.deg2rad(
        float(
            getattr(
                config,
                "radar_polar_elevation_std_deg",
                DEFAULT_RADAR_POLAR_ELEVATION_STD_DEG,
            )
        )
    )
    if min(range_std_m, azimuth_std_rad, elevation_std_rad) < 0.0:
        raise ValueError("polar radar standard deviations must be nonnegative")

    radial = scaled_position / float(range_m)
    azimuth = np.array([north_m, -east_m, 0.0], dtype=float)
    if horizontal_m > 1.0e-9:
        elevation = np.array(
            [
                -up_m * east_m / horizontal_m,
                -up_m * north_m / horizontal_m,
                horizontal_m,
            ],
            dtype=float,
        )
    else:
        elevation = np.array([0.0, 0.0, float(range_m)], dtype=float)
    jacobian = np.column_stack([radial, azimuth, elevation])
    polar_measurement_covariance = np.diag(
        [range_std_m**2, azimuth_std_rad**2, elevation_std_rad**2]
    )
    projected = jacobian @ polar_measurement_covariance @ jacobian.T
    return _symmetrize(projected)


def _radar_position_vector(row: pd.Series) -> np.ndarray | None:
    required = ("east_m", "north_m", "up_m")
    if not all(column in row for column in required):
        return None
    position = np.array([float(row[column]) for column in required], dtype=float)
    if not np.isfinite(position).all():
        return None
    return position


def _apply_cartesian_lower_bound(
    covariance: np.ndarray,
    default_covariance: np.ndarray,
) -> np.ndarray:
    covariance = _symmetrize(np.asarray(covariance, dtype=float))
    default_covariance = np.asarray(default_covariance, dtype=float)
    diagonal_deficit = np.maximum(
        np.diag(default_covariance) - np.diag(covariance),
        0.0,
    )
    return _symmetrize(covariance + np.diag(diagonal_deficit))


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _write_radar_covariance_diagnostics(
    row: pd.Series,
    row_covariance: np.ndarray,
    default_covariance: np.ndarray,
    *,
    config: Any | None = None,
) -> None:
    """Attach selected-row covariance diagnostics for ablation analysis."""

    row_covariance = np.asarray(row_covariance, dtype=float)
    default_covariance = np.asarray(default_covariance, dtype=float)
    row["association_radar_xy_std_m"] = float(
        np.sqrt(max(row_covariance[0, 0], row_covariance[1, 1], 0.0))
    )
    row["association_radar_z_std_m"] = float(np.sqrt(max(row_covariance[2, 2], 0.0)))
    row["association_radar_cov_en"] = float(row_covariance[0, 1])
    row["association_radar_cov_eu"] = float(row_covariance[0, 2])
    row["association_radar_cov_nu"] = float(row_covariance[1, 2])
    row["association_radar_covariance_adaptive"] = bool(
        not np.allclose(row_covariance, default_covariance)
    )
    if config is not None:
        row["association_radar_covariance_model"] = _radar_covariance_model(config)
