"""Range-adaptive radar covariance for tracklet-Viterbi association.

The base tracklet-Viterbi implementation uses one Cartesian covariance for all
selected radar rows.  This wrapper keeps the existing retention-aware Viterbi
path but passes an explicit per-row covariance callback into scoring and replay
so long-range radar rows are down-weighted according to their ``range_m`` field.

If an upstream runner already attached learned ``cov_*`` or ``association_cov_*``
columns, those calibrated row-wise covariances take precedence over the generic
range heuristic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines import tracklet_viterbi as _base
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi_retention import (
    run_async_cv_baseline_with_tracklet_viterbi_association as _run_retention_association,
)
from raft_uav.uncertainty import covariance_from_row

TrackletViterbiAssociationConfig = _base.TrackletViterbiAssociationConfig
DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE = True
DEFAULT_RADAR_RANGE_XY_FLOOR_STD_M = 20.0
DEFAULT_RADAR_RANGE_Z_FLOOR_STD_M = 30.0
DEFAULT_RADAR_RANGE_XY_SCALE = 0.035
DEFAULT_RADAR_RANGE_Z_SCALE = 0.050


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
    """Run retention-aware Viterbi with range-adaptive radar covariance."""

    cfg = config or TrackletViterbiAssociationConfig()
    radar_covariance_fn = _range_adaptive_covariance_fn(cfg)
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
        radar_covariance_fn=radar_covariance_fn,
    )


def _range_adaptive_covariance_fn(
    config: TrackletViterbiAssociationConfig,
) -> _base.RadarCovarianceFn:
    """Return a per-row covariance callback without patching module globals."""

    def radar_covariance_fn(
        row: pd.Series,
        default_covariance: np.ndarray,
    ) -> np.ndarray:
        row_covariance = _radar_row_covariance(row, default_covariance, config)
        _write_radar_covariance_diagnostics(row, row_covariance, default_covariance)
        return row_covariance

    return radar_covariance_fn


def _radar_row_covariance(
    row: pd.Series,
    default_covariance: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Return range-adaptive ENU radar covariance for one radar row.

    The fixed covariance remains a lower bound.  Long-range radar rows are
    down-weighted by inflating horizontal and vertical standard deviations from
    ``range_m``.  This approximates angular-error growth without requiring a
    native polar radar update.
    """

    default_covariance = np.asarray(default_covariance, dtype=float)
    row_covariance = covariance_from_row(row, 3, default_covariance)
    if _has_row_position_covariance(row):
        return row_covariance

    if not bool(
        getattr(
            config,
            "use_range_adaptive_radar_covariance",
            DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE,
        )
    ):
        return default_covariance

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


def _has_row_position_covariance(row: pd.Series) -> bool:
    """Return whether a row carries a complete positive 3-D covariance diagonal."""

    for prefix in ("association_cov", "cov"):
        has_diagonal = all(
            _positive_float(row.get(f"{prefix}_{suffix}")) is not None
            for suffix in ("ee", "nn", "uu")
        )
        if has_diagonal:
            return True
    return False


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


def _write_radar_covariance_diagnostics(
    row: pd.Series,
    row_covariance: np.ndarray,
    default_covariance: np.ndarray,
) -> None:
    """Attach selected-row covariance diagnostics for ablation analysis."""

    row_covariance = np.asarray(row_covariance, dtype=float)
    default_covariance = np.asarray(default_covariance, dtype=float)
    row["association_radar_xy_std_m"] = float(np.sqrt(max(row_covariance[0, 0], 0.0)))
    row["association_radar_z_std_m"] = float(np.sqrt(max(row_covariance[2, 2], 0.0)))
    row["association_radar_covariance_adaptive"] = bool(
        not np.allclose(row_covariance, default_covariance)
    )
