"""Legacy baseline wrapper using learned heteroscedastic RF/radar covariance.

The learned uncertainty model already writes row-wise ``cov_*`` columns.  This
wrapper keeps the existing loading, association, filtering, smoothing, and
metrics path, but temporarily hooks the relevant conversion/scoring functions so
``run-baseline`` consumes those row-wise covariances instead of fixed RF/radar
covariance defaults.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement, measurement_matrix
import raft_uav.baselines.radar_association as radar_association
import raft_uav.cli as base_cli
import raft_uav.io.aerpaw as aerpaw
import raft_uav.robust_cli as robust_cli
from raft_uav.uncertainty import HeteroscedasticUncertaintyModel, covariance_from_row
from raft_uav.uncertainty import load_uncertainty_model

_COVARIANCE_SUFFIXES_3D = ("ee", "nn", "uu", "en", "eu", "nu")


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``run-baseline`` with an ``--uncertainty-model`` JSON file.

    Example:
        raft-uav-heteroscedastic run-baseline DATA --flight F \
            --uncertainty-model outputs/uncertainty/model.json
    """

    args = list(sys.argv[1:] if argv is None else argv)
    uncertainty_model_path, delegated_args = _pop_uncertainty_model(args)
    if not delegated_args or delegated_args[0] != "run-baseline":
        raise SystemExit(
            "raft-uav-heteroscedastic wraps the 'run-baseline' subcommand; "
            "pass e.g. 'run-baseline DATA --flight FLIGHT --uncertainty-model MODEL.json'."
        )
    with heteroscedastic_covariance_hooks(uncertainty_model_path):
        return robust_cli.main(delegated_args)


def _pop_uncertainty_model(args: list[str]) -> tuple[Path, list[str]]:
    for index, value in enumerate(args):
        if value == "--uncertainty-model":
            if index + 1 >= len(args):
                raise SystemExit("--uncertainty-model requires a model JSON path")
            return Path(args[index + 1]), [*args[:index], *args[index + 2 :]]
        if value.startswith("--uncertainty-model="):
            return Path(value.split("=", 1)[1]), [*args[:index], *args[index + 1 :]]
    raise SystemExit("missing required --uncertainty-model MODEL.json")


@contextmanager
def heteroscedastic_covariance_hooks(
    uncertainty_model_path: Path,
) -> Iterator[HeteroscedasticUncertaintyModel]:
    """Temporarily make the legacy baseline consume learned covariance columns."""

    model = load_uncertainty_model(uncertainty_model_path)

    original_base_normalize_rf = base_cli.normalize_rf
    original_base_normalize_radar = base_cli.normalize_radar
    original_base_rf_measurements_to_enu = base_cli.rf_measurements_to_enu
    original_base_radar_measurements_to_enu = base_cli.radar_measurements_to_enu
    original_base_run_association = base_cli.run_async_cv_baseline_with_radar_association
    original_base_metrics = base_cli._baseline_metrics

    original_aerpaw_normalize_rf = aerpaw.normalize_rf
    original_aerpaw_normalize_radar = aerpaw.normalize_radar
    original_aerpaw_rf_measurements_to_enu = aerpaw.rf_measurements_to_enu
    original_aerpaw_radar_measurements_to_enu = aerpaw.radar_measurements_to_enu

    original_assoc_run = radar_association.run_async_cv_baseline_with_radar_association
    original_assoc_scoring = radar_association._nis_scored_candidates

    def normalize_rf_hook(*args: object, **kwargs: object) -> pd.DataFrame:
        frame = original_base_normalize_rf(*args, **kwargs)
        return _apply_model(model, frame, source="rf")

    def normalize_radar_hook(*args: object, **kwargs: object) -> pd.DataFrame:
        frame = original_base_normalize_radar(*args, **kwargs)
        return _apply_model(model, frame, source="radar")

    def rf_measurements_hook(*args: object, **kwargs: object) -> list[TrackingMeasurement]:
        if not args or not isinstance(args[0], pd.DataFrame) or "east_m" not in args[0].columns:
            return original_base_rf_measurements_to_enu(*args, **kwargs)
        frame = _apply_model(model, args[0], source="rf")
        return rf_measurements_to_enu_with_row_covariance(
            frame,
            default_std_m=float(kwargs.get("default_std_m", 75.0)),
        )

    def radar_measurements_hook(*args: object, **kwargs: object) -> list[TrackingMeasurement]:
        if not args or not isinstance(args[0], pd.DataFrame) or "east_m" not in args[0].columns:
            return original_base_radar_measurements_to_enu(*args, **kwargs)
        frame = _apply_model(model, args[0], source="radar")
        return radar_measurements_to_enu_with_row_covariance(
            frame,
            default_xy_std_m=float(kwargs.get("default_xy_std_m", 25.0)),
            default_z_std_m=float(kwargs.get("default_z_std_m", 35.0)),
            default_velocity_std_mps=float(kwargs.get("default_velocity_std_mps", 12.0)),
            include_velocity=bool(kwargs.get("include_velocity", False)),
        )

    def association_hook(*args: object, **kwargs: object) -> tuple[list[dict[str, object]], pd.DataFrame]:
        if isinstance(kwargs.get("radar"), pd.DataFrame):
            kwargs = dict(kwargs)
            kwargs["radar"] = promote_covariance_columns_for_association(kwargs["radar"])
        return original_assoc_run(*args, **kwargs)

    def metrics_hook(*args: object, **kwargs: object) -> dict[str, object]:
        metrics = original_base_metrics(*args, **kwargs)
        metrics["uncertainty_model"] = {
            "path": str(uncertainty_model_path),
            "type": "heteroscedastic-loglinear-variance",
        }
        metrics["rf_covariance"] = "heteroscedastic learned cov_* columns; fallback CEP/default"
        metrics["radar_covariance"] = "heteroscedastic learned cov_* columns; fallback fixed radar"
        return metrics

    base_cli.normalize_rf = normalize_rf_hook
    base_cli.normalize_radar = normalize_radar_hook
    base_cli.rf_measurements_to_enu = rf_measurements_hook
    base_cli.radar_measurements_to_enu = radar_measurements_hook
    base_cli.run_async_cv_baseline_with_radar_association = association_hook
    base_cli._baseline_metrics = metrics_hook

    aerpaw.normalize_rf = normalize_rf_hook
    aerpaw.normalize_radar = normalize_radar_hook
    aerpaw.rf_measurements_to_enu = rf_measurements_hook
    aerpaw.radar_measurements_to_enu = radar_measurements_hook

    radar_association.run_async_cv_baseline_with_radar_association = association_hook
    radar_association._nis_scored_candidates = nis_scored_candidates_with_row_covariance
    try:
        yield model
    finally:
        base_cli.normalize_rf = original_base_normalize_rf
        base_cli.normalize_radar = original_base_normalize_radar
        base_cli.rf_measurements_to_enu = original_base_rf_measurements_to_enu
        base_cli.radar_measurements_to_enu = original_base_radar_measurements_to_enu
        base_cli.run_async_cv_baseline_with_radar_association = original_base_run_association
        base_cli._baseline_metrics = original_base_metrics

        aerpaw.normalize_rf = original_aerpaw_normalize_rf
        aerpaw.normalize_radar = original_aerpaw_normalize_radar
        aerpaw.rf_measurements_to_enu = original_aerpaw_rf_measurements_to_enu
        aerpaw.radar_measurements_to_enu = original_aerpaw_radar_measurements_to_enu

        radar_association.run_async_cv_baseline_with_radar_association = original_assoc_run
        radar_association._nis_scored_candidates = original_assoc_scoring


def _apply_model(
    model: HeteroscedasticUncertaintyModel,
    frame: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    if frame.empty or "uncertainty_model" in frame.columns:
        return frame
    try:
        return model.apply(frame, source=source)
    except ValueError as exc:
        if f"no heads for source {source!r}" in str(exc):
            return frame
        raise


def rf_measurements_to_enu_with_row_covariance(
    rf: pd.DataFrame,
    *,
    default_std_m: float = 75.0,
) -> list[TrackingMeasurement]:
    measurements: list[TrackingMeasurement] = []
    for _, row in rf.iterrows():
        std_m = _positive_float(row.get("std_m")) or float(default_std_m)
        fallback = np.diag([std_m**2, std_m**2])
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=np.array([float(row["east_m"]), float(row["north_m"])]),
                covariance=covariance_from_row(row, 2, fallback),
                source="rf",
            )
        )
    return measurements


def radar_measurements_to_enu_with_row_covariance(
    radar: pd.DataFrame,
    *,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
    default_velocity_std_mps: float = 12.0,
    include_velocity: bool = False,
) -> list[TrackingMeasurement]:
    fallback_position_covariance = np.diag(
        [float(default_xy_std_m) ** 2, float(default_xy_std_m) ** 2, float(default_z_std_m) ** 2]
    )
    measurements: list[TrackingMeasurement] = []
    for _, row in radar.iterrows():
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
        position_covariance = covariance_from_row(row, 3, fallback_position_covariance)
        velocity = aerpaw._radar_velocity_vector_enu(row) if include_velocity else None
        if velocity is None:
            vector = position
            covariance = position_covariance
        else:
            vector = np.concatenate([position, velocity])
            covariance = np.zeros((6, 6), dtype=float)
            covariance[:3, :3] = position_covariance
            covariance[3:, 3:] = np.diag([float(default_velocity_std_mps) ** 2] * 3)
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=vector,
                covariance=covariance,
                source="radar",
            )
        )
    return measurements


def promote_covariance_columns_for_association(radar: pd.DataFrame) -> pd.DataFrame:
    out = radar.copy()
    for suffix in _COVARIANCE_SUFFIXES_3D:
        source = f"cov_{suffix}"
        target = f"association_cov_{suffix}"
        if source in out.columns and target not in out.columns:
            out[target] = out[source]
    if "uncertainty_model" in out.columns and "association_covariance_mode" not in out.columns:
        out["association_covariance_mode"] = out["uncertainty_model"]
    return out


def nis_scored_candidates_with_row_covariance(
    candidates: pd.DataFrame,
    tracker: object,
    covariance: np.ndarray,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.iloc[0:0].copy()
    observation = measurement_matrix(3)
    state = np.asarray(tracker.state, dtype=float).reshape(6)
    state_covariance = np.asarray(tracker.covariance_matrix, dtype=float).reshape(6, 6)
    state_position = observation @ state
    predicted_covariance = observation @ state_covariance @ observation.T

    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    nises: list[float] = []
    for (_, row), vector in zip(candidates.iterrows(), vectors):
        measurement_covariance = covariance_from_row(row, 3, np.asarray(covariance, dtype=float))
        innovation_covariance = predicted_covariance + measurement_covariance
        residual = vector - state_position
        try:
            precision = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(innovation_covariance)
        nises.append(float(residual.T @ precision @ residual))

    scored = candidates.copy()
    scored["association_nis"] = np.asarray(nises, dtype=float)
    scored["association_candidate_rows"] = int(len(candidates))
    return scored


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


if __name__ == "__main__":
    raise SystemExit(main())
