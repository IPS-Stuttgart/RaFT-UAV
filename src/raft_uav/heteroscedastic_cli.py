"""Run the legacy baseline with learned heteroscedastic RF/radar covariance.

This entry point keeps the existing ``raft_uav.cli`` implementation as the
single source for loading, association, filtering, smoothing, diagnostics, and
metrics.  It only installs narrow runtime hooks that apply a trained
``HeteroscedasticUncertaintyModel`` to normalized RF/radar rows and make the
legacy measurement/association code consume the resulting row-wise covariance
columns.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement, measurement_matrix
import raft_uav.baselines.radar_association as radar_association
import raft_uav.cli as legacy_cli
import raft_uav.io.aerpaw as aerpaw
from raft_uav.uncertainty import HeteroscedasticUncertaintyModel, covariance_from_row
from raft_uav.uncertainty import load_uncertainty_model

_COVARIANCE_SUFFIXES_2D = ("ee", "nn", "en")
_COVARIANCE_SUFFIXES_3D = ("ee", "nn", "uu", "en", "eu", "nu")


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``raft_uav.cli`` with ``--uncertainty-model`` covariance hooks.

    Example
    -------
    ``raft-uav-heteroscedastic run-baseline DATA --flight F --uncertainty-model model.json``
    """

    uncertainty_model_path, delegated_argv = _extract_uncertainty_model_arg(argv)
    if not delegated_argv or delegated_argv[0] != "run-baseline":
        raise SystemExit(
            "raft-uav-heteroscedastic currently wraps the legacy 'run-baseline' "
            "subcommand; pass e.g. 'run-baseline DATA --flight FLIGHT "
            "--uncertainty-model MODEL.json'."
        )
    with heteroscedastic_covariance_hooks(uncertainty_model_path):
        return legacy_cli.main(list(delegated_argv))


def _extract_uncertainty_model_arg(
    argv: Sequence[str] | None,
) -> tuple[Path, list[str]]:
    args = list(argv or [])
    for index, arg in enumerate(args):
        if arg == "--uncertainty-model":
            if index + 1 >= len(args):
                raise SystemExit("--uncertainty-model requires a path")
            path = Path(args[index + 1])
            return path, [*args[:index], *args[index + 2 :]]
        if arg.startswith("--uncertainty-model="):
            path = Path(arg.split("=", 1)[1])
            return path, [*args[:index], *args[index + 1 :]]
    raise SystemExit("missing required --uncertainty-model MODEL.json")


@contextmanager
def heteroscedastic_covariance_hooks(
    uncertainty_model_path: Path,
) -> Iterator[HeteroscedasticUncertaintyModel]:
    """Temporarily make the legacy baseline consume learned covariance columns."""

    model = load_uncertainty_model(uncertainty_model_path)

    original_legacy_normalize_rf = legacy_cli.normalize_rf
    original_legacy_normalize_radar = legacy_cli.normalize_radar
    original_legacy_rf_measurements_to_enu = legacy_cli.rf_measurements_to_enu
    original_legacy_radar_measurements_to_enu = legacy_cli.radar_measurements_to_enu
    original_legacy_run_association = legacy_cli.run_async_cv_baseline_with_radar_association
    original_legacy_baseline_metrics = legacy_cli._baseline_metrics

    original_aerpaw_normalize_rf = aerpaw.normalize_rf
    original_aerpaw_normalize_radar = aerpaw.normalize_radar
    original_aerpaw_rf_measurements_to_enu = aerpaw.rf_measurements_to_enu
    original_aerpaw_radar_measurements_to_enu = aerpaw.radar_measurements_to_enu

    original_assoc_run_association = radar_association.run_async_cv_baseline_with_radar_association
    original_assoc_nis_scored_candidates = radar_association._nis_scored_candidates

    def normalize_rf_hook(*args: object, **kwargs: object) -> pd.DataFrame:
        frame = original_legacy_normalize_rf(*args, **kwargs)
        return _apply_model_if_available(model, frame, source="rf")

    def normalize_radar_hook(*args: object, **kwargs: object) -> pd.DataFrame:
        frame = original_legacy_normalize_radar(*args, **kwargs)
        return _apply_model_if_available(model, frame, source="radar")

    def rf_measurements_to_enu_hook(*args: object, **kwargs: object) -> list[TrackingMeasurement]:
        try:
            frame = args[0]
        except IndexError:
            return original_legacy_rf_measurements_to_enu(*args, **kwargs)
        if not isinstance(frame, pd.DataFrame):
            return original_legacy_rf_measurements_to_enu(*args, **kwargs)
        default_std_m = float(kwargs.get("default_std_m", 75.0))
        if "east_m" not in frame.columns:
            return original_legacy_rf_measurements_to_enu(*args, **kwargs)
        frame = _apply_model_if_available(model, frame, source="rf")
        return rf_measurements_to_enu_with_row_covariance(frame, default_std_m=default_std_m)

    def radar_measurements_to_enu_hook(*args: object, **kwargs: object) -> list[TrackingMeasurement]:
        try:
            frame = args[0]
        except IndexError:
            return original_legacy_radar_measurements_to_enu(*args, **kwargs)
        if not isinstance(frame, pd.DataFrame):
            return original_legacy_radar_measurements_to_enu(*args, **kwargs)
        default_xy_std_m = float(kwargs.get("default_xy_std_m", 25.0))
        default_z_std_m = float(kwargs.get("default_z_std_m", 35.0))
        default_velocity_std_mps = float(kwargs.get("default_velocity_std_mps", 12.0))
        include_velocity = bool(kwargs.get("include_velocity", False))
        if "east_m" not in frame.columns:
            return original_legacy_radar_measurements_to_enu(*args, **kwargs)
        frame = _apply_model_if_available(model, frame, source="radar")
        return radar_measurements_to_enu_with_row_covariance(
            frame,
            default_xy_std_m=default_xy_std_m,
            default_z_std_m=default_z_std_m,
            default_velocity_std_mps=default_velocity_std_mps,
            include_velocity=include_velocity,
        )

    def run_association_hook(*args: object, **kwargs: object) -> tuple[list[dict[str, object]], pd.DataFrame]:
        if "radar" in kwargs and isinstance(kwargs["radar"], pd.DataFrame):
            kwargs = dict(kwargs)
            kwargs["radar"] = promote_covariance_columns_for_association(kwargs["radar"])
        return original_assoc_run_association(*args, **kwargs)

    def baseline_metrics_hook(*args: object, **kwargs: object) -> dict[str, object]:
        metrics = original_legacy_baseline_metrics(*args, **kwargs)
        metrics["uncertainty_model"] = {
            "path": str(uncertainty_model_path),
            "type": "heteroscedastic-loglinear-variance",
        }
        metrics["rf_covariance"] = "heteroscedastic learned cov_* columns; fallback CEP/default"
        metrics["radar_covariance"] = "heteroscedastic learned cov_* columns; fallback fixed radar"
        return metrics

    legacy_cli.normalize_rf = normalize_rf_hook
    legacy_cli.normalize_radar = normalize_radar_hook
    legacy_cli.rf_measurements_to_enu = rf_measurements_to_enu_hook
    legacy_cli.radar_measurements_to_enu = radar_measurements_to_enu_hook
    legacy_cli.run_async_cv_baseline_with_radar_association = run_association_hook
    legacy_cli._baseline_metrics = baseline_metrics_hook

    aerpaw.normalize_rf = normalize_rf_hook
    aerpaw.normalize_radar = normalize_radar_hook
    aerpaw.rf_measurements_to_enu = rf_measurements_to_enu_hook
    aerpaw.radar_measurements_to_enu = radar_measurements_to_enu_hook

    radar_association.run_async_cv_baseline_with_radar_association = run_association_hook
    radar_association._nis_scored_candidates = nis_scored_candidates_with_row_covariance
    try:
        yield model
    finally:
        legacy_cli.normalize_rf = original_legacy_normalize_rf
        legacy_cli.normalize_radar = original_legacy_normalize_radar
        legacy_cli.rf_measurements_to_enu = original_legacy_rf_measurements_to_enu
        legacy_cli.radar_measurements_to_enu = original_legacy_radar_measurements_to_enu
        legacy_cli.run_async_cv_baseline_with_radar_association = original_legacy_run_association
        legacy_cli._baseline_metrics = original_legacy_baseline_metrics

        aerpaw.normalize_rf = original_aerpaw_normalize_rf
        aerpaw.normalize_radar = original_aerpaw_normalize_radar
        aerpaw.rf_measurements_to_enu = original_aerpaw_rf_measurements_to_enu
        aerpaw.radar_measurements_to_enu = original_aerpaw_radar_measurements_to_enu

        radar_association.run_async_cv_baseline_with_radar_association = original_assoc_run_association
        radar_association._nis_scored_candidates = original_assoc_nis_scored_candidates


def _apply_model_if_available(
    model: HeteroscedasticUncertaintyModel,
    frame: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "uncertainty_model" in frame.columns:
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
    """Convert normalized RF rows, preferring learned row-wise covariance columns."""

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
    """Convert normalized radar rows, preferring learned row-wise covariance columns."""

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
    """Expose learned cov_* columns under association_cov_* for legacy updates."""

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
    """Score radar candidates with per-row covariance when available."""

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
    import sys

    raise SystemExit(main(sys.argv[1:]))
