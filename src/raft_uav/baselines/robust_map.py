"""RaFT-UAV adapter for PyRecEst robust linear-Gaussian MAP smoothing.

RaFT-UAV keeps its constant-velocity process model, tracking-record schema, and
RF/radar measurement-to-record matching here. The generic sparse MAP solve,
robust factor weighting, and fixed-lag windowing live in PyRecEst.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from pyrecest.smoothers import (
    LinearGaussianMeasurementFactor as _PyRecEstMeasurementFactor,
)
from pyrecest.smoothers import (
    RobustLinearGaussianMapConfig as _PyRecEstRobustMapConfig,
)
from pyrecest.smoothers import (
    fixed_lag_robust_linear_gaussian_map_smooth as _pyrecest_fixed_lag_robust_map,
)
from pyrecest.smoothers import (
    robust_linear_gaussian_map_smooth as _pyrecest_robust_map,
)
from pyrecest.smoothers.robust_linear_gaussian_map import (
    ROBUST_LINEAR_GAUSSIAN_MAP_LOSSES,
)
from scipy.optimize import linear_sum_assignment

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    constant_velocity_matrix,
    measurement_matrix,
    white_acceleration_process_noise,
)
from raft_uav.baselines.record_helpers import copy_record, record_arrays, symmetrized
from raft_uav.numeric import optional_float, optional_int

ROBUST_MAP_LOSSES = ROBUST_LINEAR_GAUSSIAN_MAP_LOSSES


@dataclass(frozen=True)
class RobustMapSmootherConfig:
    """RaFT-UAV controls for PyRecEst's robust MAP smoother."""

    loss: str = "huber"
    loss_scale: float = 3.0
    max_iterations: int = 50
    relative_tolerance: float = 1.0e-5
    measurement_time_tolerance_s: float = 1.0e-6
    process_position_floor_m: float = 0.25
    process_velocity_floor_mps: float = 0.25
    accepted_measurements_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.loss, str) or self.loss not in ROBUST_MAP_LOSSES:
            raise ValueError(f"loss must be one of {ROBUST_MAP_LOSSES}")
        object.__setattr__(
            self,
            "loss_scale",
            _finite_float(self.loss_scale, name="loss_scale", positive=True),
        )
        object.__setattr__(
            self,
            "max_iterations",
            _positive_integer(self.max_iterations, name="max_iterations"),
        )
        object.__setattr__(
            self,
            "relative_tolerance",
            _finite_float(
                self.relative_tolerance,
                name="relative_tolerance",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "measurement_time_tolerance_s",
            _finite_float(
                self.measurement_time_tolerance_s,
                name="measurement_time_tolerance_s",
            ),
        )
        object.__setattr__(
            self,
            "process_position_floor_m",
            _finite_float(
                self.process_position_floor_m,
                name="process_position_floor_m",
            ),
        )
        object.__setattr__(
            self,
            "process_velocity_floor_mps",
            _finite_float(
                self.process_velocity_floor_mps,
                name="process_velocity_floor_mps",
            ),
        )
        if not isinstance(self.accepted_measurements_only, (bool, np.bool_)):
            raise ValueError("accepted_measurements_only must be a Boolean scalar")
        object.__setattr__(
            self,
            "accepted_measurements_only",
            bool(self.accepted_measurements_only),
        )


@dataclass(frozen=True)
class RobustMapResult:
    """Compatibility representation of one RaFT-UAV MAP solve."""

    states: np.ndarray
    covariances: np.ndarray
    matched_measurements: int
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    message: str


@dataclass(frozen=True)
class _MeasurementFactor:
    index: int
    vector: np.ndarray
    covariance: np.ndarray
    source: str


def robust_map_smooth_records(
    records: list[dict[str, object]],
    *,
    measurements: Iterable[TrackingMeasurement] | None,
    acceleration_std_mps2: float,
    config: RobustMapSmootherConfig | None = None,
    lag_s: float | None = None,
) -> list[dict[str, object]]:
    """Smooth RaFT-UAV records through PyRecEst's generic robust MAP solver."""

    cfg = _validated_config(config)
    lag_value = _validated_lag_s(lag_s)
    acceleration_std = _finite_float(
        acceleration_std_mps2,
        name="acceleration_std_mps2",
    )
    if not records:
        return []

    out = [copy_record(record) for record in records]
    times, filtered_states, filtered_covariances = record_arrays(out)
    if np.any(np.diff(times) < 0.0):
        raise ValueError("tracking records must be sorted by nondecreasing time")

    if measurements is None:
        factors = _record_pseudo_measurement_factors(
            out,
            filtered_states,
            filtered_covariances,
            accepted_only=cfg.accepted_measurements_only,
        )
    else:
        factors = _matched_measurement_factors(
            out,
            measurements,
            times,
            time_tolerance_s=cfg.measurement_time_tolerance_s,
            accepted_only=cfg.accepted_measurements_only,
        )
    if not factors:
        raise ValueError("robust-map smoothing matched no measurement factors")

    transitions, process_covariances = _cv_model_sequences(
        times,
        acceleration_std_mps2=acceleration_std,
        position_floor_m=cfg.process_position_floor_m,
        velocity_floor_mps=cfg.process_velocity_floor_mps,
    )
    pyrecest_factors = _pyrecest_measurement_factors(factors)
    pyrecest_config = _PyRecEstRobustMapConfig(
        loss=cfg.loss,
        loss_scale=cfg.loss_scale,
        max_iterations=cfg.max_iterations,
        relative_tolerance=cfg.relative_tolerance,
    )

    if lag_value is None:
        result = _pyrecest_robust_map(
            filtered_states,
            prior_mean=filtered_states[0],
            prior_covariance=filtered_covariances[0],
            transition_matrices=transitions,
            process_covariances=process_covariances,
            measurements=pyrecest_factors,
            config=pyrecest_config,
        )
        for index, record in enumerate(out):
            _write_result_to_record(
                record,
                filtered_states[index],
                filtered_covariances[index],
                result.states[index],
                method="robust-map",
                lag_s=None,
                matched_measurements=result.measurement_factor_count,
                initial_cost=result.initial_cost,
                final_cost=result.final_cost,
                iterations=result.iterations,
                success=result.success,
                message=result.message,
            )
        return out

    result = _pyrecest_fixed_lag_robust_map(
        times,
        filtered_states,
        anchor_covariances=filtered_covariances,
        transition_matrices=transitions,
        process_covariances=process_covariances,
        measurements=pyrecest_factors,
        lag=lag_value,
        config=pyrecest_config,
    )
    for index, (record, window) in enumerate(zip(out, result.windows, strict=True)):
        _write_result_to_record(
            record,
            filtered_states[index],
            filtered_covariances[index],
            result.states[index],
            method="fixed-lag-map",
            lag_s=lag_value,
            matched_measurements=window.measurement_factor_count,
            initial_cost=window.initial_cost,
            final_cost=window.final_cost,
            iterations=window.iterations,
            success=window.success,
            message=window.message,
        )
    return out


def _validated_config(value: object) -> RobustMapSmootherConfig:
    if value is None:
        return RobustMapSmootherConfig()
    if not isinstance(value, RobustMapSmootherConfig):
        raise TypeError("config must be a RobustMapSmootherConfig or None")
    return value


def _validated_lag_s(value: object) -> float | None:
    if value is None:
        return None
    error = "lag_s must be a finite nonnegative real scalar or None"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        lag_s = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(lag_s) or lag_s < 0.0:
        raise ValueError(error)
    return lag_s


def _finite_float(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> float:
    number = optional_float(value)
    qualifier = "positive" if positive else "nonnegative"
    if number is None or (number <= 0.0 if positive else number < 0.0):
        raise ValueError(f"{name} must be a finite {qualifier} real scalar")
    return number


def _positive_integer(value: object, *, name: str) -> int:
    number = optional_int(value)
    if number is None or number < 1:
        raise ValueError(f"{name} must be a positive integer scalar")
    return number


def _record_pseudo_measurement_factors(
    records: list[dict[str, object]],
    states: np.ndarray,
    covariances: np.ndarray,
    *,
    accepted_only: bool,
) -> list[_MeasurementFactor]:
    factors: list[_MeasurementFactor] = []
    for index, record in enumerate(records):
        if accepted_only and not bool(record.get("accepted", True)):
            continue
        dimension = int(record.get("measurement_dim", 3) or 3)
        if dimension not in (2, 3, 6):
            dimension = 3
        observation = measurement_matrix(dimension)
        covariance = symmetrized(observation @ covariances[index] @ observation.T)
        covariance = covariance + np.eye(dimension) * 1.0e-6
        factors.append(
            _MeasurementFactor(
                index=index,
                vector=observation @ states[index],
                covariance=covariance,
                source=str(record.get("source", "posterior")),
            )
        )
    return factors


def _matched_measurement_factors(
    records: list[dict[str, object]],
    measurements: Iterable[TrackingMeasurement] | None,
    times: np.ndarray,
    *,
    time_tolerance_s: float,
    accepted_only: bool,
) -> list[_MeasurementFactor]:
    """Globally match measurements to records with cardinality-first assignment."""

    if measurements is None:
        return []
    ordered = sorted(
        measurements,
        key=lambda item: (float(item.time_s), str(item.source), int(item.vector.size)),
    )
    if not ordered or not records:
        return []

    record_times = np.asarray(times, dtype=float).reshape(-1)
    if record_times.size != len(records):
        raise ValueError("record times must align with tracking records")

    record_eligible = np.isfinite(record_times)
    if accepted_only:
        record_eligible &= np.asarray(
            [bool(record.get("accepted", True)) for record in records],
            dtype=bool,
        )
    if not record_eligible.any():
        return []

    measurement_times = np.asarray(
        [float(measurement.time_s) for measurement in ordered],
        dtype=float,
    )
    time_deltas = np.abs(measurement_times[:, None] - record_times[None, :])
    eligible = (
        record_eligible[None, :]
        & np.isfinite(time_deltas)
        & (time_deltas <= float(time_tolerance_s))
    )
    if not eligible.any():
        return []

    measurement_sources = np.asarray(
        [str(measurement.source) for measurement in ordered],
        dtype=object,
    )
    record_sources = np.asarray(
        [str(record.get("source")) for record in records],
        dtype=object,
    )
    same_source = measurement_sources[:, None] == record_sources[None, :]

    measurement_count, record_count = eligible.shape
    max_matches = min(measurement_count, record_count)
    source_penalty = float(max_matches + 1)
    unmatched_penalty = float((max_matches + 1) ** 2)
    normalized_time_error = np.zeros_like(time_deltas)
    if time_tolerance_s > 0.0:
        normalized_time_error = np.minimum(
            time_deltas / float(time_tolerance_s),
            1.0,
        )

    costs = np.full(
        (measurement_count, record_count + measurement_count),
        unmatched_penalty,
        dtype=float,
    )
    costs[:, :record_count] = np.where(
        eligible,
        np.where(same_source, 0.0, source_penalty) + normalized_time_error,
        2.0 * unmatched_penalty,
    )
    costs[:, :record_count] += np.where(
        eligible,
        np.arange(record_count, dtype=float)[None, :] * np.finfo(float).eps,
        0.0,
    )

    measurement_indices, assignment_columns = linear_sum_assignment(costs)
    assignments = sorted(
        (int(measurement_index), int(record_index))
        for measurement_index, record_index in zip(
            measurement_indices,
            assignment_columns,
            strict=True,
        )
        if record_index < record_count and eligible[measurement_index, record_index]
    )
    return [
        _MeasurementFactor(
            index=record_index,
            vector=np.asarray(
                ordered[measurement_index].vector,
                dtype=float,
            ).reshape(-1),
            covariance=np.asarray(
                ordered[measurement_index].covariance,
                dtype=float,
            ),
            source=ordered[measurement_index].source,
        )
        for measurement_index, record_index in assignments
    ]


def _pyrecest_measurement_factors(
    factors: list[_MeasurementFactor],
) -> tuple[_PyRecEstMeasurementFactor, ...]:
    return tuple(
        _PyRecEstMeasurementFactor(
            state_index=factor.index,
            measurement=factor.vector,
            observation_matrix=measurement_matrix(factor.vector.size),
            covariance=factor.covariance,
            metadata={"source": factor.source},
        )
        for factor in factors
    )


def _cv_model_sequences(
    times: np.ndarray,
    *,
    acceleration_std_mps2: float,
    position_floor_m: float,
    velocity_floor_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    state_count = len(times)
    if state_count <= 1:
        empty = np.empty((0, 6, 6), dtype=float)
        return empty, empty.copy()
    transitions: list[np.ndarray] = []
    process_covariances: list[np.ndarray] = []
    floor = np.diag(
        [
            position_floor_m**2,
            position_floor_m**2,
            position_floor_m**2,
            velocity_floor_mps**2,
            velocity_floor_mps**2,
            velocity_floor_mps**2,
        ]
    )
    for previous, current in zip(times[:-1], times[1:], strict=True):
        dt_s = max(0.0, float(current - previous))
        transitions.append(constant_velocity_matrix(dt_s))
        covariance = white_acceleration_process_noise(
            dt_s,
            acceleration_std_mps2,
        )
        process_covariances.append(symmetrized(covariance + floor))
    return np.stack(transitions), np.stack(process_covariances)


def _write_result_to_record(
    record: dict[str, object],
    filtered_state: np.ndarray,
    filtered_covariance: np.ndarray,
    state: np.ndarray,
    *,
    method: str,
    lag_s: float | None,
    matched_measurements: int,
    initial_cost: float,
    final_cost: float,
    iterations: int,
    success: bool,
    message: str,
) -> None:
    record["filtered_state"] = filtered_state.copy()
    record["filtered_covariance"] = filtered_covariance.copy()
    record["state"] = np.asarray(state, dtype=float).copy()
    # PyRecEst deliberately does not yet expose MAP marginals. Preserve the
    # filtered covariance and label it accurately rather than presenting it as a
    # smoother covariance.
    record["covariance"] = filtered_covariance.copy()
    record["smoother_method"] = method
    record["smoother_lag_s"] = lag_s
    record["map_covariance_source"] = "filtered"
    record["map_solver"] = "pyrecest.robust_linear_gaussian_map"
    record["map_success"] = bool(success)
    record["map_iterations"] = int(iterations)
    record["map_initial_cost"] = float(initial_cost)
    record["map_final_cost"] = float(final_cost)
    record["map_matched_measurements"] = int(matched_measurements)
    record["map_message"] = str(message)
