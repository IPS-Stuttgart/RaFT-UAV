"""Compatibility fixes for robust-MAP measurement matching."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from raft_uav.baselines import robust_map as _robust_map
from raft_uav.baselines.kalman import TrackingMeasurement

_ORIGINAL_RECORD_PSEUDO_MEASUREMENT_FACTORS = (
    _robust_map._record_pseudo_measurement_factors
)


def _record_accepted(record: dict[str, object], *, index: int) -> bool:
    """Return a strict Boolean acceptance flag for one tracking record."""

    value = record.get("accepted", True)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"records[{index}].accepted must be a Boolean scalar")
    return bool(value)


def _record_pseudo_measurement_factors(
    records: list[dict[str, object]],
    states: np.ndarray,
    covariances: np.ndarray,
    *,
    accepted_only: bool,
) -> list[_robust_map._MeasurementFactor]:
    """Build posterior factors without applying truthiness to acceptance flags."""

    if not accepted_only:
        return _ORIGINAL_RECORD_PSEUDO_MEASUREMENT_FACTORS(
            records,
            states,
            covariances,
            accepted_only=False,
        )

    normalized_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        normalized = dict(record)
        normalized["accepted"] = _record_accepted(record, index=index)
        normalized_records.append(normalized)
    return _ORIGINAL_RECORD_PSEUDO_MEASUREMENT_FACTORS(
        normalized_records,
        states,
        covariances,
        accepted_only=True,
    )


def _matched_measurement_factors(
    records: list[dict[str, object]],
    measurements: Iterable[TrackingMeasurement] | None,
    times: np.ndarray,
    *,
    time_tolerance_s: float,
    accepted_only: bool,
) -> list[_robust_map._MeasurementFactor]:
    """Match measurements globally without losing feasible one-to-one factors.

    The assignment first maximizes the number of matched measurements, then
    maximizes source-consistent matches, and finally minimizes total timestamp
    error. Rejected records are removed before assignment in accepted-only mode.
    """

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
            [
                _record_accepted(record, index=index)
                for index, record in enumerate(records)
            ],
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
    # Stabilize exact timestamp/source ties in favor of the earlier record.
    costs[:, :record_count] += np.where(
        eligible,
        np.arange(record_count, dtype=float)[None, :] * np.finfo(float).eps,
        0.0,
    )

    measurement_indices, assignment_columns = linear_sum_assignment(costs)
    assignments = sorted(
        (
            int(measurement_index),
            int(record_index),
        )
        for measurement_index, record_index in zip(
            measurement_indices,
            assignment_columns,
            strict=True,
        )
        if record_index < record_count and eligible[measurement_index, record_index]
    )

    return [
        _robust_map._MeasurementFactor(
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


def apply_robust_map_accepted_matching_patch() -> None:
    """Install the corrected matchers in the legacy robust-MAP implementation."""

    _robust_map._record_pseudo_measurement_factors = _record_pseudo_measurement_factors
    _robust_map._matched_measurement_factors = _matched_measurement_factors
