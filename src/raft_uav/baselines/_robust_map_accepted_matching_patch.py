"""Compatibility fix for accepted-only robust-MAP measurement matching."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from raft_uav.baselines import robust_map as _robust_map
from raft_uav.baselines.kalman import TrackingMeasurement


def _matched_measurement_factors(
    records: list[dict[str, object]],
    measurements: Iterable[TrackingMeasurement] | None,
    times: np.ndarray,
    *,
    time_tolerance_s: float,
    accepted_only: bool,
) -> list[_robust_map._MeasurementFactor]:
    """Match measurements without allowing rejected records to shadow accepted ones."""

    if measurements is None:
        return []
    used_record_indices: set[int] = set()
    factors: list[_robust_map._MeasurementFactor] = []
    ordered = sorted(
        measurements,
        key=lambda item: (float(item.time_s), str(item.source), int(item.vector.size)),
    )
    for measurement in ordered:
        candidate_indices = _robust_map._candidate_record_indices(
            times,
            float(measurement.time_s),
            tolerance_s=time_tolerance_s,
        )
        candidate_indices = [idx for idx in candidate_indices if idx not in used_record_indices]
        if accepted_only:
            candidate_indices = [
                idx for idx in candidate_indices if bool(records[idx].get("accepted", True))
            ]
        source_matches = [
            idx for idx in candidate_indices if str(records[idx].get("source")) == measurement.source
        ]
        if source_matches:
            candidate_indices = source_matches
        if not candidate_indices:
            continue
        best_index = min(candidate_indices, key=lambda idx: abs(times[idx] - measurement.time_s))
        used_record_indices.add(best_index)
        factors.append(
            _robust_map._MeasurementFactor(
                index=int(best_index),
                vector=np.asarray(measurement.vector, dtype=float).reshape(-1),
                covariance=np.asarray(measurement.covariance, dtype=float),
                source=measurement.source,
            )
        )
    return factors


def apply_robust_map_accepted_matching_patch() -> None:
    """Install the corrected matcher in the legacy robust-MAP implementation."""

    _robust_map._matched_measurement_factors = _matched_measurement_factors
