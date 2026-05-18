"""Causal RF bootstrap guards for truth-free radar association paths.

Truth-free online association should not initialize a single-target tracker from
an arbitrary pre-RF Fortem radar candidate chosen only by class probability.  If
RF measurements are available, these wrappers drop radar frames before the first
RF timestamp before delegating to the existing association implementations.  This
keeps radar-only diagnostic inputs supported while making RF+radar runs start
from the first RF observation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, TypeVar

import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement

_F = TypeVar("_F", bound=Callable[..., Any])


def _materialize_rf_measurements(
    rf_measurements: Iterable[TrackingMeasurement] | None,
) -> list[TrackingMeasurement] | None:
    if rf_measurements is None:
        return None
    return list(rf_measurements)


def _first_rf_time_s(rf_measurements: list[TrackingMeasurement] | None) -> float | None:
    if not rf_measurements:
        return None
    return min(float(measurement.time_s) for measurement in rf_measurements)


def _radar_from_first_rf(
    radar: pd.DataFrame | None,
    rf_measurements: list[TrackingMeasurement] | None,
) -> pd.DataFrame | None:
    """Return radar rows whose timestamps are not before the first RF update."""

    first_rf_time_s = _first_rf_time_s(rf_measurements)
    if radar is None or first_rf_time_s is None or radar.empty or "time_s" not in radar.columns:
        return radar
    time_s = pd.to_numeric(radar["time_s"], errors="coerce")
    return radar.loc[time_s >= first_rf_time_s].copy()


def _wrap_truth_free_rf_bootstrap(
    function: _F,
    *,
    skip_oracle_association: bool = False,
) -> _F:
    """Wrap a keyword-only association runner with first-RF radar trimming."""

    if getattr(function, "_causal_rf_bootstrap_wrapped", False):
        return function

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if skip_oracle_association and kwargs.get("association") == "oracle-nearest-truth":
            return function(*args, **kwargs)
        rf_measurements = _materialize_rf_measurements(kwargs.get("rf_measurements"))
        if rf_measurements is None:
            return function(*args, **kwargs)
        updated_kwargs = dict(kwargs)
        updated_kwargs["rf_measurements"] = rf_measurements
        updated_kwargs["radar"] = _radar_from_first_rf(kwargs.get("radar"), rf_measurements)
        return function(*args, **updated_kwargs)

    setattr(wrapped, "_causal_rf_bootstrap_wrapped", True)
    return wrapped  # type: ignore[return-value]


def apply_causal_bootstrap_patches() -> None:
    """Install first-RF bootstrap guards on all truth-free public entry points."""

    from raft_uav.baselines import radar_association
    from raft_uav.baselines import tracklet_viterbi_range_covariance
    from raft_uav.baselines import tracklet_viterbi_result
    from raft_uav.baselines import tracklet_viterbi_retention

    radar_association.run_async_cv_baseline_with_radar_association = _wrap_truth_free_rf_bootstrap(
        radar_association.run_async_cv_baseline_with_radar_association,
        skip_oracle_association=True,
    )
    tracklet_viterbi_retention.run_async_cv_baseline_with_tracklet_viterbi_association = (
        _wrap_truth_free_rf_bootstrap(
            tracklet_viterbi_retention.run_async_cv_baseline_with_tracklet_viterbi_association,
        )
    )
    tracklet_viterbi_range_covariance.run_async_cv_baseline_with_tracklet_viterbi_association = (
        _wrap_truth_free_rf_bootstrap(
            tracklet_viterbi_range_covariance.run_async_cv_baseline_with_tracklet_viterbi_association,
        )
    )
    tracklet_viterbi_result.run_async_cv_baseline_with_tracklet_viterbi_result = (
        _wrap_truth_free_rf_bootstrap(
            tracklet_viterbi_result.run_async_cv_baseline_with_tracklet_viterbi_result,
        )
    )
