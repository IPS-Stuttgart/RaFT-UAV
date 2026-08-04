"""Compatibility validation for radar-association training truth gates.

The maintained implementation lives in the sibling ``radar_likelihood_training.py``
module. This package preserves the public import path while rejecting malformed
truth-matching gates before they can silently widen or empty the training data.
"""

from __future__ import annotations

from collections.abc import Iterable
import importlib.util
from pathlib import Path
import sys

import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_likelihood_training.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._radar_likelihood_training_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load radar likelihood training from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_COLLECT_RADAR_ASSOCIATION_TRAINING_FRAME = (
    _IMPL.collect_radar_association_training_frame
)


def _validated_nonnegative_gate(value: object, *, name: str) -> float:
    """Return a finite non-negative scalar gate or raise a stable error."""

    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return normalized


def _validated_positive_gate(value: object, *, name: str) -> float:
    """Return a finite positive scalar gate or raise a stable error."""

    normalized = optional_float(value)
    if normalized is None or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return normalized


def _single_best_candidate(frame: pd.DataFrame, *, score_column: str) -> pd.Series:
    """Select one minimum-score row without relying on index-label uniqueness."""

    position = int(frame[score_column].reset_index(drop=True).idxmin())
    return frame.iloc[position].copy()


def _student_selected_candidate(
    scored: pd.DataFrame,
    *,
    teacher_association: str,
    current_track_id: int | None,
    track_switch_nis_ratio: float,
) -> pd.Series | None:
    """Select exactly one student candidate even when row labels are duplicated."""

    if scored.empty:
        return None
    score_column = (
        "association_score"
        if "association_score" in scored.columns
        else "association_nis"
    )
    best = _single_best_candidate(scored, score_column=score_column)
    if (
        teacher_association == "prediction-nis"
        or current_track_id is None
        or "track_id" not in scored
    ):
        return best
    current = scored.loc[scored["track_id"] == current_track_id]
    if current.empty:
        return best
    current_best = _single_best_candidate(current, score_column=score_column)
    best_track_id = optional_int(best.get("track_id"))
    if best_track_id == current_track_id:
        return best
    if float(best["association_nis"]) < float(current_best["association_nis"]) * float(
        track_switch_nis_ratio
    ):
        return best
    return current_best


def collect_radar_association_training_frame(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    flight_name: str | None = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    candidate_catprob_threshold: float | None = 0.5,
    positive_gate_m: float = 50.0,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
    teacher_association: str = "oracle",
    track_switch_nis_ratio: float = 0.5,
) -> pd.DataFrame:
    """Collect training rows after validating all truth-matching gates."""

    positive_gate = _validated_positive_gate(positive_gate_m, name="positive_gate_m")
    distance_gate = _validated_nonnegative_gate(truth_gate_m, name="truth_gate_m")
    time_gate = _validated_nonnegative_gate(
        truth_time_gate_s,
        name="truth_time_gate_s",
    )
    return _ORIGINAL_COLLECT_RADAR_ASSOCIATION_TRAINING_FRAME(
        rf_measurements=rf_measurements,
        radar=radar,
        truth=truth,
        flight_name=flight_name,
        acceleration_std_mps2=acceleration_std_mps2,
        radar_xy_std_m=radar_xy_std_m,
        radar_z_std_m=radar_z_std_m,
        candidate_catprob_threshold=candidate_catprob_threshold,
        positive_gate_m=positive_gate,
        truth_gate_m=distance_gate,
        truth_time_gate_s=time_gate,
        teacher_association=teacher_association,
        track_switch_nis_ratio=track_switch_nis_ratio,
    )


_IMPL._student_selected_candidate = _student_selected_candidate
_IMPL.collect_radar_association_training_frame = collect_radar_association_training_frame

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_nonnegative_gate"] = _validated_nonnegative_gate
globals()["_validated_positive_gate"] = _validated_positive_gate
globals()["_single_best_candidate"] = _single_best_candidate
globals()["_student_selected_candidate"] = _student_selected_candidate
globals()["collect_radar_association_training_frame"] = (
    collect_radar_association_training_frame
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
