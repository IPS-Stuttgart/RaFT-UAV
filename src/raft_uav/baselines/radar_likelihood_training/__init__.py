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
from raft_uav.numeric import optional_float

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
    """Collect training rows after validating both truth-matching gates."""

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
        positive_gate_m=positive_gate_m,
        truth_gate_m=distance_gate,
        truth_time_gate_s=time_gate,
        teacher_association=teacher_association,
        track_switch_nis_ratio=track_switch_nis_ratio,
    )


_IMPL.collect_radar_association_training_frame = collect_radar_association_training_frame

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_nonnegative_gate"] = _validated_nonnegative_gate
globals()["collect_radar_association_training_frame"] = (
    collect_radar_association_training_frame
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
