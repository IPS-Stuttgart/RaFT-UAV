from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement, TrackingUpdateDiagnostics
from raft_uav.diagnostics.paper_strict import _tracking_record


def _record(selected_row: pd.Series) -> dict[str, object]:
    measurement = TrackingMeasurement(
        time_s=1.0,
        vector=np.zeros(3),
        covariance=np.eye(3),
        source="radar",
        _apply_runtime_calibration=False,
    )
    tracker = SimpleNamespace(
        state=np.zeros(6),
        covariance_matrix=np.eye(6),
    )
    diagnostics = TrackingUpdateDiagnostics(
        time_s=1.0,
        source="radar",
        measurement_dim=3,
        accepted=True,
        update_action="updated",
        nis=0.0,
        gate_threshold=7.815,
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=0.0,
    )
    return _tracking_record(
        measurement,
        tracker,
        diagnostics,
        association_mode="paper-strict",
        selected_row=selected_row,
    )


def test_paper_strict_record_preserves_large_integer_identifiers() -> None:
    record = _record(
        pd.Series(
            {
                "track_id": "9007199254740993",
                "frame_index": "9007199254740995",
            }
        )
    )

    assert record["track_id"] == 9007199254740993
    assert record["frame_index"] == 9007199254740995


def test_paper_strict_record_rejects_non_integer_identifier_metadata() -> None:
    record = _record(
        pd.Series(
            {
                "track_id": "7.5",
                "frame_index": np.array([7]),
            }
        )
    )

    assert "track_id" not in record
    assert "frame_index" not in record
