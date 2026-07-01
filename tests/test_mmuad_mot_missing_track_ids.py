from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.mot import (
    MultiObjectTrackerConfig,
    compute_multi_object_metrics,
    run_mmuad_multi_object_tracker,
)
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame


def test_multi_object_metrics_falls_back_for_missing_like_truth_track_ids() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq0"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [5.0, 5.0],
            "track_id": [None, "   "],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq0"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [5.0, 5.0],
            "output_track_id": ["mot_1", "mot_1"],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth)

    assert metrics["count"] == 2
    assert metrics["mean_3d_m"] == pytest.approx(0.0)
    assert "gt_count" not in metrics


def test_multi_object_tracker_attaches_errors_for_blank_truth_track_id_column() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq0"],
                "time_s": [0.0],
                "source": ["detector"],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [5.0],
                "confidence": [1.0],
            }
        )
    )
    truth = TruthFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq0"],
                "time_s": [0.0],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [5.0],
                "track_id": [""],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(
        candidates,
        truth,
        config=MultiObjectTrackerConfig(max_association_distance_m=5.0),
    )

    assert "error_3d_m" in output.estimates.columns
    assert output.metrics["pooled"]["mean_3d_m"] == pytest.approx(0.0)
