from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import (
    build_source_calibration_pairs,
    fit_source_transforms,
)


def test_source_calibration_pairs_are_scoped_by_physical_flight() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["shared", "shared"],
                "flight_id": ["flight-a", "flight-b"],
                "time_s": [0.0, 0.0],
                "source": ["lidar", "lidar"],
                "track_id": ["a", "b"],
                "x_m": [0.0, 100.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 0.0],
                "confidence": [1.0, 1.0],
            }
        )
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "x_m": [1.0, 101.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    pairs = build_source_calibration_pairs(
        candidates,
        truth,
        max_truth_time_delta_s=0.01,
        max_pair_distance_m=10.0,
    )

    assert len(pairs) == 2
    by_flight = pairs.set_index("flight_id")
    assert by_flight.loc["flight-a", "truth_x_m"] == pytest.approx(1.0)
    assert by_flight.loc["flight-b", "truth_x_m"] == pytest.approx(101.0)
    assert by_flight.loc["flight-a", "pair_error_before_m"] == pytest.approx(1.0)
    assert by_flight.loc["flight-b", "pair_error_before_m"] == pytest.approx(1.0)


def test_source_calibration_rejects_one_sided_ambiguous_flight_metadata() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["shared", "shared"],
                "flight_id": ["flight-a", "flight-b"],
                "time_s": [0.0, 0.0],
                "source": ["lidar", "lidar"],
                "track_id": ["a", "b"],
                "x_m": [0.0, 100.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 0.0],
                "confidence": [1.0, 1.0],
            }
        )
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="both provide flight_id"):
        build_source_calibration_pairs(
            candidates,
            truth,
            max_truth_time_delta_s=0.01,
            max_pair_distance_m=10.0,
        )


def test_source_translation_alpha_cv_holds_out_physical_flights() -> None:
    pairs = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared", "shared", "shared"],
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "source": ["lidar", "lidar", "lidar", "lidar"],
            "x_m": [0.0, 0.0, 0.0, 0.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "truth_x_m": [1.0, 1.0, 3.0, 3.0],
            "truth_y_m": [0.0, 0.0, 0.0, 0.0],
            "truth_z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )

    transforms, summary = fit_source_transforms(
        pairs,
        mode="source-translation",
        min_pairs_per_source=1,
        source_translation_alpha_grid=[0.0, 1.0],
    )

    assert summary.loc[0, "source_translation_alpha_cv_fold_count"] == 2
    assert transforms["lidar"].metadata["source_translation_alpha_cv_scope"] == (
        "sequence_id+flight_id"
    )
