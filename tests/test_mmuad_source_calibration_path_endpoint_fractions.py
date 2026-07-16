from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import fit_source_calibration
from raft_uav.mmuad.source_calibration_path_ensemble import (
    CALIBRATION_FRACTION_COLUMN,
    EFFECTIVE_ALPHA_COLUMN,
    INTERPOLATED_COLUMN,
    build_source_calibration_path_ensemble,
)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 2.0, 4.0],
            "y_m": [10.0, 11.0, 12.0],
            "z_m": [3.0, 3.5, 4.0],
        }
    )


def _candidate_rows() -> pd.DataFrame:
    truth = _truth_rows()
    return pd.DataFrame(
        {
            "sequence_id": truth["sequence_id"],
            "time_s": truth["time_s"],
            "source": ["lidar_360"] * 3,
            "track_id": ["a", "b", "c"],
            "x_m": truth["x_m"] + 10.0,
            "y_m": truth["y_m"] - 4.0,
            "z_m": truth["z_m"] + 2.0,
            "confidence": [0.8, 0.8, 0.8],
        }
    )


def _calibration_payload() -> dict:
    payload, _pairs, _summary = fit_source_calibration(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        mode="source-translation",
        max_truth_time_delta_s=0.1,
        max_pair_distance_m=50.0,
        min_pairs_per_source=2,
    )
    return payload


def test_path_ensemble_preserves_near_endpoint_fractions() -> None:
    fractions = (1.0e-9, 1.0 - 1.0e-6)

    ensemble = build_source_calibration_path_ensemble(
        CandidateFrame(_candidate_rows().iloc[[0]]),
        _calibration_payload(),
        fractions=fractions,
    ).rows.sort_values(CALIBRATION_FRACTION_COLUMN)

    assert ensemble[CALIBRATION_FRACTION_COLUMN].tolist() == pytest.approx(fractions)
    assert ensemble[INTERPOLATED_COLUMN].tolist() == [True, True]
    assert ensemble["x_m"].tolist() == pytest.approx(
        [10.0 * (1.0 - fractions[0]), 10.0 * (1.0 - fractions[1])]
    )
    assert (ensemble[EFFECTIVE_ALPHA_COLUMN] > 0.0).all()
    assert ensemble["candidate_branch"].nunique() == 2
