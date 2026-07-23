from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.evaluation.oracle_candidate_coverage as oracle_coverage


@pytest.mark.parametrize(
    "value",
    [
        7.25,
        "8.5",
        True,
        np.bool_(False),
        np.array([9]),
        np.ma.masked,
    ],
)
def test_oracle_candidate_coverage_rejects_malformed_identifiers(value: object) -> None:
    assert oracle_coverage._optional_int(value) is None


def test_oracle_candidate_coverage_preserves_large_exact_identifiers() -> None:
    identifier = "9007199254740993"

    assert oracle_coverage._optional_int(identifier) == 9007199254740993


def test_oracle_candidate_diagnostics_do_not_truncate_fractional_ids() -> None:
    candidates = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "track_id": [12.75],
            "track_index": [3.5],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    diagnostic = oracle_coverage._oracle_candidate_for_frame(
        candidates,
        truth=truth,
        truth_time_gate_s=1.0,
        truth_gate_m=None,
    )

    assert diagnostic["oracle_candidate_found"]
    assert diagnostic["oracle_track_id"] is None
    assert diagnostic["oracle_track_index"] is None
