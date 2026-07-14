from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_mixture_gap_frames import (
    build_frame_gap_table,
)


def test_exact_frame_gap_join_does_not_match_invalid_timestamps() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": ["not-a-time"],
            "position_error_3d_m": [3.0],
        }
    )
    oracle_frames = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": ["also-not-a-time"],
            "oracle_all_3d_m": [1.0],
        }
    )

    gap = build_frame_gap_table(estimates, oracle_frames)

    assert gap.empty


@pytest.mark.parametrize(
    ("duplicate_input", "message"),
    [
        ("estimates", "estimates contain duplicate frame keys"),
        ("oracle", "oracle frames contain duplicate frame keys"),
    ],
)
def test_exact_frame_gap_join_rejects_duplicate_rounded_frame_keys(
    duplicate_input: str,
    message: str,
) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "position_error_3d_m": [3.0],
        }
    )
    oracle_frames = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "oracle_all_3d_m": [1.0],
        }
    )
    duplicated_times = [0.0000001, 0.0000002]
    if duplicate_input == "estimates":
        estimates = pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA"],
                "time_s": duplicated_times,
                "position_error_3d_m": [3.0, 4.0],
            }
        )
    else:
        oracle_frames = pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA"],
                "time_s": duplicated_times,
                "oracle_all_3d_m": [1.0, 2.0],
            }
        )

    with pytest.raises(ValueError, match=message):
        build_frame_gap_table(
            estimates,
            oracle_frames,
            time_round_decimals=6,
        )
