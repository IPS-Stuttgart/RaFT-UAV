from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [1.0] * 4,
            "source": ["a", "a", "b", "c"],
            "track_id": ["best", "duplicate", "protected", "far"],
            "x_m": [0.0, 0.1, 0.2, 5.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.9, 0.1, 0.5],
            "candidate_reservoir_protected": [False, False, True, False],
        }
    )


def test_diversity_suppresses_near_duplicate_and_keeps_far_candidate() -> None:
    output = diversify_candidate_reservoir(_rows(), radius_m=1.0)
    assert set(output["track_id"]) == {"best", "protected", "far"}
    assert "duplicate" not in set(output["track_id"])


def test_diversity_can_disable_protected_override() -> None:
    output = diversify_candidate_reservoir(
        _rows(), radius_m=1.0, preserve_protected=False
    )
    assert set(output["track_id"]) == {"best", "far"}


def test_diversity_respects_per_frame_cap() -> None:
    output = diversify_candidate_reservoir(_rows(), radius_m=0.0, max_candidates_per_frame=2)
    assert len(output) == 2
    assert output["candidate_diversity_rank"].tolist() == [1, 2]


def test_diversity_does_not_expand_duplicate_input_index_labels() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": [1.0] * 3,
            "track_id": ["best", "duplicate", "far"],
            "x_m": [0.0, 0.1, 5.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.9, 0.5],
            "candidate_reservoir_protected": [False, False, False],
        },
        index=[7, 7, 8],
    )

    output = diversify_candidate_reservoir(
        rows,
        radius_m=1.0,
        max_candidates_per_frame=2,
    )

    assert output["track_id"].tolist() == ["best", "far"]
    assert len(output) == 2


@pytest.mark.parametrize("serialized_false", ["False", "0", "off", "no"])
def test_diversity_does_not_protect_serialized_false_flags(
    serialized_false: str,
) -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "track_id": ["serialized_false", "best"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [0.1, 1.0],
            "candidate_reservoir_protected": [serialized_false, False],
        }
    )

    output = diversify_candidate_reservoir(
        rows,
        radius_m=0.0,
        max_candidates_per_frame=1,
    )

    assert output["track_id"].tolist() == ["best"]
