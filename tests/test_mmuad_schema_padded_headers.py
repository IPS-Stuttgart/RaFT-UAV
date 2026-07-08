import pandas as pd
import pytest

from raft_uav.mmuad.schema import (
    normalize_candidate_columns,
    normalize_time_column_aliases,
    normalize_truth_columns,
)


def test_candidate_normalizer_accepts_padded_canonical_and_alias_headers():
    raw = pd.DataFrame(
        {
            " Sequence_ID ": [" seqA "],
            " Time_S ": ["1.25"],
            " Source ": [" radar "],
            " Position.X ": [10.0],
            " Position.Y ": [20.0],
            " Position.Z ": [30.0],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqA"
    assert rows.loc[0, "time_s"] == pytest.approx(1.25)
    assert rows.loc[0, "source"] == "radar"
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [10.0, 20.0, 30.0]


def test_time_normalizer_accepts_padded_flattened_stamp_pair_headers():
    raw = pd.DataFrame({" header.stamp.sec ": [7], " header.stamp.nanosec ": [125_000_000]})

    rows = normalize_time_column_aliases(raw)

    assert rows.loc[0, "time_s"] == pytest.approx(7.125)


def test_truth_normalizer_accepts_padded_flattened_pose_headers():
    raw = pd.DataFrame(
        {
            " Sequence_ID ": ["seqB"],
            " stamp.sec ": [4],
            " stamp.nsec ": [500_000_000],
            " bbox.center.position.x ": [7.0],
            " bbox.center.position.y ": [8.0],
            " bbox.center.position.z ": [9.0],
        }
    )

    rows = normalize_truth_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqB"
    assert rows.loc[0, "time_s"] == pytest.approx(4.5)
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [7.0, 8.0, 9.0]
