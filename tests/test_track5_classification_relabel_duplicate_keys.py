from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import relabel_track5_classification


def _pose_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,1)", "(1,0,1)", "(5,0,2)"],
            "Classification": [0, 0, 3],
        }
    )


def _classification_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(9,9,9)", "(8,8,8)", "(7,7,7)"],
            "Classification": [1, 1, 2],
        }
    )


@pytest.mark.parametrize("mode", ["by-sequence-majority", "by-nearest-time"])
def test_relabel_rejects_duplicate_classification_row_keys(mode: str) -> None:
    source = _classification_rows()
    duplicate = source.iloc[[0]].copy()
    duplicate["Classification"] = 2
    source = pd.concat([source, duplicate], ignore_index=True)

    with pytest.raises(
        ValueError,
        match=r"classification_submission contains duplicate Sequence/Timestamp keys",
    ):
        relabel_track5_classification(_pose_rows(), source, mode=mode)


def test_relabel_rejects_duplicate_pose_row_keys() -> None:
    pose = pd.concat([_pose_rows(), _pose_rows().iloc[[0]]], ignore_index=True)

    with pytest.raises(
        ValueError,
        match=r"pose_submission contains duplicate Sequence/Timestamp keys",
    ):
        relabel_track5_classification(
            pose,
            _classification_rows(),
            mode="by-sequence-majority",
        )


def test_relabel_allows_same_timestamp_in_different_sequences() -> None:
    result = relabel_track5_classification(
        _pose_rows(),
        _classification_rows(),
        mode="by-nearest-time",
    )

    assert result.rows["Classification"].tolist() == [1, 1, 2]
