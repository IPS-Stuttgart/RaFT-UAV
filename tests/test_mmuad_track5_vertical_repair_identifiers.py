from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001", "001", "001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [10.0, 10.0, 10.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize("invalid_sequence_id", [None, "", "   ", np.nan, True])
def test_vertical_repair_rejects_invalid_sequence_ids(
    invalid_sequence_id: object,
) -> None:
    submission = _submission_rows()
    submission["sequence_id"] = submission["sequence_id"].astype(object)
    submission.loc[1, "sequence_id"] = invalid_sequence_id

    with pytest.raises(
        ValueError,
        match=r"invalid fixed-grid identifiers: sequence_id rows \[1\]",
    ):
        repair_track5_vertical_spikes(submission)


@pytest.mark.parametrize("invalid_classification", [-1, 4, 1.5, "unknown"])
def test_vertical_repair_rejects_invalid_classifications(
    invalid_classification: object,
) -> None:
    submission = _submission_rows()
    submission["Classification"] = submission["Classification"].astype(object)
    submission.loc[1, "Classification"] = invalid_classification

    with pytest.raises(
        ValueError,
        match=r"invalid fixed-grid identifiers: Classification rows \[1\]",
    ):
        repair_track5_vertical_spikes(submission)


def test_vertical_repair_accepts_canonicalizable_identifiers() -> None:
    submission = _submission_rows()
    submission["sequence_id"] = ["001", np.str_("001"), 1]
    submission["Classification"] = [np.int64(2), "2.0", 3.0]

    repaired, _diagnostics = repair_track5_vertical_spikes(submission)

    assert repaired["sequence_id"].tolist() == ["001", "001", "1"]
    assert repaired["Classification"].tolist() == [2.0, 2.0, 3.0]
