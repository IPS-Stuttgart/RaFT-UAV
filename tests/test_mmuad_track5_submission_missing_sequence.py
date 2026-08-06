from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


@pytest.mark.parametrize("missing_sequence", ["", "   "])
def test_normalized_track5_loader_rejects_missing_sequence_ids(
    tmp_path: Path,
    missing_sequence: str,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", missing_sequence],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [1.0, 1.0],
            "classification": [1, 1],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="invalid normalized Track 5 sequence_id row",
    ):
        load_track5_submission(path)
