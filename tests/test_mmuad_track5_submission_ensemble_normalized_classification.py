from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _normalized_submission(classification: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "classification": [classification, 2],
        }
    )


@pytest.mark.parametrize("classification", ["1.5", "4", "true"])
def test_load_track5_submission_rejects_invalid_normalized_classification(
    tmp_path: Path,
    classification: object,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    _normalized_submission(classification).to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid Track 5 Classification"):
        load_track5_submission(path)
