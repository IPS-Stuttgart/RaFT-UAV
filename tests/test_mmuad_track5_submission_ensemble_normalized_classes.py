from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _normalized_submission(classification: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "classification": [classification],
        }
    )


@pytest.mark.parametrize(
    "classification",
    ["1.5", "1.000001", "4", "True", ""],
)
def test_load_track5_submission_rejects_malformed_normalized_classes(
    tmp_path: Path,
    classification: object,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    _normalized_submission(classification).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="invalid normalized Track 5 Classification",
    ):
        load_track5_submission(path)


def test_load_track5_submission_keeps_exact_normalized_classes(tmp_path: Path) -> None:
    path = tmp_path / "normalized_submission.csv"
    rows = pd.concat(
        [
            _normalized_submission(1),
            _normalized_submission("2.0").assign(sequence_id="002"),
            _normalized_submission(3.0).assign(sequence_id="003"),
        ],
        ignore_index=True,
    )
    rows.to_csv(path, index=False)

    loaded = load_track5_submission(path)

    assert loaded["Classification"].tolist() == [1, 2, 3]
