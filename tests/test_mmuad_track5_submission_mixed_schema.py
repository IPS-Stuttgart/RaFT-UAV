from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _normalized_columns() -> dict[str, list[object]]:
    return {
        "sequence_id": ["seq0001"],
        "time_s": [1.0],
        "state_x_m": [10.0],
        "state_y_m": [20.0],
        "state_z_m": [30.0],
    }


def test_malformed_official_columns_cannot_fall_back_to_normalized_schema(
    tmp_path,
) -> None:
    rows = pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [1.0],
            "Position": ["not-a-position"],
            "Classification": [2],
            **_normalized_columns(),
        }
    )
    path = tmp_path / "mixed_schema.csv"
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid Track 5"):
        load_track5_submission(path)


def test_normalized_only_submission_still_uses_the_compatibility_fallback(
    tmp_path,
) -> None:
    rows = pd.DataFrame(
        {
            **_normalized_columns(),
            "Classification": [2],
        }
    )
    path = tmp_path / "normalized.csv"
    rows.to_csv(path, index=False)

    loaded = load_track5_submission(path)

    assert loaded[["sequence_id", "time_s"]].to_dict("records") == [
        {"sequence_id": "seq0001", "time_s": 1.0}
    ]
    assert loaded.loc[0, "Classification"] == 2
    assert loaded.loc[0, "state_x_m"] == 10.0
