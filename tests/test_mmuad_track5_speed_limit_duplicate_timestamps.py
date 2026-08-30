from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission(
    sequence_ids: list[str],
    times: list[float],
    x_positions: list[float],
) -> pd.DataFrame:
    row_count = len(sequence_ids)
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "time_s": times,
            "state_x_m": x_positions,
            "state_y_m": [0.0] * row_count,
            "state_z_m": [0.0] * row_count,
            "Classification": [1] * row_count,
        }
    )


def test_speed_limit_rejects_duplicate_timestamps_within_sequence() -> None:
    submission = _submission(
        ["seq", "seq"],
        [1.0, 1.0],
        [0.0, 1_000.0],
    )

    with pytest.raises(
        ValueError,
        match="duplicate timestamps within a sequence",
    ):
        project_track5_speed_limit(submission, max_speed_mps=1.0)


def test_speed_limit_allows_same_timestamp_in_different_sequences() -> None:
    submission = _submission(
        ["seq-a", "seq-b"],
        [1.0, 1.0],
        [0.0, 1_000.0],
    )

    limited, diagnostics = project_track5_speed_limit(
        submission,
        max_speed_mps=1.0,
    )

    assert len(limited) == 2
    assert len(diagnostics) == 2
    assert limited["sequence_id"].tolist() == ["seq-a", "seq-b"]
