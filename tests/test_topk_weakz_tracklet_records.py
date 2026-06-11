from __future__ import annotations

from raft_uav.baselines.topk_weakz_tracklet import (
    _POSTERIOR_FRAME_COLUMNS,
    records_to_frame,
)


def test_records_to_frame_preserves_schema_for_empty_records() -> None:
    frame = records_to_frame([])

    assert list(frame.columns) == _POSTERIOR_FRAME_COLUMNS
    assert frame.empty


def test_records_to_frame_uses_stable_schema_for_records() -> None:
    frame = records_to_frame(
        [
            {
                "time_s": 1.5,
                "source": "rf",
                "accepted": True,
                "update_action": "accepted",
                "state": [1, 2, 3, 4, 5, 6],
                "nis": 0.25,
                "residual_norm_m": 2.0,
            }
        ]
    )

    assert list(frame.columns) == _POSTERIOR_FRAME_COLUMNS
    assert frame.loc[0, "time_s"] == 1.5
    assert frame.loc[0, "east_m"] == 1.0
    assert frame.loc[0, "v_up_mps"] == 6.0
