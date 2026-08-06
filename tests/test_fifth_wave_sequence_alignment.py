from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.fifth_wave_diagnostics import (
    estimate_error_frame,
    paired_error_delta_frame,
    vertical_horizontal_error_summary,
)


def _pooled_sequences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def test_error_alignment_does_not_cross_equal_timestamps_between_sequences() -> None:
    truth = _pooled_sequences()
    method_a = truth.copy()
    method_b = truth.copy()
    method_b["east_m"] += 10.0

    deltas = paired_error_delta_frame(
        method_a,
        method_b,
        truth,
        max_time_delta_s=0.0,
    )

    assert deltas["sequence_id"].tolist() == ["flight-a", "flight-b"]
    np.testing.assert_allclose(deltas["error_a_m"], [0.0, 0.0])
    np.testing.assert_allclose(deltas["error_b_m"], [10.0, 10.0])
    np.testing.assert_allclose(deltas["delta_error_m"], [-10.0, -10.0])

    errors = estimate_error_frame(
        method_a,
        truth,
        max_time_delta_s=0.0,
    )
    assert errors["sequence_id"].tolist() == ["flight-a", "flight-b"]
    np.testing.assert_allclose(errors["error_3d_m"], [0.0, 0.0])

    vertical = vertical_horizontal_error_summary(
        method_a,
        truth,
        max_time_delta_s=0.0,
    )
    assert vertical["matched_count"] == 2
    assert vertical["horizontal_rmse_m"] == 0.0
    assert vertical["up_rmse_m"] == 0.0


@pytest.mark.parametrize(
    ("function_name", "drop_from"),
    [
        ("paired", "method_a"),
        ("paired", "method_b"),
        ("paired", "truth"),
        ("estimate", "estimates"),
        ("estimate", "truth"),
    ],
)
def test_sequence_aware_alignment_rejects_partial_sequence_metadata(
    function_name: str,
    drop_from: str,
) -> None:
    truth = _pooled_sequences()
    method_a = truth.copy()
    method_b = truth.copy()
    frames = {
        "method_a": method_a,
        "method_b": method_b,
        "estimates": method_a,
        "truth": truth,
    }
    frames[drop_from] = frames[drop_from].drop(columns="sequence_id")

    with pytest.raises(
        ValueError,
        match="sequence_id must be present in every aligned frame",
    ):
        if function_name == "paired":
            paired_error_delta_frame(
                frames["method_a"],
                frames["method_b"],
                frames["truth"],
            )
        else:
            estimate_error_frame(
                frames["estimates"],
                frames["truth"],
            )
