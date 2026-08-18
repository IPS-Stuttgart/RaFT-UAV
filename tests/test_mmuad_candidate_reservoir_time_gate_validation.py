from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables


def _reservoir(*, time_s: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a"],
            "time_s": [time_s],
            "source": ["radar"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "confidence": [1.0],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )


@pytest.mark.parametrize(
    "max_delta_s",
    [
        np.nan,
        np.inf,
        -np.inf,
        -0.1,
        True,
        np.bool_(False),
        0.5 + 0.0j,
        [0.5],
        np.asarray([0.5]),
        np.ma.masked,
    ],
)
def test_oracle_recall_rejects_invalid_truth_time_gate(max_delta_s: object) -> None:
    with pytest.raises(
        ValueError,
        match="max_truth_time_delta_s must be a finite non-negative real scalar",
    ):
        build_oracle_recall_tables(
            _reservoir(),
            _truth(),
            max_truth_time_delta_s=max_delta_s,
        )


def test_oracle_recall_accepts_zero_dimensional_truth_time_gate() -> None:
    frame_rows, _, _ = build_oracle_recall_tables(
        _reservoir(time_s=0.25),
        _truth(),
        max_truth_time_delta_s=np.asarray(0.5),
    )

    assert len(frame_rows) == 1
    assert frame_rows.loc[0, "truth_time_delta_s"] == pytest.approx(0.25)


def test_oracle_recall_keeps_finite_time_gate_active() -> None:
    frame_rows, pooled, by_sequence = build_oracle_recall_tables(
        _reservoir(time_s=10.0),
        _truth(),
        max_truth_time_delta_s=0.5,
    )

    assert frame_rows.empty
    assert pooled.empty
    assert by_sequence.empty
