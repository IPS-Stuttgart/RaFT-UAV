from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "timestamp": [0.0, 10.0],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "z": [2.0, 2.0],
            "uav_type": ["2", "2"],
            "score": [1.0, 1.0],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame({"sequence_id": ["seq1"], "time_s": [5.0]})


def _boxed(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


@pytest.mark.parametrize(
    "gap",
    [
        True,
        np.bool_(False),
        1.0 + 0.0j,
        [0.5],
        np.asarray([0.5]),
        np.ma.masked,
        _boxed(np.bool_(True)),
        _boxed(np.complex128(1.0 + 2.0j)),
        _boxed(np.asarray([0.5])),
    ],
)
def test_completion_rejects_non_real_scalar_interpolation_gaps(gap: object) -> None:
    with pytest.raises(
        ValueError,
        match="max_interpolation_gap_s must be a finite non-negative number",
    ):
        complete_results_to_truth_timestamps(
            _results(),
            _template(),
            max_interpolation_gap_s=gap,
        )


def test_completion_rejects_cyclic_interpolation_gap() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(
        ValueError,
        match="max_interpolation_gap_s must be a finite non-negative number",
    ):
        complete_results_to_truth_timestamps(
            _results(),
            _template(),
            max_interpolation_gap_s=cyclic,
        )


@pytest.mark.parametrize(
    "gap",
    [
        np.asarray(0.0),
        _boxed(np.asarray(0.0)),
    ],
)
def test_completion_accepts_zero_dimensional_interpolation_gap(gap: object) -> None:
    completed = complete_results_to_truth_timestamps(
        _results(),
        _template(),
        max_interpolation_gap_s=gap,
    )

    assert completed.rows.iloc[0]["x"] == 0.0
    assert completed.diagnostics["completion_method"].tolist() == ["hold_before"]
