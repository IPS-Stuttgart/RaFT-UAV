from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import evaluate_mmaud_results


def _results_and_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "timestamp": [0.0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "uav_type": ["0"],
            "score": [1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [10.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    return results, truth


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.array([0.5]),
        np.array([True]),
        -0.1,
        float("nan"),
        float("inf"),
        0.5 + 0.0j,
        np.ma.masked,
    ],
)
def test_nearest_time_evaluator_rejects_invalid_time_gates(value: object) -> None:
    results, truth = _results_and_truth()

    with pytest.raises(
        ValueError,
        match="max_time_delta_s must be a finite nonnegative real scalar",
    ):
        evaluate_mmaud_results(
            results,
            truth,
            max_time_delta_s=value,
        )


def test_nearest_time_evaluator_validates_gate_before_empty_truth_return() -> None:
    results, truth = _results_and_truth()

    with pytest.raises(ValueError, match="max_time_delta_s"):
        evaluate_mmaud_results(
            results,
            truth.iloc[0:0],
            max_time_delta_s=float("nan"),
        )


@pytest.mark.parametrize("value", [10.0, "10.0", np.float64(10.0), np.array(10.0)])
def test_nearest_time_evaluator_accepts_real_scalar_like_time_gates(value: object) -> None:
    results, truth = _results_and_truth()

    evaluation = evaluate_mmaud_results(
        results,
        truth,
        max_time_delta_s=value,
    )

    assert evaluation["summary"]["max_time_delta_s"] == pytest.approx(10.0)
    assert evaluation["summary"]["matched_count"] == 1
