from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from raft_uav.mmuad.class_probability_calibration import (
    classification_calibration_metrics,
    expected_calibration_error,
    multiclass_nll,
)


MetricFunction = Callable[[np.ndarray, np.ndarray], object]
_METRICS: tuple[MetricFunction, ...] = (
    classification_calibration_metrics,
    multiclass_nll,
    expected_calibration_error,
)
_PROBABILITIES = np.asarray([[0.1, 0.9]], dtype=float)


@pytest.mark.parametrize("metric", _METRICS, ids=lambda function: function.__name__)
@pytest.mark.parametrize(
    "invalid_truth",
    [
        pytest.param(np.asarray([-1]), id="negative"),
        pytest.param(np.asarray([2]), id="out-of-range"),
        pytest.param(np.asarray([0.5]), id="fractional"),
        pytest.param(np.asarray([True]), id="boolean"),
        pytest.param(np.ma.asarray([1], mask=[True]), id="masked"),
    ],
)
def test_calibration_metrics_reject_invalid_truth_indices(
    metric: MetricFunction,
    invalid_truth: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"truth_indices must contain exact class indices in \[0, 1\]",
    ):
        metric(_PROBABILITIES, invalid_truth)


@pytest.mark.parametrize("metric", _METRICS, ids=lambda function: function.__name__)
def test_calibration_metrics_reject_truth_length_mismatch(
    metric: MetricFunction,
) -> None:
    with pytest.raises(
        ValueError,
        match="one class index per probability row",
    ):
        metric(_PROBABILITIES, np.asarray([0, 1]))


def test_calibration_metrics_preserve_exact_scalar_like_indices() -> None:
    metrics = classification_calibration_metrics(
        _PROBABILITIES,
        np.asarray(["1"], dtype=object),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] == pytest.approx(-np.log(0.9))
    assert metrics["brier"] == pytest.approx(0.02)
