import numpy as np
import pytest

from raft_uav.evaluation.metrics import summarize_errors


def test_summarize_errors_ignores_masked_backing_values() -> None:
    errors = np.ma.array([1.0, 100.0], mask=[False, True])

    summary = summarize_errors(errors)

    assert summary["count"] == 1.0
    assert summary["mean_m"] == 1.0
    assert summary["max_m"] == 1.0


def test_summarize_errors_treats_all_masked_input_as_empty() -> None:
    errors = np.ma.array([10.0, 20.0], mask=True)

    summary = summarize_errors(errors)

    assert summary["count"] == 0.0
    assert summary["mean_m"] is None
    assert summary["max_m"] is None


def test_summarize_errors_rejects_complex_values() -> None:
    with pytest.raises(ValueError, match="errors_m must contain only real values"):
        summarize_errors(np.array([1.0 + 2.0j]))
