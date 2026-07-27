from __future__ import annotations

import numpy as np
import pytest

from raft_uav.evaluation.metrics import summarize_errors


def test_summarize_errors_ignores_masked_samples() -> None:
    errors = np.ma.array(
        [1_000.0, 3.0, 4.0],
        mask=[True, False, False],
    )

    summary = summarize_errors(errors)

    assert summary["count"] == 2.0
    assert summary["mean_m"] == 3.5
    assert summary["mae_m"] == 3.5
    assert summary["std_m"] == 0.5
    assert summary["max_m"] == 4.0
    np.testing.assert_allclose(summary["rmse_m"], np.sqrt(12.5))


def test_summarize_errors_rejects_complex_arrays() -> None:
    with pytest.raises(ValueError, match="errors_m must contain only real values"):
        summarize_errors(np.array([3.0 + 4.0j]))
