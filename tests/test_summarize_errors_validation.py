import numpy as np
import pytest

from raft_uav.evaluation.metrics import summarize_errors


def test_summarize_errors_rejects_negative_position_errors():
    with pytest.raises(ValueError, match="non-negative"):
        summarize_errors(np.array([1.0, -0.25, np.nan]))
