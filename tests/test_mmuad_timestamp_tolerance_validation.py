import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import evaluate_mmaud_results
from raft_uav.mmuad.submission import validate_official_track5_submission


_INVALID_TIMESTAMP_TOLERANCES = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(np.bool_(True), id="numpy-true"),
    pytest.param(np.array(True), id="zero-dimensional-boolean-array"),
    pytest.param(np.array([0.1]), id="non-scalar-array"),
)


def _official_results(*, timestamp: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [timestamp],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )


def _truth(*, timestamp: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [timestamp],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "uav_type": ["1"],
        }
    )


@pytest.mark.parametrize("tolerance", _INVALID_TIMESTAMP_TOLERANCES)
def test_public_track5_evaluator_rejects_invalid_timestamp_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite non-negative"):
        evaluate_mmaud_results(
            _official_results(timestamp=1000.0),
            _truth(timestamp=0.0),
            metric_protocol="public-track5",
            timestamp_tolerance_s=tolerance,
        )


@pytest.mark.parametrize("tolerance", _INVALID_TIMESTAMP_TOLERANCES)
def test_submission_validator_rejects_invalid_timestamp_tolerance(tmp_path, tolerance):
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_official_track5_submission(
            tmp_path / "missing.zip",
            timestamp_tolerance_s=tolerance,
        )


def test_public_track5_evaluator_keeps_finite_numpy_scalar_tolerance():
    evaluated = evaluate_mmaud_results(
        _official_results(timestamp=0.05),
        _truth(timestamp=0.0),
        metric_protocol="public-track5",
        timestamp_tolerance_s=np.float64(0.1),
    )

    assert evaluated["summary"]["timestamp_tolerance_s"] == 0.1
    assert evaluated["summary"]["matched_count"] == 1
