from __future__ import annotations

import numpy as np
import pytest

from raft_uav.evaluation.oracle_gap_decomposition import OracleGapConfig


@pytest.mark.parametrize(
    "field_name",
    [
        "plausible_candidate_gate_m",
        "truth_time_gate_s",
        "estimate_time_gate_s",
        "drift_error_gate_m",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, np.bool_(False)])
def test_oracle_gap_config_rejects_nonfinite_and_boolean_thresholds(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name} must be a finite number"):
        OracleGapConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    [
        "plausible_candidate_gate_m",
        "truth_time_gate_s",
        "estimate_time_gate_s",
        "drift_error_gate_m",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_oracle_gap_config_rejects_nonpositive_thresholds(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name} must be positive"):
        OracleGapConfig(**{field_name: value})


def test_oracle_gap_config_accepts_finite_positive_numpy_scalars() -> None:
    config = OracleGapConfig(
        plausible_candidate_gate_m=np.float64(25.0),
        truth_time_gate_s=np.float32(0.5),
        estimate_time_gate_s=np.int64(2),
        drift_error_gate_m=np.float64(125.0),
    )

    assert float(config.plausible_candidate_gate_m) == pytest.approx(25.0)
    assert float(config.truth_time_gate_s) == pytest.approx(0.5)
    assert float(config.estimate_time_gate_s) == pytest.approx(2.0)
    assert float(config.drift_error_gate_m) == pytest.approx(125.0)
