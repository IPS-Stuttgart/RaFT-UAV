from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.oracle_candidate_coverage import (
    build_oracle_candidate_coverage_diagnostics,
)


def _empty_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    radar = pd.DataFrame(columns=["time_s", "east_m", "north_m", "up_m"])
    truth = pd.DataFrame(columns=["time_s", "east_m", "north_m", "up_m"])
    return radar, truth


@pytest.mark.parametrize(
    "gate",
    [-0.1, np.nan, np.inf, -np.inf, True, 1.0 + 0.0j, np.array([1.0])],
)
def test_oracle_coverage_rejects_invalid_truth_time_gate(gate: object) -> None:
    radar, truth = _empty_inputs()

    with pytest.raises(ValueError, match="truth_time_gate_s"):
        build_oracle_candidate_coverage_diagnostics(
            radar=radar,
            truth=truth,
            truth_time_gate_s=gate,
        )


@pytest.mark.parametrize(
    "gate",
    [-0.1, np.nan, np.inf, -np.inf, False, 1.0 + 0.0j, np.array([1.0])],
)
def test_oracle_coverage_rejects_invalid_truth_distance_gate(gate: object) -> None:
    radar, truth = _empty_inputs()

    with pytest.raises(ValueError, match="truth_gate_m"):
        build_oracle_candidate_coverage_diagnostics(
            radar=radar,
            truth=truth,
            truth_gate_m=gate,
        )


def test_oracle_coverage_accepts_zero_truth_gates() -> None:
    radar, truth = _empty_inputs()

    report, summary = build_oracle_candidate_coverage_diagnostics(
        radar=radar,
        truth=truth,
        truth_time_gate_s=np.array(0.0),
        truth_gate_m=0.0,
    )

    assert report.empty
    assert summary["radar_frame_count"] == 0
