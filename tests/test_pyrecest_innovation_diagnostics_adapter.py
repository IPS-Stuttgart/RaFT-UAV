from __future__ import annotations

import numpy as np

from raft_uav.baselines.pyrecest_innovation_diagnostics import (
    normalized_innovation_squared,
    raft_innovation_diagnostic_record,
    raft_linear_innovation_diagnostic,
    summarize_raft_innovation_records,
)


def test_raft_linear_innovation_diagnostic_wraps_pyrecest() -> None:
    diagnostic = raft_linear_innovation_diagnostic(
        mean=np.array([0.0, 0.0, 0.0]),
        covariance_matrix=np.eye(3),
        measurement_vector=np.array([2.0, 0.0]),
        observation_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        measurement_covariance=np.eye(2),
        gate_threshold=3.0,
        source="rf",
        action="updated",
        time_s=10.0,
    )

    assert diagnostic.accepted is True
    assert diagnostic.source == "rf"
    assert diagnostic.action == "updated"
    assert np.isclose(diagnostic.nis, 2.0)

    record = raft_innovation_diagnostic_record(diagnostic)
    assert record["time_s"] == 10.0
    assert record["measurement_dim"] == 2
    assert record["update_action"] == "updated"
    assert record["residual_norm_m"] == 2.0


def test_raft_linear_innovation_diagnostic_rejects_outlier() -> None:
    diagnostic = raft_linear_innovation_diagnostic(
        mean=np.zeros(2),
        covariance_matrix=np.eye(2),
        measurement_vector=np.array([10.0, 0.0]),
        observation_matrix=np.eye(2),
        measurement_covariance=np.eye(2),
        gate_threshold=5.991,
    )

    assert diagnostic.accepted is False
    assert diagnostic.nis > 5.991


def test_normalized_innovation_squared_reexport() -> None:
    assert np.isclose(
        normalized_innovation_squared(np.array([2.0, 1.0]), np.diag([4.0, 1.0])),
        2.0,
    )


def test_summarize_raft_innovation_records() -> None:
    summaries = summarize_raft_innovation_records(
        [
            {"source": "rf", "accepted": True, "measurement_dim": 2, "nis": 1.0, "residual_norm_m": 2.0},
            {"source": "rf", "accepted": False, "measurement_dim": 2, "nis": 9.0, "residual_norm_m": 5.0},
            {"source": "radar", "accepted": True, "measurement_dim": 3, "nis": 2.0, "residual_norm_m": 1.0},
        ],
        source="rf",
    )

    by_group = {summary.group: summary for summary in summaries}
    assert by_group["rf"].count == 2
    assert by_group["rf"].accepted_count == 1
    assert by_group["rf"].rejected_count == 1
    assert np.isclose(by_group["rf"].nis_mean, 5.0)
