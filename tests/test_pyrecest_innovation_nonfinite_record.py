from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from raft_uav.baselines.pyrecest_innovation_diagnostics import (
    raft_innovation_diagnostic_record,
)


@pytest.mark.parametrize("nis", [np.nan, np.inf, -np.inf])
def test_nonfinite_nis_does_not_export_zero_mahalanobis_distance(nis: float) -> None:
    diagnostic = SimpleNamespace(
        time=1.0,
        source="rf",
        measurement_dim=2,
        accepted=False,
        action="rejected",
        nis=nis,
        gate_threshold=5.991,
        residual_norm=10.0,
    )

    record = raft_innovation_diagnostic_record(diagnostic)

    assert record["mahalanobis_distance"] is None


def test_finite_nis_keeps_mahalanobis_distance() -> None:
    diagnostic = SimpleNamespace(
        time=1.0,
        source="rf",
        measurement_dim=2,
        accepted=True,
        action="updated",
        nis=9.0,
        gate_threshold=5.991,
        residual_norm=3.0,
    )

    record = raft_innovation_diagnostic_record(diagnostic)

    assert record["mahalanobis_distance"] == 3.0
