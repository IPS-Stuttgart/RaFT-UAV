from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import (
    build_track5_uncertainty_ensemble,
)


def _write_sparse_sigma_estimate(path: Path) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 100.0],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "predicted_sigma_m": [1.0, np.nan],
        }
    ).to_csv(path, index=False)


def _template(time_s: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [time_s],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def test_uncertainty_ensemble_uses_fallback_for_stale_sigma(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _write_sparse_sigma_estimate(estimate_csv)

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("estimate", estimate_csv, 1.0)],
        template=_template(100.0),
        fallback_sigma_m=30.0,
        max_nearest_time_delta_s=1.0,
    )

    assert estimates.iloc[0]["ensemble_effective_sigma_m"] == pytest.approx(30.0)
    assert diagnostics.iloc[0]["effective_sigma_m"] == pytest.approx(30.0)
    assert diagnostics.attrs["input_summary"][0]["mean_sigma_m"] == pytest.approx(
        30.0
    )


def test_uncertainty_ensemble_keeps_nearby_sigma_with_freshness_gate(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _write_sparse_sigma_estimate(estimate_csv)

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("estimate", estimate_csv, 1.0)],
        template=_template(0.5),
        fallback_sigma_m=30.0,
        max_nearest_time_delta_s=1.0,
    )

    assert estimates.iloc[0]["ensemble_effective_sigma_m"] == pytest.approx(1.0)
    assert diagnostics.iloc[0]["effective_sigma_m"] == pytest.approx(1.0)


def test_uncertainty_ensemble_preserves_endpoint_hold_without_freshness_gate(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _write_sparse_sigma_estimate(estimate_csv)

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("estimate", estimate_csv, 1.0)],
        template=_template(100.0),
        fallback_sigma_m=30.0,
        max_nearest_time_delta_s=None,
    )

    assert estimates.iloc[0]["ensemble_effective_sigma_m"] == pytest.approx(1.0)
    assert diagnostics.iloc[0]["effective_sigma_m"] == pytest.approx(1.0)
