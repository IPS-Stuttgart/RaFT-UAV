from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble


def _write_estimate(path: Path, *, x_m: float) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
            "predicted_sigma_m": [1.0],
        }
    ).to_csv(path, index=False)


def test_uncertainty_ensemble_large_weights_do_not_overflow(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_estimate(first, x_m=0.0)
    _write_estimate(second, x_m=10.0)
    template = pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [
            EstimateInput("first", first, 9.0e307),
            EstimateInput("second", second, 1.0e308),
        ],
        template=template,
    )

    row = estimates.iloc[0]
    assert np.isfinite(
        row[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all()
    assert row["state_x_m"] == pytest.approx(5.2631578947368425)
    assert row["ensemble_position_spread_m"] == pytest.approx(4.986149584487534)
    assert row["ensemble_effective_sigma_m"] == pytest.approx(7.254762501100116e-155)
    assert diagnostics.iloc[0]["position_spread_m"] == pytest.approx(4.986149584487534)
    assert [entry["global_weight"] for entry in diagnostics.attrs["input_summary"]] == [
        9.0e307,
        1.0e308,
    ]
