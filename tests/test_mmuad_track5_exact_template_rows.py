from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    build_track5_consensus_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_ensemble import (
    EstimateInput,
    build_track5_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import (
    build_spread_guarded_estimate_ensemble,
)
from raft_uav.mmuad.track5_uncertainty_ensemble import (
    build_track5_uncertainty_ensemble,
)


_CLOSE_TIMES = [0.0, 0.5e-9]


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq_close", "seq_close"],
            "Timestamp": _CLOSE_TIMES,
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [1, 1],
        }
    )


def _estimates(*, with_sigma: bool = False) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq_close", "seq_close"],
            "time_s": _CLOSE_TIMES,
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [1.0, 1.0],
        }
    )
    if with_sigma:
        rows["predicted_sigma_m"] = 1.0
    return rows


def _assert_independent_rows(
    estimates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    estimate_count_column: str,
    diagnostic_count_column: str,
) -> None:
    assert estimates["time_s"].tolist() == _CLOSE_TIMES
    assert estimates["state_x_m"].tolist() == pytest.approx([0.0, 100.0])
    assert estimates[estimate_count_column].tolist() == [1, 1]
    assert diagnostics[diagnostic_count_column].tolist() == [1, 1]


def test_fixed_weight_ensemble_keeps_close_template_rows_independent() -> None:
    estimates, diagnostics = build_track5_estimate_ensemble(
        [("only", _estimates(), 1.0)],
        _template(),
        max_nearest_time_delta_s=0.0,
    )

    _assert_independent_rows(
        estimates,
        diagnostics,
        estimate_count_column="ensemble_source_count",
        diagnostic_count_column="candidate_input_count",
    )


def test_uncertainty_ensemble_keeps_close_template_rows_independent(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimates(with_sigma=True).to_csv(estimate_csv, index=False)

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("only", estimate_csv, 1.0)],
        template=_template(),
        max_nearest_time_delta_s=0.0,
    )

    _assert_independent_rows(
        estimates,
        diagnostics,
        estimate_count_column="ensemble_source_count",
        diagnostic_count_column="candidate_input_count",
    )


def test_consensus_ensemble_keeps_close_template_rows_independent() -> None:
    estimates, diagnostics = build_track5_consensus_estimate_ensemble(
        [("only", _estimates(), 1.0)],
        _template(),
        consensus_radius_m=1.0,
        max_nearest_time_delta_s=0.0,
    )

    _assert_independent_rows(
        estimates,
        diagnostics,
        estimate_count_column="consensus_input_count",
        diagnostic_count_column="valid_input_count",
    )


def test_spread_guard_keeps_close_template_rows_independent() -> None:
    estimates, diagnostics = build_spread_guarded_estimate_ensemble(
        [("only", _estimates(), 1.0)],
        _template(),
        spread_threshold_m=1_000.0,
        max_nearest_time_delta_s=0.0,
    )

    assert estimates["time_s"].tolist() == _CLOSE_TIMES
    assert estimates["state_x_m"].tolist() == pytest.approx([0.0, 100.0])
    assert diagnostics["valid_input_count"].tolist() == [1, 1]
