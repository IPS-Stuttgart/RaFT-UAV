from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs
from raft_uav.mmuad.track5_estimate_ensemble_weight_search import search_track5_estimate_ensemble_weights


def _padded_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )


def _padded_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )


def _write_numeric_like_estimate(path: Path) -> None:
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    ).to_csv(path, index=False)


def test_track5_estimate_ensemble_preserves_padded_sequence_ids_from_csv(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _write_numeric_like_estimate(estimate_csv)

    paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=[parse_estimate_spec(f"candidate={estimate_csv}")],
        template=_padded_template(),
        output_dir=tmp_path / "out",
        default_classification=1,
        max_nearest_time_delta_s=0.0,
    )

    official = pd.read_csv(paths["official_results_csv"], dtype=str, keep_default_na=False)
    assert official.loc[0, "Sequence"] == "001"
    assert official.loc[0, "Position"] == "(1,2,3)"


def test_weight_search_preserves_padded_estimate_sequence_ids_from_csv(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _write_numeric_like_estimate(estimate_csv)

    grid, best = search_track5_estimate_ensemble_weights(
        [EstimateInput("candidate", estimate_csv)],
        template=_padded_template(),
        truth=_padded_truth(),
        weight_step=1.0,
    )

    assert int(grid.loc[0, "matched_rows"]) == 1
    assert best["metrics"]["matched_rows"] == 1
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)
