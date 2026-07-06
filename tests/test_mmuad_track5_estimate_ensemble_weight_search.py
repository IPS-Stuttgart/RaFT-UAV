from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_weight_search import main as weight_search_main
from raft_uav.mmuad.track5_estimate_ensemble_weight_search import search_track5_estimate_ensemble_weights
from raft_uav.mmuad.track5_estimate_ensemble_weight_search import write_weight_search_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "x_m": [0.0, 1.0, 4.0],
            "y_m": [0.0, 0.0, 4.0],
            "z_m": [0.0, 0.0, 4.0],
        }
    )


def _good_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [0.0, 1.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _bad_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [10.0, 11.0, 14.0],
            "state_y_m": [10.0, 10.0, 14.0],
            "state_z_m": [10.0, 10.0, 14.0],
        }
    )


def test_weight_search_selects_best_single_estimate(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)

    grid, best = search_track5_estimate_ensemble_weights(
        [
            EstimateInput("good", good_csv),
            EstimateInput("bad", bad_csv),
        ],
        template=_template(),
        truth=_truth(),
        weight_step=0.5,
    )

    by_sequence = grid.attrs["by_sequence"]
    assert len(grid) == 3
    assert len(by_sequence) == 6
    assert set(by_sequence["sequence_id"]) == {"seq0001", "seq0002"}
    assert best["weights"]["good"] == pytest.approx(1.0)
    assert best["weights"]["bad"] == pytest.approx(0.0)
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)
    assert len(best["by_sequence_metrics"]) == 2


def test_weight_search_matches_integer_and_decimal_timestamps(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    _good_estimate().to_csv(good_csv, index=False)
    truth = _truth()
    truth["time_s"] = pd.Series([0, 1, 0], dtype="int64")

    grid, best = search_track5_estimate_ensemble_weights(
        [EstimateInput("good", good_csv)],
        template=_template(),
        truth=truth,
        weight_step=1.0,
    )

    assert int(grid.loc[0, "matched_rows"]) == 3
    assert best["metrics"]["matched_rows"] == 3
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)


def test_weight_search_writes_grid_best_and_submission(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)

    paths = write_weight_search_outputs(
        estimate_inputs=[
            EstimateInput("good", good_csv),
            EstimateInput("bad", bad_csv),
        ],
        template=_template(),
        truth=_truth(),
        output_dir=tmp_path / "out",
        weight_step=0.5,
        write_best_submission=True,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["weight_grid_csv"].exists()
    assert paths["weight_grid_by_sequence_csv"].exists()
    assert paths["best_weights_json"].exists()
    assert paths["best_official_zip"].exists()
    best = json.loads(paths["best_weights_json"].read_text(encoding="utf-8"))
    assert best["weights"] == {"good": 1.0, "bad": 0.0}
    grid = pd.read_csv(paths["weight_grid_csv"])
    by_sequence = pd.read_csv(paths["weight_grid_by_sequence_csv"])
    assert set(grid.columns).issuperset({"weight_good", "weight_bad", "pose_mse_m2"})
    assert set(by_sequence.columns).issuperset(
        {"weight_grid_index", "sequence_id", "weight_good", "weight_bad", "pose_mse_m2"}
    )


def test_weight_search_cli_writes_best_config(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = weight_search_main(
        [
            "--estimate-csv",
            f"good={good_csv}",
            "--estimate-csv",
            f"bad={bad_csv}",
            "--template",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--weight-step",
            "0.5",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_ensemble_weight_grid.csv").exists()
    assert (output_dir / "mmuad_track5_ensemble_weight_grid_by_sequence.csv").exists()
    assert (output_dir / "mmuad_track5_ensemble_best_weights.json").exists()


def test_weight_search_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-ensemble-weight-search"]
        == "raft_uav.mmuad.track5_estimate_ensemble_weight_search:main"
    )
