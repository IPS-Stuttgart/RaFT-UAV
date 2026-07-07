from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_weight_search import generate_simplex_weight_grid
from raft_uav.mmuad.track5_estimate_weight_search import main as weight_search_main
from raft_uav.mmuad.track5_estimate_weight_search import search_estimate_ensemble_weights
from raft_uav.mmuad.track5_estimate_weight_search import write_estimate_weight_search_outputs


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 5.0, 0.0],
            "x_m": [0.0, 5.0, 10.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 2.0],
            "class_name": ["2", "2", "1"],
        }
    )


def _good_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 5.0, 0.0],
            "state_x_m": [0.0, 5.0, 10.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 2.0],
        }
    )


def _bad_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 5.0, 0.0],
            "state_x_m": [10.0, 15.0, 20.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 2.0],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)", "(5,0,0)", "(10,0,2)"],
            "Classification": [2, 2, 1],
        }
    )


def test_simplex_weight_grid_sums_to_one() -> None:
    grid = generate_simplex_weight_grid(3, step=0.5)
    assert len(grid) == 6
    assert all(sum(row) == pytest.approx(1.0) for row in grid)
    assert (1.0, 0.0, 0.0) in grid


def test_weight_search_selects_best_estimate(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)

    summary, best_estimates, _, best_weights = search_estimate_ensemble_weights(
        [parse_estimate_spec(f"good={good_csv}"), parse_estimate_spec(f"bad={bad_csv}")],
        truth=_truth(),
        template=_template(),
        class_map={"seq0001": "2", "seq0002": "1"},
        step=0.5,
    )

    assert best_weights == {"good": 1.0, "bad": 0.0}
    assert float(summary.iloc[0]["pose_mse"]) == pytest.approx(0.0)
    assert float(summary.iloc[0]["classification_accuracy"]) == pytest.approx(1.0)
    assert len(best_estimates) == 3


def test_weight_search_outputs_best_config_and_submission(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)

    paths = write_estimate_weight_search_outputs(
        estimate_inputs=[parse_estimate_spec(f"good={good_csv}"), parse_estimate_spec(f"bad={bad_csv}")],
        truth=_truth(),
        template=_template(),
        class_map={"seq0001": "2", "seq0002": "1"},
        output_dir=tmp_path / "out",
        step=0.5,
        write_best_submission=True,
    )

    assert paths["summary_csv"].exists()
    assert paths["best_config_json"].exists()
    payload = json.loads(paths["best_config_json"].read_text(encoding="utf-8"))
    assert payload["weights"] == {"good": 1.0, "bad": 0.0}
    assert paths["best_submission_official_zip"].exists()


def test_weight_search_cli_writes_artifacts(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    truth_csv = tmp_path / "truth.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)
    _truth().to_csv(truth_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = weight_search_main(
        [
            "--estimate-csv",
            f"good={good_csv}",
            "--estimate-csv",
            f"bad={bad_csv}",
            "--truth-file",
            str(truth_csv),
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--step",
            "0.5",
            "--write-best-submission",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_estimate_weight_search.csv").exists()
    assert (output_dir / "mmuad_track5_estimate_weight_search_best_config.json").exists()
    assert (output_dir / "best_submission" / "ug2_submission.zip").exists()
