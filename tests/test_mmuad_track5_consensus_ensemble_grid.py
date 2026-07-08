from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_consensus_ensemble_grid import main as consensus_grid_main
from raft_uav.mmuad.track5_consensus_ensemble_grid import search_track5_consensus_ensemble_grid
from raft_uav.mmuad.track5_consensus_ensemble_grid import write_consensus_grid_outputs
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


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
            "x_m": [0.0, 1.0, 5.0],
            "y_m": [0.0, 0.0, 5.0],
            "z_m": [0.0, 0.0, 5.0],
        }
    )


def _good_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [0.0, 1.0, 5.0],
            "state_y_m": [0.0, 0.0, 5.0],
            "state_z_m": [0.0, 0.0, 5.0],
        }
    )


def _outlier_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [100.0, 101.0, 105.0],
            "state_y_m": [100.0, 100.0, 105.0],
            "state_z_m": [100.0, 100.0, 105.0],
        }
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    good = tmp_path / "good.csv"
    outlier = tmp_path / "outlier.csv"
    template = tmp_path / "template.csv"
    truth = tmp_path / "truth.csv"
    class_map = tmp_path / "class_map.csv"
    _good_estimates().to_csv(good, index=False)
    _outlier_estimates().to_csv(outlier, index=False)
    _template().to_csv(template, index=False)
    _truth().to_csv(truth, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map,
        index=False,
    )
    return good, outlier, template, truth, class_map


def test_consensus_ensemble_grid_prefers_tight_consensus(tmp_path: Path) -> None:
    good, outlier, _, _, _ = _write_inputs(tmp_path)
    grid, best = search_track5_consensus_ensemble_grid(
        [
            EstimateInput("good", good, 1.0),
            EstimateInput("outlier", outlier, 1.0),
        ],
        template=_template(),
        truth=_truth(),
        consensus_radius_m=(1.0, 500.0),
        min_consensus_weight_fraction=(0.0,),
        fallback_policy=("max-weight",),
    )

    assert len(grid) == 2
    assert best["consensus_radius_m"] == pytest.approx(1.0)
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)
    wide = grid.loc[grid["consensus_radius_m"] == 500.0].iloc[0]
    assert float(wide["pose_mse_m2"]) > 0.0


def test_consensus_ensemble_grid_writes_best_submission(tmp_path: Path) -> None:
    good, outlier, template, truth, class_map = _write_inputs(tmp_path)
    output = tmp_path / "out"
    paths = write_consensus_grid_outputs(
        estimate_inputs=[
            EstimateInput("good", good, 1.0),
            EstimateInput("outlier", outlier, 1.0),
        ],
        template=_template(),
        truth=_truth(),
        output_dir=output,
        consensus_radius_m=(1.0,),
        min_consensus_weight_fraction=(0.0,),
        fallback_policy=("max-weight",),
        write_best_submission=True,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["grid_csv"].exists()
    assert paths["grid_by_sequence_csv"].exists()
    assert paths["best_config_json"].exists()
    assert paths["best_official_zip"].exists()
    best = json.loads(paths["best_config_json"].read_text(encoding="utf-8"))
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)


def test_consensus_ensemble_grid_cli_writes_outputs(tmp_path: Path) -> None:
    good, outlier, template, truth, class_map = _write_inputs(tmp_path)
    output = tmp_path / "cli"
    status = consensus_grid_main(
        [
            "--estimate-csv",
            f"good={good}",
            "--estimate-csv",
            f"outlier={outlier}",
            "--template",
            str(template),
            "--truth-csv",
            str(truth),
            "--output-dir",
            str(output),
            "--consensus-radius-m",
            "1,500",
            "--min-consensus-weight-fraction",
            "0",
            "--fallback-policy",
            "max-weight",
            "--write-best-submission",
            "--class-map",
            str(class_map),
        ]
    )

    assert status == 0
    assert (output / "mmuad_track5_consensus_ensemble_grid.csv").exists()
    assert (output / "best_consensus_ensemble" / "ug2_submission.zip").exists()


def test_consensus_ensemble_grid_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-consensus-ensemble-grid"]
        == "raft_uav.mmuad.track5_consensus_ensemble_grid:main"
    )
