from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_spread_guard_search import main as search_main
from raft_uav.mmuad.track5_spread_guard_search import search_track5_spread_guard_settings
from raft_uav.mmuad.track5_spread_guard_search import write_spread_guard_search_outputs


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


def _trusted_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [0.0, 1.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _outlier_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [20.0, 21.0, 24.0],
            "state_y_m": [20.0, 20.0, 24.0],
            "state_z_m": [20.0, 20.0, 24.0],
        }
    )


def test_spread_guard_search_selects_low_threshold_label_fallback(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    outlier = tmp_path / "outlier.csv"
    _trusted_estimate().to_csv(trusted, index=False)
    _outlier_estimate().to_csv(outlier, index=False)

    grid, best = search_track5_spread_guard_settings(
        [
            EstimateInput("trusted", trusted, 0.5),
            EstimateInput("outlier", outlier, 0.5),
        ],
        template=_template(),
        truth=_truth(),
        spread_thresholds_m=(0.0, 100.0),
        fallback_policies=("label",),
        fallback_labels=("trusted",),
    )

    assert len(grid) == 2
    assert best["spread_threshold_m"] == pytest.approx(0.0)
    assert best["fallback_policy"] == "label"
    assert best["fallback_label"] == "trusted"
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)
    high_threshold = grid.loc[grid["spread_threshold_m"] == 100.0].iloc[0]
    assert float(high_threshold["pose_mse_m2"]) > 0.0


def test_spread_guard_search_writes_best_submission(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    outlier = tmp_path / "outlier.csv"
    _trusted_estimate().to_csv(trusted, index=False)
    _outlier_estimate().to_csv(outlier, index=False)

    paths = write_spread_guard_search_outputs(
        estimate_inputs=[
            EstimateInput("trusted", trusted, 0.5),
            EstimateInput("outlier", outlier, 0.5),
        ],
        template=_template(),
        truth=_truth(),
        output_dir=tmp_path / "out",
        spread_thresholds_m=(0.0, 100.0),
        fallback_policies=("label",),
        fallback_labels=("trusted",),
        write_best_submission=True,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["grid_csv"].exists()
    assert paths["best_config_json"].exists()
    assert paths["best_official_zip"].exists()
    best = json.loads(paths["best_config_json"].read_text(encoding="utf-8"))
    assert best["fallback_label"] == "trusted"
    official = pd.read_csv(paths["best_official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_spread_guard_search_cli_writes_artifacts(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    outlier = tmp_path / "outlier.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    _trusted_estimate().to_csv(trusted, index=False)
    _outlier_estimate().to_csv(outlier, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = search_main(
        [
            "--estimate-csv",
            f"trusted={trusted}@0.5",
            "--estimate-csv",
            f"outlier={outlier}@0.5",
            "--template",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(tmp_path / "out"),
            "--spread-threshold-m",
            "0,100",
            "--fallback-policy",
            "label",
            "--fallback-label",
            "trusted",
        ]
    )

    assert status == 0
    assert (tmp_path / "out" / "mmuad_track5_spread_guard_search_grid.csv").exists()
    assert (tmp_path / "out" / "mmuad_track5_spread_guard_best_config.json").exists()


def test_spread_guard_search_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-spread-guard-search"]
        == "raft_uav.mmuad.track5_spread_guard_search:main"
    )
