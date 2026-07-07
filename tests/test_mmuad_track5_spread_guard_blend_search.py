from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_spread_guard_blend_search import main as blend_search_main
from raft_uav.mmuad.track5_spread_guard_blend_search import (
    search_track5_spread_guard_blend_settings,
)
from raft_uav.mmuad.track5_spread_guard_blend_search import write_spread_guard_blend_search_outputs


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
            "x_m": [2.0, 3.0, 6.0],
            "y_m": [0.0, 0.0, 6.0],
            "z_m": [0.0, 0.0, 6.0],
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
            "state_x_m": [8.0, 9.0, 12.0],
            "state_y_m": [0.0, 0.0, 12.0],
            "state_z_m": [0.0, 0.0, 12.0],
        }
    )


def test_spread_guard_blend_search_can_select_partial_fallback_blend(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    outlier = tmp_path / "outlier.csv"
    _trusted_estimate().to_csv(trusted, index=False)
    _outlier_estimate().to_csv(outlier, index=False)

    grid, best = search_track5_spread_guard_blend_settings(
        [
            EstimateInput("trusted", trusted, 0.5),
            EstimateInput("outlier", outlier, 0.5),
        ],
        template=_template(),
        truth=_truth(),
        spread_thresholds_m=(0.0,),
        fallback_policies=("label",),
        fallback_labels=("trusted",),
        fallback_blends=(0.0, 0.5, 1.0),
    )

    assert len(grid) == 3
    assert best["fallback_policy"] == "label"
    assert best["fallback_label"] == "trusted"
    assert best["fallback_blend"] == pytest.approx(0.5)
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)


def test_spread_guard_blend_search_writes_best_submission(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    outlier = tmp_path / "outlier.csv"
    _trusted_estimate().to_csv(trusted, index=False)
    _outlier_estimate().to_csv(outlier, index=False)

    paths = write_spread_guard_blend_search_outputs(
        estimate_inputs=[
            EstimateInput("trusted", trusted, 0.5),
            EstimateInput("outlier", outlier, 0.5),
        ],
        template=_template(),
        truth=_truth(),
        output_dir=tmp_path / "out",
        spread_thresholds_m=(0.0,),
        fallback_policies=("label",),
        fallback_labels=("trusted",),
        fallback_blends=(0.0, 0.5, 1.0),
        write_best_submission=True,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["grid_csv"].exists()
    assert paths["best_config_json"].exists()
    assert paths["best_official_zip"].exists()
    best = json.loads(paths["best_config_json"].read_text(encoding="utf-8"))
    assert best["fallback_blend"] == pytest.approx(0.5)
    official = pd.read_csv(paths["best_official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_spread_guard_blend_search_cli_and_entrypoint(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    outlier = tmp_path / "outlier.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    _trusted_estimate().to_csv(trusted, index=False)
    _outlier_estimate().to_csv(outlier, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = blend_search_main(
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
            "0",
            "--fallback-policy",
            "label",
            "--fallback-label",
            "trusted",
            "--fallback-blend",
            "0,0.5,1",
        ]
    )

    assert status == 0
    assert (tmp_path / "out" / "mmuad_track5_spread_guard_blend_search_grid.csv").exists()
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-spread-guard-blend-search"]
        == "raft_uav.mmuad.track5_spread_guard_blend_search:main"
    )
