from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import load_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble_grid import EnsembleGridRow
from raft_uav.mmuad.track5_estimate_ensemble_grid import _row_sort_key
from raft_uav.mmuad.track5_estimate_ensemble_grid import evaluate_estimate_ensemble_weight_grid
from raft_uav.mmuad.track5_estimate_ensemble_grid import generate_simplex_weight_grid
from raft_uav.mmuad.track5_estimate_ensemble_grid import main as grid_main
from raft_uav.mmuad.track5_estimate_ensemble_grid import write_estimate_ensemble_weight_grid_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 5.0, 0.0],
            "x_m": [0.0, 5.0, 4.0],
            "y_m": [0.0, 0.0, 4.0],
            "z_m": [0.0, 0.0, 4.0],
            "class_name": ["2", "2", "1"],
        }
    )


def _estimate_good() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _estimate_bad() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [20.0, 30.0, 24.0],
            "state_y_m": [20.0, 20.0, 24.0],
            "state_z_m": [20.0, 20.0, 24.0],
        }
    )


def _estimate_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [100.0, 110.0, 104.0],
            "state_y_m": [100.0, 100.0, 104.0],
            "state_z_m": [100.0, 100.0, 104.0],
        }
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    template = tmp_path / "template.csv"
    truth = tmp_path / "truth.csv"
    class_map = tmp_path / "class_map.csv"
    _estimate_good().to_csv(good, index=False)
    _estimate_bad().to_csv(bad, index=False)
    _template().to_csv(template, index=False)
    _truth().to_csv(truth, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map,
        index=False,
    )
    return good, bad, template, truth, class_map


def test_generate_simplex_weight_grid() -> None:
    grid = generate_simplex_weight_grid(2, step=0.5)
    assert set(grid) == {(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)}
    assert generate_simplex_weight_grid(2, step=0.5, include_singletons=False) == [
        (0.5, 0.5)
    ]
    with pytest.raises(ValueError, match="divide 1.0"):
        generate_simplex_weight_grid(2, step=0.3)


def test_estimate_ensemble_weight_grid_selects_best_weight(tmp_path: Path) -> None:
    good, bad, _, _, _ = _write_inputs(tmp_path)
    summary, by_sequence, best_weights = evaluate_estimate_ensemble_weight_grid(
        [parse_estimate_spec(f"good={good}"), parse_estimate_spec(f"bad={bad}")],
        template=_template(),
        truth=_truth(),
        weight_grid=generate_simplex_weight_grid(2, step=0.5),
        default_classification=2,
    )

    assert best_weights == (1.0, 0.0)
    assert summary.iloc[0]["weight_good"] == pytest.approx(1.0)
    assert summary.iloc[0]["aggregation_policy"] == "weighted-mean"
    assert summary.iloc[0]["pose_mse"] == pytest.approx(0.0)
    assert set(by_sequence["sequence_id"]) == {"seq0001", "seq0002"}


def test_estimate_ensemble_policy_grid_can_choose_robust_policy(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    outlier = tmp_path / "outlier.csv"
    _estimate_good().to_csv(good, index=False)
    _estimate_bad().to_csv(bad, index=False)
    _estimate_outlier().to_csv(outlier, index=False)

    summary, _, best_weights = evaluate_estimate_ensemble_weight_grid(
        [
            parse_estimate_spec(f"good={good}"),
            parse_estimate_spec(f"bad={bad}"),
            parse_estimate_spec(f"outlier={outlier}"),
        ],
        template=_template(),
        truth=_truth(),
        weight_grid=[(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)],
        default_classification=2,
        aggregation_policies=("weighted-mean", "weighted-median"),
    )

    assert best_weights == (pytest.approx(1.0 / 3.0),) * 3
    assert summary.iloc[0]["aggregation_policy"] == "weighted-median"
    assert summary.iloc[0]["pose_mse"] < summary.iloc[-1]["pose_mse"]


def test_estimate_ensemble_grid_sort_key_demotes_nonfinite_rows() -> None:
    nonfinite = EnsembleGridRow(
        weights=(1.0, 0.0),
        aggregation_policy="weighted-mean",
        trim_fraction=0.2,
        pose_mse=float("nan"),
        rmse_m=float("nan"),
        mean_error_m=float("nan"),
        p95_error_m=float("nan"),
        max_error_m=float("nan"),
        class_accuracy=None,
        matched_count=0,
    )
    finite = EnsembleGridRow(
        weights=(0.0, 1.0),
        aggregation_policy="weighted-mean",
        trim_fraction=0.2,
        pose_mse=1.0,
        rmse_m=1.0,
        mean_error_m=1.0,
        p95_error_m=2.0,
        max_error_m=3.0,
        class_accuracy=None,
        matched_count=3,
    )

    assert _row_sort_key(finite) < _row_sort_key(nonfinite)


def test_estimate_ensemble_weight_grid_writes_best_artifacts(tmp_path: Path) -> None:
    good, bad, _, _, class_map = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    paths = write_estimate_ensemble_weight_grid_outputs(
        estimate_inputs=[parse_estimate_spec(f"good={good}"), parse_estimate_spec(f"bad={bad}")],
        template=_template(),
        truth=_truth(),
        weight_grid=generate_simplex_weight_grid(2, step=0.5),
        output_dir=output_dir,
        class_map_path=class_map,
    )

    assert paths["summary_csv"].exists()
    assert paths["manifest_json"].exists()
    assert paths["best_config_json"].exists()
    assert paths["best_official_zip"].exists()
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    best_config = json.loads(paths["best_config_json"].read_text(encoding="utf-8"))
    assert manifest["best_weights"] == [1.0, 0.0]
    assert manifest["best_weight_config_json"] == str(paths["best_config_json"])
    assert manifest["best_aggregation_policy"] == "weighted-mean"
    assert manifest["best"]["pose_mse"] == pytest.approx(0.0)
    assert best_config["weights"] == {"good": 1.0, "bad": 0.0}
    assert best_config["aggregation_policy"] == "weighted-mean"
    assert load_estimate_weight_config(paths["best_config_json"]) == {"good": 1.0, "bad": 0.0}


def test_estimate_ensemble_weight_grid_cli_writes_outputs(tmp_path: Path) -> None:
    good, bad, template, truth, class_map = _write_inputs(tmp_path)
    output_dir = tmp_path / "out_cli"
    status = grid_main(
        [
            "--estimate-csv",
            f"good={good}",
            "--estimate-csv",
            f"bad={bad}",
            "--template",
            str(template),
            "--truth",
            str(truth),
            "--class-map",
            str(class_map),
            "--output-dir",
            str(output_dir),
            "--step",
            "0.5",
            "--aggregation-policy",
            "grid",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_estimate_ensemble_weight_grid.csv").exists()
    assert (output_dir / "mmuad_track5_estimate_ensemble_best_config.json").exists()
    assert (output_dir / "best_ensemble" / "ug2_submission.zip").exists()
    manifest = json.loads((output_dir / "mmuad_track5_estimate_ensemble_weight_grid_manifest.json").read_text())
    assert manifest["best_aggregation_policy"] in {"weighted-mean", "weighted-median", "trimmed-mean"}


def test_estimate_ensemble_weight_grid_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-ensemble-grid"]
        == "raft_uav.mmuad.track5_estimate_ensemble_grid:main"
    )
