from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
    main as geomedian_main,
    weighted_geometric_median,
    write_track5_geometric_median_outputs,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _estimate_near_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 0.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _estimate_near_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.2, 0.2, 4.2],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _estimate_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [90.0, 90.0, 90.0],
            "state_y_m": [0.0, 0.0, 90.0],
            "state_z_m": [0.0, 0.0, 90.0],
        }
    )


def test_weighted_geometric_median_rejects_single_outlier() -> None:
    center, iterations, displacement = weighted_geometric_median(
        pd.DataFrame(
            {
                "x": [0.0, 0.0, 90.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            }
        ).to_numpy(float),
        pd.Series([1.0, 1.0, 1.0]).to_numpy(float),
    )

    assert center[0] == pytest.approx(0.0, abs=1.0e-3)
    assert center[1] == pytest.approx(0.0, abs=1.0e-6)
    assert iterations > 0
    assert displacement < 1.0e-4


def test_track5_geomedian_ensemble_is_robust_to_outlier_trajectory() -> None:
    estimates, diagnostics = build_track5_geometric_median_ensemble(
        [
            ("near_a", _estimate_near_a(), 1.0),
            ("near_b", _estimate_near_b(), 1.0),
            ("outlier", _estimate_outlier(), 1.0),
        ],
        _template(),
    )

    midpoint = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(0.2, abs=0.25)
    assert midpoint["state_y_m"] == pytest.approx(0.0, abs=0.1)
    assert midpoint["state_z_m"] == pytest.approx(0.0, abs=0.1)
    assert midpoint["geomedian_source_count"] == 3
    assert diagnostics["geomedian_to_weighted_mean_m"].max() > 20.0


def test_track5_geomedian_outputs_leaderboard_ready_zip(tmp_path: Path) -> None:
    near_a = tmp_path / "near_a.csv"
    near_b = tmp_path / "near_b.csv"
    outlier = tmp_path / "outlier.csv"
    _estimate_near_a().to_csv(near_a, index=False)
    _estimate_near_b().to_csv(near_b, index=False)
    _estimate_outlier().to_csv(outlier, index=False)

    paths = write_track5_geometric_median_outputs(
        estimate_inputs=[
            EstimateInput("near_a", near_a, 1.0),
            EstimateInput("near_b", near_b, 1.0),
            EstimateInput("outlier", outlier, 1.0),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_estimate_rows"] == 3
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_track5_geomedian_cli_writes_outputs(tmp_path: Path) -> None:
    near_a = tmp_path / "near_a.csv"
    near_b = tmp_path / "near_b.csv"
    outlier = tmp_path / "outlier.csv"
    template = tmp_path / "template.csv"
    class_map = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _estimate_near_a().to_csv(near_a, index=False)
    _estimate_near_b().to_csv(near_b, index=False)
    _estimate_outlier().to_csv(outlier, index=False)
    _template().to_csv(template, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map,
        index=False,
    )

    status = geomedian_main(
        [
            "--estimate-csv",
            f"near_a={near_a}",
            "--estimate-csv",
            f"near_b={near_b}",
            "--estimate-csv",
            f"outlier={outlier}",
            "--template",
            str(template),
            "--class-map",
            str(class_map),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_geomedian_manifest.json").exists()


def test_track5_geomedian_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-geomedian-ensemble"]
        == "raft_uav.mmuad.track5_geometric_median_ensemble:main"
    )
