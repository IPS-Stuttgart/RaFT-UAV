from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    build_track5_consensus_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_consensus_ensemble import main as consensus_main
from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    write_track5_consensus_ensemble_outputs,
)
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 10.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    )


def _estimate(label_shift: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 10.0],
            "state_x_m": [label_shift, 10.0 + label_shift],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [1.0, 1.0],
        }
    )


def test_consensus_ensemble_rejects_divergent_outlier_branch() -> None:
    estimates, diagnostics = build_track5_consensus_estimate_ensemble(
        [
            ("near_a", _estimate(0.0), 1.0),
            ("near_b", _estimate(0.4), 1.0),
            ("outlier", _estimate(40.0), 1.0),
        ],
        _template(),
        consensus_radius_m=2.0,
    )

    first = estimates.iloc[0]
    assert first["state_x_m"] == pytest.approx(0.2)
    assert first["consensus_input_count"] == 2
    assert first["consensus_labels"] == "near_a;near_b"
    assert diagnostics.iloc[0]["rejected_labels"] == "outlier"


def test_consensus_ensemble_can_fallback_when_cluster_weight_is_too_small() -> None:
    estimates, diagnostics = build_track5_consensus_estimate_ensemble(
        [
            ("small_cluster_a", _estimate(0.0), 1.0),
            ("small_cluster_b", _estimate(0.3), 1.0),
            ("heavy", _estimate(30.0), 5.0),
        ],
        _template(),
        consensus_radius_m=2.0,
        min_consensus_weight_fraction=0.8,
        fallback_policy="max-weight",
    )

    first = estimates.iloc[0]
    assert first["state_x_m"] == pytest.approx(30.0)
    assert bool(first["consensus_fallback_applied"]) is True
    assert diagnostics.iloc[0]["selection_reason"] == "fallback_low_consensus_weight"


def test_consensus_ensemble_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    near_a = tmp_path / "near_a.csv"
    near_b = tmp_path / "near_b.csv"
    outlier = tmp_path / "outlier.csv"
    class_map = tmp_path / "class_map.csv"
    _estimate(0.0).to_csv(near_a, index=False)
    _estimate(0.4).to_csv(near_b, index=False)
    _estimate(40.0).to_csv(outlier, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(class_map, index=False)

    paths = write_track5_consensus_ensemble_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"near_a={near_a}"),
            parse_estimate_spec(f"near_b={near_b}"),
            parse_estimate_spec(f"outlier={outlier}"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2"},
        consensus_radius_m=2.0,
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_rows"] == 2
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2]
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_consensus_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    near_a = tmp_path / "near_a.csv"
    near_b = tmp_path / "near_b.csv"
    outlier = tmp_path / "outlier.csv"
    template = tmp_path / "template.csv"
    class_map = tmp_path / "class_map.csv"
    output = tmp_path / "out"
    _estimate(0.0).to_csv(near_a, index=False)
    _estimate(0.4).to_csv(near_b, index=False)
    _estimate(40.0).to_csv(outlier, index=False)
    _template().to_csv(template, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(class_map, index=False)

    status = consensus_main(
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
            str(output),
            "--consensus-radius-m",
            "2.0",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output / "ug2_submission.zip").exists()
    assert (output / "mmuad_track5_consensus_ensemble_manifest.json").exists()


def test_consensus_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-consensus-ensemble"]
        == "raft_uav.mmuad.track5_estimate_consensus_ensemble:main"
    )
