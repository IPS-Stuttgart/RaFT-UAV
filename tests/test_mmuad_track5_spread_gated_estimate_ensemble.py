from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_spread_gated_estimate_ensemble import (
    build_spread_gated_estimate_ensemble,
)
from raft_uav.mmuad.track5_spread_gated_estimate_ensemble import main as spread_gate_main
from raft_uav.mmuad.track5_spread_gated_estimate_ensemble import (
    write_spread_gated_estimate_ensemble_outputs,
)
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 10.0, 0.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [2, 2, 2, 1],
        }
    )


def _anchor_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 5.0],
            "state_y_m": [0.0, 0.0, 5.0],
            "state_z_m": [0.0, 0.0, 5.0],
        }
    )


def _outlier_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [100.0, 110.0, 105.0],
            "state_y_m": [0.0, 0.0, 105.0],
            "state_z_m": [0.0, 0.0, 105.0],
        }
    )


def test_spread_gated_ensemble_uses_anchor_when_disagreement_is_large() -> None:
    estimates, diagnostics = build_spread_gated_estimate_ensemble(
        [
            ("anchor", _anchor_estimate(), 0.5),
            ("outlier", _outlier_estimate(), 0.5),
        ],
        _template(),
        anchor_label="anchor",
        spread_threshold_m=20.0,
    )

    midpoint = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["spread_gated"] is True or bool(midpoint["spread_gated"])
    assert midpoint["state_x_m"] == pytest.approx(5.0)
    assert midpoint["state_y_m"] == pytest.approx(0.0)
    assert diagnostics["spread_gated"].sum() == 4


def test_spread_gated_ensemble_keeps_base_ensemble_when_spread_is_small() -> None:
    estimates, _ = build_spread_gated_estimate_ensemble(
        [
            ("anchor", _anchor_estimate(), 0.5),
            ("outlier", _outlier_estimate(), 0.5),
        ],
        _template(),
        anchor_label="anchor",
        spread_threshold_m=200.0,
    )

    midpoint = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 5.0)
    ].iloc[0]
    assert bool(midpoint["spread_gated"]) is False
    assert midpoint["state_x_m"] == pytest.approx(55.0)


def test_spread_gated_ensemble_writes_upload_ready_outputs(tmp_path: Path) -> None:
    anchor_csv = tmp_path / "anchor.csv"
    outlier_csv = tmp_path / "outlier.csv"
    _anchor_estimate().to_csv(anchor_csv, index=False)
    _outlier_estimate().to_csv(outlier_csv, index=False)
    paths = write_spread_gated_estimate_ensemble_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"anchor={anchor_csv}@0.5"),
            parse_estimate_spec(f"outlier={outlier_csv}@0.5"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        anchor_label="anchor",
        spread_threshold_m=20.0,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["spread_gated_rows"] == 4
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2, 1]


def test_spread_gated_ensemble_cli_and_entrypoint(tmp_path: Path) -> None:
    anchor_csv = tmp_path / "anchor.csv"
    outlier_csv = tmp_path / "outlier.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _anchor_estimate().to_csv(anchor_csv, index=False)
    _outlier_estimate().to_csv(outlier_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = spread_gate_main(
        [
            "--estimate-csv",
            f"anchor={anchor_csv}@0.5",
            "--estimate-csv",
            f"outlier={outlier_csv}@0.5",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--anchor-label",
            "anchor",
            "--spread-threshold-m",
            "20",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_spread_gated_manifest.json").exists()
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-spread-gated-estimate-ensemble"]
        == "raft_uav.mmuad.track5_spread_gated_estimate_ensemble:main"
    )
