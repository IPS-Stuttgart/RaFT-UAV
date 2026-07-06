from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import (
    build_spread_guarded_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import main as spread_guard_main
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import write_spread_guard_outputs
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
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
            "state_x_m": [50.0, 51.0, 54.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def test_spread_guard_falls_back_when_estimates_disagree() -> None:
    estimates, diagnostics = build_spread_guarded_estimate_ensemble(
        [
            ("trusted", _trusted_estimate(), 0.6),
            ("outlier", _outlier_estimate(), 0.4),
        ],
        _template(),
        spread_threshold_m=5.0,
        fallback_policy="max-weight",
    )

    first = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 0.0)
    ].iloc[0]
    assert first["spread_guard_applied"] is True or bool(first["spread_guard_applied"])
    assert first["spread_guard_chosen_label"] == "trusted"
    assert first["state_x_m"] == pytest.approx(0.0)
    assert diagnostics["spread_guard_applied"].all()


def test_spread_guard_uses_weighted_mean_below_threshold() -> None:
    estimates, _ = build_spread_guarded_estimate_ensemble(
        [
            ("trusted", _trusted_estimate(), 0.6),
            ("outlier", _outlier_estimate(), 0.4),
        ],
        _template(),
        spread_threshold_m=100.0,
        fallback_policy="max-weight",
    )

    first = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 0.0)
    ].iloc[0]
    assert not bool(first["spread_guard_applied"])
    assert first["spread_guard_chosen_label"] == "weighted-mean"
    assert first["state_x_m"] == pytest.approx(20.0)


def test_spread_guard_fallback_blend_keeps_controlled_weighted_mean_fraction() -> None:
    estimates, diagnostics = build_spread_guarded_estimate_ensemble(
        [
            ("trusted", _trusted_estimate(), 0.6),
            ("outlier", _outlier_estimate(), 0.4),
        ],
        _template(),
        spread_threshold_m=5.0,
        fallback_policy="max-weight",
        fallback_blend=0.25,
    )

    first = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 0.0)
    ].iloc[0]
    first_diag = diagnostics.loc[
        (diagnostics["sequence_id"] == "seq0001") & (diagnostics["time_s"] == 0.0)
    ].iloc[0]
    # Weighted mean is 20 m and fallback is 0 m, so a 25% blend outputs 5 m.
    assert bool(first["spread_guard_applied"])
    assert first["state_x_m"] == pytest.approx(5.0)
    assert first["spread_guard_fallback_blend"] == pytest.approx(0.25)
    assert first_diag["fallback_x_m"] == pytest.approx(0.0)
    assert first_diag["weighted_x_m"] == pytest.approx(20.0)


def test_spread_guard_rejects_invalid_fallback_blend() -> None:
    with pytest.raises(ValueError, match="fallback_blend"):
        build_spread_guarded_estimate_ensemble(
            [("trusted", _trusted_estimate(), 1.0)],
            _template(),
            spread_threshold_m=5.0,
            fallback_blend=1.5,
        )


def test_spread_guard_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    trusted_csv = tmp_path / "trusted.csv"
    outlier_csv = tmp_path / "outlier.csv"
    _trusted_estimate().to_csv(trusted_csv, index=False)
    _outlier_estimate().to_csv(outlier_csv, index=False)
    paths = write_spread_guard_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"trusted={trusted_csv}@0.6"),
            parse_estimate_spec(f"outlier={outlier_csv}@0.4"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        spread_threshold_m=5.0,
        fallback_blend=0.25,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["guard_applied_rows"] == 3
    assert manifest["fallback_blend"] == pytest.approx(0.25)
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_spread_guard_cli_writes_outputs(tmp_path: Path) -> None:
    trusted_csv = tmp_path / "trusted.csv"
    outlier_csv = tmp_path / "outlier.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _trusted_estimate().to_csv(trusted_csv, index=False)
    _outlier_estimate().to_csv(outlier_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = spread_guard_main(
        [
            "--estimate-csv",
            f"trusted={trusted_csv}@0.6",
            "--estimate-csv",
            f"outlier={outlier_csv}@0.4",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--spread-threshold-m",
            "5",
            "--fallback-blend",
            "0.25",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_spread_guard_manifest.json").exists()
    diagnostics = pd.read_csv(output_dir / "mmuad_track5_spread_guard_diagnostics.csv")
    assert diagnostics["spread_guard_fallback_blend"].eq(0.25).all()


def test_spread_guard_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-spread-guard-ensemble"]
        == "raft_uav.mmuad.track5_estimate_ensemble_spread_guard:main"
    )
