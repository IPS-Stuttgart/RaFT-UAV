from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble_text_cli import main as ensemble_main


def test_track5_estimate_ensemble_cli_keeps_zero_padded_sequence_ids(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(estimate_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    ).to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["001"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = ensemble_main(
        [
            "--estimate-csv",
            f"model={estimate_csv}",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(
        output_dir / "mmuad_track5_ensemble_estimates.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert estimates.loc[0, "sequence_id"] == "001"
    assert float(estimates.loc[0, "state_x_m"]) == pytest.approx(1.0)
    assert float(estimates.loc[0, "state_y_m"]) == pytest.approx(2.0)
    assert float(estimates.loc[0, "state_z_m"]) == pytest.approx(3.0)
    assert int(float(estimates.loc[0, "ensemble_source_count"])) == 1

    validation = json.loads((output_dir / "mmuad_track5_ensemble_validation.json").read_text())
    assert validation["leaderboard_ready"] is True


def test_estimate_ensemble_console_script_uses_text_id_wrapper() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-ensemble"]
        == "raft_uav.mmuad.track5_estimate_ensemble_text_cli:main"
    )
