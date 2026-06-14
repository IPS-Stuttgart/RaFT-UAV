from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.leaderboard import (
    build_mmuad_leaderboard,
    load_leaderboard_config,
    write_leaderboard_artifacts,
)


def _write_truth(path: Path) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "class_name": ["1", "1"],
        }
    ).to_csv(path, index=False)


def _write_results(path: Path, *, offset_m: float, uav_type: str = "1") -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "timestamp": [0.0, 1.0],
            "x": [offset_m, 1.0 + offset_m],
            "y": [0.0, 0.0],
            "z": [10.0, 10.0],
            "uav_type": [uav_type, uav_type],
            "score": [1.0, 1.0],
        }
    ).to_csv(path, index=False)


def test_mmuad_local_leaderboard_ranks_public_track5_rows(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    _write_truth(truth)
    _write_results(good, offset_m=0.0, uav_type="1")
    _write_results(bad, offset_m=5.0, uav_type="2")
    config = tmp_path / "leaderboard.json"
    config.write_text(
        json.dumps(
            {
                "default_truth": truth.name,
                "default_metric_protocol": "public-track5",
                "methods": [
                    {"method": "bad", "results_csv": bad.name},
                    {"method": "good", "results_csv": good.name},
                ],
            }
        ),
        encoding="utf-8",
    )

    entries = load_leaderboard_config(config)
    result = build_mmuad_leaderboard(entries)

    assert result.rows.loc[0, "method"] == "good"
    assert result.rows.loc[0, "pose_mse_loss_m2"] == 0.0
    assert result.rows.loc[0, "uav_type_accuracy"] == 1.0
    assert result.rows.loc[1, "pose_mse_loss_m2"] > 0.0
    assert result.rows.loc[1, "uav_type_accuracy"] == 0.0


def test_mmuad_local_leaderboard_writes_artifacts(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    result_csv = tmp_path / "results.csv"
    _write_truth(truth)
    _write_results(result_csv, offset_m=1.0, uav_type="1")
    entries = load_leaderboard_config(
        _write_config(
            tmp_path,
            {
                "default_truth": truth.name,
                "methods": [{"method": "candidate", "results_csv": result_csv.name}],
            },
        )
    )
    result = build_mmuad_leaderboard(entries)

    paths = write_leaderboard_artifacts(result, output_dir=tmp_path / "out")

    assert Path(paths["leaderboard_csv"]).exists()
    assert Path(paths["leaderboard_json"]).exists()
    markdown = Path(paths["leaderboard_md"]).read_text(encoding="utf-8")
    assert "candidate" in markdown
    assert "Codabench" in markdown


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config = tmp_path / "leaderboard.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config
