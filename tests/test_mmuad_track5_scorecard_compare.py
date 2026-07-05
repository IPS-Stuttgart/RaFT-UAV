from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_scorecard_compare import (
    build_pairwise_scorecard_deltas,
    compare_track5_scorecards,
    main as compare_main,
)


def _write_scorecard(path: Path, *, name: str, mse: float, p95: float, acc: float) -> Path:
    payload = {
        "results_path": f"{name}.csv",
        "scorecard_leaderboard_ready": True,
        "codabench_upload_ready": True,
        "validation": {"leaderboard_ready": True},
        "public_track5": {
            "leaderboard_ready": True,
            "pooled": {
                "pose_mse_loss_m2": mse,
                "rmse_3d_m": mse**0.5,
                "p95_3d_m": p95,
                "max_3d_m": p95 + 3.0,
                "uav_type_accuracy": acc,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_track5_scorecards_ranks_by_pose_then_p95(tmp_path: Path) -> None:
    low_p95 = _write_scorecard(tmp_path / "low.json", name="low", mse=10.0, p95=4.0, acc=0.7)
    high_p95 = _write_scorecard(tmp_path / "high.json", name="high", mse=10.0, p95=5.0, acc=0.9)
    high_mse = _write_scorecard(tmp_path / "mse.json", name="mse", mse=12.0, p95=2.0, acc=0.8)

    comparison = compare_track5_scorecards(
        [high_mse, high_p95, low_p95],
        pose_reference_mse=11.0,
        top3_reference_mse=9.0,
        class_reference_accuracy=0.75,
    )

    assert comparison["rank"].tolist() == [1, 2, 3]
    assert comparison.loc[0, "scorecard_label"] == "low"
    assert comparison.loc[0, "beats_pose_reference"] is True
    assert comparison.loc[0, "beats_top3_pose_reference"] is False
    assert comparison.loc[1, "beats_class_reference"] is True
    assert comparison.loc[2, "pose_mse_delta_to_best"] == 2.0


def test_pairwise_scorecard_deltas_use_best_as_baseline(tmp_path: Path) -> None:
    best = _write_scorecard(tmp_path / "best.json", name="best", mse=4.0, p95=2.0, acc=0.8)
    other = _write_scorecard(tmp_path / "other.json", name="other", mse=9.0, p95=3.0, acc=0.7)
    comparison = compare_track5_scorecards([other, best])

    deltas = build_pairwise_scorecard_deltas(comparison)

    assert len(deltas) == 2
    assert deltas.loc[0, "pose_mse_delta"] == 0.0
    assert deltas.loc[1, "pose_mse_delta"] == 5.0
    assert deltas.loc[1, "rmse_delta_m"] == 1.0
    assert deltas.loc[1, "class_accuracy_delta"] == -0.1


def test_scorecard_compare_cli_writes_outputs(tmp_path: Path) -> None:
    scorecard_a = _write_scorecard(tmp_path / "a.json", name="a", mse=16.0, p95=6.0, acc=0.5)
    scorecard_b = _write_scorecard(tmp_path / "b.json", name="b", mse=9.0, p95=4.0, acc=0.6)
    output_csv = tmp_path / "comparison.csv"
    pairwise_csv = tmp_path / "pairwise.csv"
    summary_json = tmp_path / "summary.json"

    status = compare_main(
        [
            str(scorecard_a),
            str(scorecard_b),
            "--output-csv",
            str(output_csv),
            "--pairwise-delta-csv",
            str(pairwise_csv),
            "--summary-json",
            str(summary_json),
            "--pose-reference-mse",
            "12",
        ]
    )

    assert status == 0
    comparison = pd.read_csv(output_csv)
    pairwise = pd.read_csv(pairwise_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert comparison.loc[0, "pose_mse_loss_m2"] == 9.0
    assert pairwise.loc[1, "pose_mse_delta"] == 7.0
    assert summary["scorecard_count"] == 2
    assert summary["best_pose_mse_loss_m2"] == 9.0
