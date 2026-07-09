from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard_compare import (
    _bool_or_none,
    compare_pose_by_sequence_tables,
    compare_track5_scorecards,
    main as compare_main,
)


def _scorecard(path: Path, *, name: str, mse: float, p95: float, acc: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "results_path": f"{name}.csv",
                "scorecard_leaderboard_ready": True,
                "codabench_upload_ready": True,
                "validation": {"leaderboard_ready": True},
                "public_track5": {
                    "pooled": {
                        "pose_mse_loss_m2": mse,
                        "rmse_3d_m": mse**0.5,
                        "p95_3d_m": p95,
                        "max_3d_m": p95 + 1.0,
                        "uav_type_accuracy": acc,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _pose_by_sequence(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": sequence_id,
                "count": count,
                "mse": mse,
                "rmse": mse**0.5,
                "mean_3d": mse**0.5,
                "median_3d": mse**0.5,
                "p95_3d": mse**0.5,
                "max_3d": mse**0.5,
            }
            for sequence_id, count, mse in rows
        ]
    )


def test_scorecard_comparison_ranks_by_pose_then_p95(tmp_path: Path) -> None:
    a = _scorecard(tmp_path / "a.json", name="a", mse=10.0, p95=5.0, acc=0.9)
    b = _scorecard(tmp_path / "b.json", name="b", mse=10.0, p95=4.0, acc=0.7)
    c = _scorecard(tmp_path / "c.json", name="c", mse=12.0, p95=2.0, acc=0.8)

    table = compare_track5_scorecards([a, c, b], pose_reference_mse=11.0)

    assert table["scorecard_label"].tolist() == ["b", "a", "c"]
    assert table["rank"].tolist() == [1, 2, 3]
    assert bool(table.loc[0, "beats_pose_reference"])
    assert table.loc[2, "pose_mse_delta_to_best"] == 2.0


def test_scorecard_compare_cli_writes_outputs(tmp_path: Path) -> None:
    a = _scorecard(tmp_path / "a.json", name="a", mse=16.0, p95=6.0, acc=0.5)
    b = _scorecard(tmp_path / "b.json", name="b", mse=9.0, p95=4.0, acc=0.6)
    output_csv = tmp_path / "comparison.csv"
    summary_json = tmp_path / "summary.json"

    status = compare_main(
        [
            str(a),
            str(b),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert status == 0
    table = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert table.loc[0, "scorecard_label"] == "b"
    assert table.loc[0, "pose_mse_loss_m2"] == 9.0
    assert summary["scorecard_count"] == 2
    assert summary["best_label"] == "b"


def test_pose_by_sequence_comparison_reports_regressions() -> None:
    baseline = _pose_by_sequence([("seq001", 10, 100.0), ("seq002", 5, 25.0)])
    candidate = _pose_by_sequence([("seq001", 10, 80.0), ("seq002", 5, 30.0)])

    delta, summary = compare_pose_by_sequence_tables(
        baseline,
        candidate,
        regression_tolerance_mse=1.0,
    )

    assert delta.loc[delta["sequence_id"] == "seq001", "delta_mse"].iloc[0] == -20.0
    assert delta.loc[delta["sequence_id"] == "seq002", "delta_mse"].iloc[0] == 5.0
    assert summary["common_sequence_count"] == 2
    assert summary["improved_sequence_count"] == 1
    assert summary["regressed_sequence_count"] == 1
    assert summary["weighted_delta_mse"] == pytest.approx((-20.0 * 10 + 5.0 * 5) / 15)
    assert summary["worst_regression_sequence"] == "seq002"


def test_scorecard_compare_cli_writes_pose_by_sequence_delta(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline_pose.csv"
    candidate_csv = tmp_path / "candidate_pose.csv"
    delta_csv = tmp_path / "pose_delta.csv"
    delta_json = tmp_path / "pose_delta.json"
    _pose_by_sequence([("seq001", 10, 100.0), ("seq002", 5, 25.0)]).to_csv(
        baseline_csv,
        index=False,
    )
    _pose_by_sequence([("seq001", 10, 80.0), ("seq002", 5, 30.0)]).to_csv(
        candidate_csv,
        index=False,
    )

    status = compare_main(
        [
            "--baseline-pose-by-sequence-csv",
            str(baseline_csv),
            "--candidate-pose-by-sequence-csv",
            str(candidate_csv),
            "--pose-delta-csv",
            str(delta_csv),
            "--pose-delta-summary-json",
            str(delta_json),
            "--regression-tolerance-mse",
            "1",
        ]
    )

    assert status == 0
    delta = pd.read_csv(delta_csv)
    summary = json.loads(delta_json.read_text(encoding="utf-8"))
    assert delta["sequence_id"].tolist() == ["seq001", "seq002"]
    assert summary["weighted_delta_mse"] == pytest.approx((-20.0 * 10 + 5.0 * 5) / 15)
    assert summary["regressed_sequence_count"] == 1


def test_scorecard_compare_bool_flags_accept_numeric_export_encodings() -> None:
    assert _bool_or_none(1.0) is True
    assert _bool_or_none("1.0") is True
    assert _bool_or_none(2) is True
    assert _bool_or_none(0.0) is False
    assert _bool_or_none("0.0") is False
    assert _bool_or_none("false") is False
    assert _bool_or_none("not-a-boolean") is None
