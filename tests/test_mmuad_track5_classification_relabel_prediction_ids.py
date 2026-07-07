from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_classification_relabel import main as relabel_main


def test_track5_classification_prediction_cli_preserves_zero_padded_sequences(
    tmp_path: Path,
) -> None:
    pose_csv = tmp_path / "pose.csv"
    predictions_csv = tmp_path / "sequence_predictions.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "Sequence": ["001", "001", "002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,1)", "(1,0,1)", "(5,0,2)"],
            "Classification": [0, 0, 3],
        }
    ).to_csv(pose_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001", "001", "002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)", "(0,0,0)", "(0,0,0)"],
            "Classification": [1, 1, 2],
        }
    ).to_csv(template_csv, index=False)
    predictions_csv.write_text(
        "heldout_sequence,predicted_probability_0,predicted_probability_1,"
        "predicted_probability_2,predicted_probability_3\n"
        "001,0.05,0.80,0.10,0.05\n"
        "002,0.10,0.15,0.70,0.05\n",
        encoding="utf-8",
    )

    status = relabel_main(
        [
            "--pose-submission",
            str(pose_csv),
            "--classification-predictions",
            str(predictions_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    results = pd.read_csv(output_dir / "mmaud_results_relabelled.csv", dtype={"Sequence": str})
    assert results["Sequence"].tolist() == ["001", "001", "002"]
    assert results["Classification"].tolist() == [1, 1, 2]
