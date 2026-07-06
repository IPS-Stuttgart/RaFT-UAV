from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad.evaluate import load_submission_csv


def test_load_submission_csv_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    path.write_text(
        "\n".join(
            [
                "Sequence,time_s,x_m,y_m,z_m,track_id",
                "001,0.0,1.0,2.0,3.0,uav-a",
                "010,1.0,4.0,5.0,6.0,uav-b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_submission_csv(path)

    assert rows["sequence_id"].tolist() == ["001", "010"]
    assert rows["track_id"].tolist() == ["uav-a", "uav-b"]
    assert rows["time_s"].tolist() == [0.0, 1.0]
