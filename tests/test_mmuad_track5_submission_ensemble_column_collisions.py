from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


@pytest.mark.parametrize(
    ("header", "normalized_name"),
    [
        (
            "sequence_id,time_s, TIME_S ,state_x_m,state_y_m,state_z_m,Classification",
            "time_s",
        ),
        (
            "sequence_id,time_s,state_x_m,state_x_m,state_y_m,state_z_m,Classification",
            "state_x_m",
        ),
    ],
    ids=("normalized", "exact"),
)
def test_submission_ensemble_rejects_ambiguous_physical_headers(
    tmp_path: Path,
    header: str,
    normalized_name: str,
) -> None:
    path = tmp_path / "submission.csv"
    path.write_text(
        f"{header}\nseq0001,0,1,2,3,4,0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"ambiguous Track 5 submission columns.*{normalized_name}",
    ):
        load_track5_submission(path)


def test_submission_ensemble_accepts_unique_padded_normalized_headers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission.csv"
    path.write_text(
        " sequence_id , time_s , state_x_m ,state_y_m,state_z_m, Classification \n"
        "seq0001,0,1,2,3,0\n",
        encoding="utf-8",
    )

    rows = load_track5_submission(path)

    assert rows["sequence_id"].tolist() == ["seq0001"]
    assert rows["time_s"].tolist() == [0.0]
    assert rows[["state_x_m", "state_y_m", "state_z_m"]].to_numpy().tolist() == [
        [1.0, 2.0, 3.0]
    ]
    assert rows["Classification"].tolist() == [0]
