from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from raft_uav.mmuad import estimate_csv
from raft_uav.mmuad.io import load_candidate_csv, load_truth_csv


@pytest.mark.parametrize(
    ("loader", "csv_text", "collided_header"),
    [
        (
            load_truth_csv,
            "time_s, TIME_S,x_m,y_m,z_m\n0,1,2,3,4\n",
            "time_s",
        ),
        (
            load_candidate_csv,
            "time_s,x_m, X_M,y_m,z_m\n0,1,2,3,4,5\n",
            "x_m",
        ),
    ],
    ids=("truth", "candidate"),
)
def test_csv_loaders_reject_headers_that_collide_after_normalization(
    tmp_path: Path,
    loader: Callable[[Path], object],
    csv_text: str,
    collided_header: str,
) -> None:
    path = tmp_path / "rows.csv"
    path.write_text(csv_text, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"CSV headers are ambiguous.*{collided_header}",
    ):
        loader(path)


@pytest.mark.parametrize(
    ("csv_text", "collided_header"),
    [
        (
            "sequence_id, SEQUENCE_ID,time_s,state_x_m,state_y_m,state_z_m\n"
            "001,002,0,1,2,3\n",
            "sequence_id",
        ),
        (
            "sequence_id,time_s,state_x_m, state_x_m,state_y_m,state_z_m\n"
            "001,0,1,9,2,3\n",
            "state_x_m",
        ),
    ],
    ids=("case", "whitespace"),
)
def test_estimate_csv_reader_rejects_ambiguous_normalized_headers(
    tmp_path: Path,
    csv_text: str,
    collided_header: str,
) -> None:
    path = tmp_path / "estimate.csv"
    path.write_text(csv_text, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"estimate CSV headers are ambiguous.*{collided_header}",
    ):
        estimate_csv.read_estimate_csv(path)


def test_guarded_estimate_csv_reader_rejects_ambiguous_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "estimate.csv"
    path.write_text(
        "sequence_id,time_s,state_x_m, STATE_X_M,state_y_m,state_z_m\n"
        "001,0,1,9,2,3\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(estimate_csv, "_called_from_track5_estimate_grid", lambda: True)
    monkeypatch.setattr(
        estimate_csv,
        "_called_from_candidate_reservoir_cli",
        lambda: False,
    )

    with pytest.raises(ValueError, match="estimate CSV headers are ambiguous"):
        estimate_csv._read_csv_with_track5_estimate_grid_guard(path)
