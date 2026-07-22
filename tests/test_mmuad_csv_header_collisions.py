from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

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
