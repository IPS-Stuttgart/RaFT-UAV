from __future__ import annotations

import gzip

import pytest

from raft_uav.mmuad.io import load_point_cloud_file_as_points


@pytest.mark.parametrize(
    ("suffix", "separator"),
    [
        (".tsv.gz", "\t"),
        (".txt.gz", " "),
    ],
)
def test_compressed_delimited_point_clouds_use_logical_suffix(
    tmp_path,
    suffix: str,
    separator: str,
) -> None:
    path = tmp_path / f"points{suffix}"
    header = separator.join(("sequence_id", "time_s", "x_m", "y_m", "z_m"))
    row = separator.join(("001", "1.5", "1.0", "2.0", "3.0"))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(f"{header}\n{row}\n")

    points = load_point_cloud_file_as_points(path, source="test-lidar")

    assert points[
        ["sequence_id", "time_s", "x_m", "y_m", "z_m", "source"]
    ].to_dict("records") == [
        {
            "sequence_id": "001",
            "time_s": 1.5,
            "x_m": 1.0,
            "y_m": 2.0,
            "z_m": 3.0,
            "source": "test-lidar",
        }
    ]
