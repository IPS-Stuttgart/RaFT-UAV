from __future__ import annotations

import gzip

import pytest

from raft_uav.mmuad.io import load_point_cloud_file_as_points


def test_gzipped_tsv_point_cloud_uses_tab_delimiter(tmp_path) -> None:
    path = tmp_path / "points.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            "sequence_id\ttime_s\tx_m\ty_m\tz_m\n"
            "sequence-a\t1.5\t10.0\t20.0\t30.0\n"
        )

    points = load_point_cloud_file_as_points(path, source="lidar")

    assert len(points) == 1
    row = points.iloc[0]
    assert row["sequence_id"] == "sequence-a"
    assert float(row["time_s"]) == pytest.approx(1.5)
    assert float(row["x_m"]) == pytest.approx(10.0)
    assert float(row["y_m"]) == pytest.approx(20.0)
    assert float(row["z_m"]) == pytest.approx(30.0)
    assert row["source"] == "lidar"
