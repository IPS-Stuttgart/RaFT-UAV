from __future__ import annotations

from raft_uav.mmuad.io import (
    load_candidate_file,
    load_point_cloud_file_as_points,
    load_truth_file,
)


def test_csv_trajectory_loaders_preserve_text_sequence_ids_and_strip_headers(tmp_path) -> None:
    truth_csv = tmp_path / "truth.csv"
    truth_csv.write_text(
        " sequence_id , time_s , x_m , y_m , z_m \n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    candidate_csv = tmp_path / "candidates.csv"
    candidate_csv.write_text(
        " sequence_id , time_s , source , x_m , y_m , z_m , confidence \n"
        "001,0.0,radar,4.0,5.0,6.0,0.75\n",
        encoding="utf-8",
    )
    points_csv = tmp_path / "points.csv"
    points_csv.write_text(
        " sequence_id , time_s , x_m , y_m , z_m , source \n"
        "001,0.0,7.0,8.0,9.0,lidar\n",
        encoding="utf-8",
    )

    truth = load_truth_file(truth_csv).rows
    candidates = load_candidate_file(candidate_csv, source="fallback").rows
    points = load_point_cloud_file_as_points(points_csv, source="fallback")

    assert truth["sequence_id"].tolist() == ["001"]
    assert candidates["sequence_id"].tolist() == ["001"]
    assert points["sequence_id"].tolist() == ["001"]
    assert candidates["source"].tolist() == ["radar"]
    assert points["source"].tolist() == ["lidar"]
    assert candidates["confidence"].tolist() == [0.75]
