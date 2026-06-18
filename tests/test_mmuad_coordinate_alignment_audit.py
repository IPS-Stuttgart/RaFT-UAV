from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.coordinate_alignment_audit import (
    build_coordinate_alignment_audit,
    main as coordinate_audit_main,
)


def _write_point_frame(path: Path, rows: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["x_m", "y_m", "z_m"]).to_csv(path, index=False)


def _write_truth(path: Path, *, offset_x: float = 0.0) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [10.0 + offset_x, 11.0 + offset_x],
            "y_m": [2.0, 2.0],
            "z_m": [3.0, 3.0],
        }
    ).to_csv(path, index=False)


def _toy_sequence_root(tmp_path: Path, *, sensor_offset_x: float = 0.0) -> Path:
    root = tmp_path / "mmuad"
    seq = root / "seq001" / "lidar_360"
    _write_point_frame(
        seq / "0.0.csv",
        [
            (10.0 + sensor_offset_x, 2.0, 3.0),
            (10.1 + sensor_offset_x, 2.0, 3.0),
        ],
    )
    _write_point_frame(
        seq / "1.0.csv",
        [
            (11.0 + sensor_offset_x, 2.0, 3.0),
            (11.1 + sensor_offset_x, 2.0, 3.0),
        ],
    )
    return root


def test_coordinate_alignment_audit_reports_as_is_close_to_truth(tmp_path: Path) -> None:
    root = _toy_sequence_root(tmp_path)
    truth = tmp_path / "truth.csv"
    _write_truth(truth)

    audit = build_coordinate_alignment_audit(
        root,
        truth,
        voxel_size_m=0.5,
        min_cluster_points=1,
        max_time_delta_s=0.1,
        include_translation_diagnostic=False,
    )

    as_is = audit.loc[
        (audit["sensor"] == "lidar_360") & (audit["variant"] == "as-is")
    ].iloc[0]
    swapped = audit.loc[
        (audit["sensor"] == "lidar_360") & (audit["variant"] == "x-y-swap")
    ].iloc[0]
    assert as_is["axis_permutation"] == "x,y,z"
    assert as_is["axis_sign"] == "+,+,+"
    assert float(as_is["fraction_frames_with_cluster_within_5m"]) == 1.0
    assert float(as_is["mean_nearest_cluster_to_truth_distance_m"]) < 0.2
    assert float(swapped["mean_nearest_cluster_to_truth_distance_m"]) > 10.0


def test_coordinate_alignment_median_translation_diagnostic_improves_offset_cloud(
    tmp_path: Path,
) -> None:
    root = _toy_sequence_root(tmp_path, sensor_offset_x=100.0)
    truth = tmp_path / "truth.csv"
    _write_truth(truth)

    audit = build_coordinate_alignment_audit(
        root,
        truth,
        voxel_size_m=0.5,
        min_cluster_points=1,
        max_time_delta_s=0.1,
    )

    as_is = audit.loc[audit["variant"] == "as-is"].iloc[0]
    translated = audit.loc[audit["variant"] == "as-is+median-translation"].iloc[0]
    assert float(as_is["mean_nearest_cluster_to_truth_distance_m"]) > 99.0
    assert float(translated["mean_nearest_cluster_to_truth_distance_m"]) < 0.2
    assert translated["translation_mode"] == "per-sequence-median-diagnostic"
    assert -101.0 < float(translated["translation_x_m"]) < -99.0


def test_coordinate_alignment_cli_writes_requested_csv(tmp_path: Path) -> None:
    root = _toy_sequence_root(tmp_path)
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    _write_truth(truth)

    status = coordinate_audit_main(
        [
            str(root),
            "--truth-file",
            str(truth),
            "--output-dir",
            str(output),
            "--voxel-size-m",
            "0.5",
            "--min-cluster-points",
            "1",
            "--max-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    path = output / "mmuad_coordinate_alignment_audit.csv"
    assert path.exists()
    rows = pd.read_csv(path)
    assert {
        "sensor",
        "axis_permutation",
        "axis_sign",
        "scale",
        "translation_mode",
        "p95_nearest_cluster_to_truth_distance_m",
        "fraction_frames_with_cluster_within_20m",
    }.issubset(rows.columns)
