from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.sequence_alignment_audit import (
    build_sequence_alignment_audit,
    main as sequence_alignment_audit_main,
)


def _write_point_frame(path: Path, rows: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["x_m", "y_m", "z_m"]).to_csv(path, index=False)


def _write_truth(path: Path) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq0002", "seq0002", "seq0002"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [10.0, 11.0, 12.0],
            "y_m": [2.0, 2.0, 2.0],
            "z_m": [3.0, 3.0, 3.0],
        }
    ).to_csv(path, index=False)


def _toy_sequence_root(tmp_path: Path) -> Path:
    root = tmp_path / "mmuad"
    seq = root / "seq0002" / "lidar_360"
    _write_point_frame(seq / "0.0.csv", [(10.0, 2.0, 3.0), (10.1, 2.0, 3.0)])
    _write_point_frame(seq / "1.0.csv", [])
    _write_point_frame(seq / "2.0.csv", [(40.0, 2.0, 3.0)])
    return root


def test_sequence_alignment_audit_reports_extraction_and_alignment_stats(
    tmp_path: Path,
) -> None:
    root = _toy_sequence_root(tmp_path)
    truth = tmp_path / "truth.csv"
    _write_truth(truth)

    audit = build_sequence_alignment_audit(
        root,
        truth,
        sequence_glob="seq0002",
        voxel_size_m=0.5,
        min_cluster_points=1,
        max_time_delta_s=0.1,
        include_translation_diagnostic=False,
    )

    as_is = audit.loc[
        (audit["sensor"] == "lidar_360") & (audit["variant"] == "as-is")
    ].iloc[0]
    assert as_is["sequence_id"] == "seq0002"
    assert int(as_is["source_frame_count"]) == 3
    assert int(as_is["loaded_source_frame_count"]) == 3
    assert int(as_is["empty_frame_count"]) == 1
    assert int(as_is["no_candidate_source_frame_count"]) == 1
    assert int(as_is["candidate_frame_count"]) == 2
    assert int(as_is["candidate_count"]) == 2
    assert float(as_is["source_time_matched_truth_frame_fraction"]) == 1.0
    assert float(as_is["matched_truth_frame_fraction"]) == 2 / 3
    assert float(as_is["fraction_frames_with_cluster_within_5m"]) == 1 / 3
    assert float(as_is["cluster_point_count_max"]) == 2.0
    assert float(as_is["cluster_range_3d_m_p95"]) > 30.0


def test_sequence_alignment_median_translation_diagnostic_is_available(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mmuad"
    seq = root / "seq0002" / "lidar_360"
    _write_point_frame(seq / "0.0.csv", [(110.0, 2.0, 3.0)])
    _write_point_frame(seq / "1.0.csv", [(111.0, 2.0, 3.0)])
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0002", "seq0002"],
            "time_s": [0.0, 1.0],
            "x_m": [10.0, 11.0],
            "y_m": [2.0, 2.0],
            "z_m": [3.0, 3.0],
        }
    ).to_csv(truth, index=False)

    audit = build_sequence_alignment_audit(
        root,
        truth,
        sequence_glob="seq0002",
        voxel_size_m=0.5,
        min_cluster_points=1,
        max_time_delta_s=0.1,
    )

    as_is = audit.loc[audit["variant"] == "as-is"].iloc[0]
    translated = audit.loc[audit["variant"] == "as-is+median-translation"].iloc[0]
    assert float(as_is["mean_nearest_cluster_to_truth_distance_m"]) > 99.0
    assert float(translated["mean_nearest_cluster_to_truth_distance_m"]) < 0.1
    assert translated["translation_mode"] == "per-sequence-median-diagnostic"


def test_sequence_alignment_audit_cli_writes_requested_csv(tmp_path: Path) -> None:
    root = _toy_sequence_root(tmp_path)
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    _write_truth(truth)

    status = sequence_alignment_audit_main(
        [
            str(root),
            "--truth-file",
            str(truth),
            "--output-dir",
            str(output),
            "--sequence-glob",
            "seq0002",
            "--voxel-size-m",
            "0.5",
            "--min-cluster-points",
            "1",
            "--max-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    path = output / "mmuad_sequence_alignment_audit.csv"
    assert path.exists()
    rows = pd.read_csv(path)
    assert {
        "sequence_id",
        "source_frame_count",
        "empty_frame_count",
        "mean_nearest_source_frame_abs_time_delta_s",
        "p95_nearest_cluster_to_truth_distance_m",
        "fraction_frames_with_cluster_within_20m",
    }.issubset(rows.columns)
