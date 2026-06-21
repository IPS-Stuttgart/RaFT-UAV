from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.radar_extraction_audit import (
    build_radar_extraction_audit,
    main as radar_extraction_audit_main,
)


def _write_sequence(root: Path) -> Path:
    radar = root / "seq0002" / "radar_enhance_pcl"
    radar.mkdir(parents=True)
    np.save(radar / "0.0.npy", np.empty((0, 3), dtype=float))
    np.save(
        radar / "1.0.npy",
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
    np.save(
        radar / "2.0.npy",
        np.asarray(
            [
                [2.0, 0.0, 0.0],
                [2.1, 0.0, 0.0],
                [2.2, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
    return root


def test_radar_extraction_audit_distinguishes_empty_sparse_and_clustered_frames(
    tmp_path: Path,
) -> None:
    root = _write_sequence(tmp_path / "mmuad")

    audit = build_radar_extraction_audit(root, sequence_glob="seq0002", voxel_size_m=0.5)

    assert len(audit) == 3
    empty = audit.loc[audit["timestamp"] == 0.0].iloc[0]
    sparse = audit.loc[audit["timestamp"] == 1.0].iloc[0]
    clustered = audit.loc[audit["timestamp"] == 2.0].iloc[0]
    assert empty["raw_shape"] == "0x3"
    assert int(empty["raw_point_count"]) == 0
    assert empty["reason_no_candidates"] == "raw_empty"
    assert int(sparse["raw_point_count"]) == 2
    assert int(sparse["finite_xyz_count"]) == 2
    assert int(sparse["cluster_count_min1"]) == 2
    assert int(sparse["cluster_count_min3"]) == 0
    assert sparse["reason_no_candidates"] == "clusters_below_min3"
    assert int(clustered["cluster_count_min3"]) == 1
    assert clustered["reason_no_candidates"] == "candidates_present_min3"
    assert float(clustered["range_median"]) == 2.1
    assert clustered["min_xyz"] == "[2.0,0.0,0.0]"


def test_radar_extraction_audit_cli_writes_requested_csv(tmp_path: Path) -> None:
    root = _write_sequence(tmp_path / "mmuad")
    output = tmp_path / "out"

    status = radar_extraction_audit_main(
        [
            str(root),
            "--output-dir",
            str(output),
            "--sequence-glob",
            "seq0002",
            "--voxel-size-m",
            "0.5",
        ]
    )

    assert status == 0
    path = output / "mmuad_radar_extraction_audit.csv"
    assert path.exists()
    rows = pd.read_csv(path)
    assert {
        "sequence",
        "timestamp",
        "raw_shape",
        "raw_point_count",
        "finite_xyz_count",
        "cluster_count_min1",
        "cluster_count_min3",
        "cluster_count_min5",
        "reason_no_candidates",
    }.issubset(rows.columns)
