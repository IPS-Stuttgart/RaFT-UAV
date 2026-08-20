from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.mmuad.radar_extraction_audit import build_radar_extraction_audit


def _write_npz_sequence(root: Path) -> Path:
    radar = root / "seq0002" / "radar_enhance_pcl"
    radar.mkdir(parents=True)
    points = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [2.2, 0.0, 0.0],
        ],
        dtype=float,
    )
    np.savez(
        radar / "2.0.npz",
        metadata=np.asarray([123.0]),
        Points=points,
    )
    return root


def test_radar_audit_uses_candidate_loader_npz_point_array_selection(
    tmp_path: Path,
) -> None:
    root = _write_npz_sequence(tmp_path / "mmuad")

    audit = build_radar_extraction_audit(
        root,
        sequence_glob="seq0002",
        voxel_size_m=0.5,
    )

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["raw_shape"] == "3x3"
    assert int(row["raw_point_count"]) == 3
    assert int(row["finite_xyz_count"]) == 3
    assert int(row["cluster_count_min3"]) == 1
    assert row["reason_no_candidates"] == "candidates_present_min3"


@pytest.mark.parametrize(
    "voxel_size_m",
    [
        None,
        0.0,
        -0.5,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.asarray([0.5]),
        np.ma.masked,
    ],
)
def test_radar_audit_rejects_invalid_voxel_sizes_before_discovery(
    tmp_path: Path,
    voxel_size_m: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="voxel_size_m must be a positive finite scalar",
    ):
        build_radar_extraction_audit(tmp_path, voxel_size_m=voxel_size_m)


def test_radar_audit_accepts_zero_dimensional_real_voxel_size(
    tmp_path: Path,
) -> None:
    audit = build_radar_extraction_audit(
        tmp_path,
        voxel_size_m=np.asarray(0.5),
    )

    assert audit.empty
