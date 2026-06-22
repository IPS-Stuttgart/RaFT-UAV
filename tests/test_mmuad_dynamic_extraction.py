from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.cli import main as mmuad_cli_main
from raft_uav.mmuad.io import point_rows_to_candidates
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export


def _background_and_moving_points() -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    static_offsets = [(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.04, 0.0)]
    moving_offsets = [(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.04, 0.0)]
    for time_s in (0.0, 1.0, 2.0):
        for dx, dy, dz in static_offsets:
            records.append(
                {
                    "sequence_id": "seq001",
                    "source": "livox_avia",
                    "time_s": time_s,
                    "x_m": dx,
                    "y_m": dy,
                    "z_m": dz,
                }
            )
        for dx, dy, dz in moving_offsets:
            records.append(
                {
                    "sequence_id": "seq001",
                    "source": "livox_avia",
                    "time_s": time_s,
                    "x_m": 5.0 + time_s + dx,
                    "y_m": 1.0 + dy,
                    "z_m": 2.0 + dz,
                }
            )
    return pd.DataFrame.from_records(records)


def test_dynamic_point_extraction_removes_persistent_background() -> None:
    points = _background_and_moving_points()

    static = point_rows_to_candidates(
        points,
        voxel_size_m=0.25,
        min_points=3,
        point_extraction_mode="static",
    )
    dynamic = point_rows_to_candidates(
        points,
        voxel_size_m=0.25,
        min_points=3,
        point_extraction_mode="dynamic",
        dynamic_background_voxel_size_m=0.5,
        dynamic_background_min_frame_fraction=0.6,
        dynamic_background_min_frames=2,
    )

    assert len(static.rows) == 6
    assert len(dynamic.rows) == 3
    assert set(dynamic.rows["point_extraction_mode"]) == {"dynamic"}
    assert dynamic.rows["x_m"].min() > 4.5
    assert int(dynamic.rows["dynamic_background_removed_points"].max()) == 9
    assert int(dynamic.rows["dynamic_background_persistent_voxel_count"].max()) >= 1


def test_static_plus_dynamic_extraction_writes_union_modes() -> None:
    union = point_rows_to_candidates(
        _background_and_moving_points(),
        voxel_size_m=0.25,
        min_points=3,
        point_extraction_mode="static-plus-dynamic",
        dynamic_background_voxel_size_m=0.5,
        dynamic_background_min_frame_fraction=0.6,
        dynamic_background_min_frames=2,
    )

    assert set(union.rows["point_extraction_mode"]) == {"static", "dynamic"}
    assert len(union.rows.loc[union.rows["point_extraction_mode"] == "static"]) == 6
    assert len(union.rows.loc[union.rows["point_extraction_mode"] == "dynamic"]) == 3


def test_sequence_root_dynamic_extraction_batches_point_frames(tmp_path: Path) -> None:
    root = tmp_path / "mmuad"
    livox = root / "seq001" / "livox_avia"
    livox.mkdir(parents=True)
    points = _background_and_moving_points()
    for time_s, group in points.groupby("time_s", sort=True):
        np.save(livox / f"{time_s:.1f}.npy", group[["x_m", "y_m", "z_m"]].to_numpy(float))

    [paths] = discover_sequence_paths(root)
    candidates, _truth, _calibration = load_sequence_export(
        paths,
        voxel_size_m=0.25,
        min_cluster_points=3,
        point_extraction_mode="dynamic",
        dynamic_background_voxel_size_m=0.5,
        dynamic_background_min_frame_fraction=0.6,
        dynamic_background_min_frames=2,
    )

    assert len(candidates.rows) == 3
    assert candidates.rows["x_m"].min() > 4.5
    assert set(candidates.rows["point_extraction_mode"]) == {"dynamic"}


def test_cli_dynamic_extraction_selects_dynamic_candidates(tmp_path: Path) -> None:
    point_csv = tmp_path / "points.csv"
    _background_and_moving_points().to_csv(point_csv, index=False)
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--point-cloud-csv",
            str(point_csv),
            "--point-extraction-mode",
            "dynamic",
            "--dynamic-background-voxel-size-m",
            "0.5",
            "--dynamic-background-min-frame-fraction",
            "0.6",
            "--dynamic-background-min-frames",
            "2",
            "--voxel-size-m",
            "0.25",
            "--min-cluster-points",
            "3",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    selected = pd.read_csv(output / "mmuad_selected_tracklets.csv")
    assert not selected.empty
    assert set(selected["point_extraction_mode"]) == {"dynamic"}
    assert selected["x_m"].min() > 4.5
