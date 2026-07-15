from __future__ import annotations

import numpy as np
import pytest

import raft_uav.mmuad.io as mmuad_io


LARGE_INTEGER = 2**53 + 1


@pytest.mark.parametrize(
    "value",
    [
        LARGE_INTEGER,
        str(LARGE_INTEGER),
        np.int64(LARGE_INTEGER),
        np.array(LARGE_INTEGER, dtype=np.int64),
    ],
)
def test_dynamic_background_preserves_exact_large_integer_controls(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    captured: dict[str, object] = {}

    def fake_dynamic_point_residuals(
        points: object,
        *,
        voxel_size_m: float,
        min_frame_fraction: float,
        min_frames: int,
        neighbor_radius_voxels: int,
    ) -> tuple[object, dict[str, object]]:
        captured.update(
            {
                "voxel_size_m": voxel_size_m,
                "min_frame_fraction": min_frame_fraction,
                "min_frames": min_frames,
                "neighbor_radius_voxels": neighbor_radius_voxels,
            }
        )
        return points, {}

    monkeypatch.setattr(
        mmuad_io,
        "_ORIGINAL_DYNAMIC_POINT_RESIDUALS",
        fake_dynamic_point_residuals,
    )
    points = object()

    residuals, stats = mmuad_io._dynamic_point_residuals(
        points,
        voxel_size_m=1.0,
        min_frame_fraction=0.5,
        min_frames=value,
        neighbor_radius_voxels=value,
    )

    assert residuals is points
    assert stats == {}
    assert captured["min_frames"] == LARGE_INTEGER
    assert captured["neighbor_radius_voxels"] == LARGE_INTEGER
