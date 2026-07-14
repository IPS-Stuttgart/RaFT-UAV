from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.io import _dynamic_point_residuals, point_rows_to_candidates


EMPTY_POINTS = pd.DataFrame(
    columns=["sequence_id", "source", "time_s", "x_m", "y_m", "z_m"]
)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        1.5,
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        pd.NA,
        np.array([1]),
        "not-an-integer",
    ],
)
def test_dynamic_background_rejects_invalid_min_frames(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="--dynamic-background-min-frames must be a positive integer",
    ):
        _dynamic_point_residuals(
            EMPTY_POINTS,
            voxel_size_m=1.0,
            min_frame_fraction=0.5,
            min_frames=value,  # type: ignore[arg-type]
            neighbor_radius_voxels=0,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0.5,
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        pd.NA,
        np.array([0]),
        "not-an-integer",
    ],
)
def test_dynamic_background_rejects_invalid_neighbor_radius(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "--dynamic-background-neighbor-radius-voxels must be a "
            "non-negative integer"
        ),
    ):
        _dynamic_point_residuals(
            EMPTY_POINTS,
            voxel_size_m=1.0,
            min_frame_fraction=0.5,
            min_frames=1,
            neighbor_radius_voxels=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("min_frames", "neighbor_radius_voxels"),
    [
        (1, 0),
        (1.0, 0.0),
        (np.int64(2), np.float64(1.0)),
        (np.array(3), np.array(2)),
        ("4", "3"),
    ],
)
def test_dynamic_background_accepts_integer_equivalent_scalars(
    min_frames: object,
    neighbor_radius_voxels: object,
) -> None:
    residuals, stats = _dynamic_point_residuals(
        EMPTY_POINTS,
        voxel_size_m=1.0,
        min_frame_fraction=0.5,
        min_frames=min_frames,  # type: ignore[arg-type]
        neighbor_radius_voxels=neighbor_radius_voxels,  # type: ignore[arg-type]
    )

    assert residuals.empty
    assert stats == {}


def test_point_candidate_public_path_uses_validated_dynamic_controls() -> None:
    points = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "source": ["lidar"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "--dynamic-background-neighbor-radius-voxels must be a "
            "non-negative integer"
        ),
    ):
        point_rows_to_candidates(
            points,
            point_extraction_mode="dynamic",
            dynamic_background_neighbor_radius_voxels=-1,
        )
