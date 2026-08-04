from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.radar import radar_polar_frame_to_candidates


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("range_m", True),
        ("range_m", np.bool_(False)),
        ("azimuth_deg", True),
        ("elevation_deg", np.bool_(False)),
        ("azimuth_rad", 1.0 + 2.0j),
        ("elevation_rad", np.array(1.0 + 2.0j)),
    ],
)
def test_radar_polar_rejects_lossy_geometry_cells(
    column: str,
    value: object,
) -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_deg": [0.0],
            "elevation_deg": [0.0],
        }
    )
    if column == "azimuth_rad":
        frame = frame.drop(columns=["azimuth_deg"])
    if column == "elevation_rad":
        frame = frame.drop(columns=["elevation_deg"])
    frame[column] = pd.Series([value], dtype=object)

    with pytest.raises(ValueError, match=rf"{column!r} contains"):
        radar_polar_frame_to_candidates(frame)


def test_radar_polar_accepts_recursively_boxed_real_geometry_cells() -> None:
    inner = np.empty((), dtype=object)
    inner[()] = 10.0
    outer = np.empty((), dtype=object)
    outer[()] = inner
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": pd.Series([outer], dtype=object),
            "azimuth_deg": [90.0],
            "elevation_deg": [0.0],
        }
    )

    candidates = radar_polar_frame_to_candidates(frame)

    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0]],
        atol=1.0e-9,
    )


def test_radar_polar_rejects_cyclic_zero_dimensional_geometry_cells() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": pd.Series([cyclic], dtype=object),
            "azimuth_deg": [0.0],
        }
    )

    with pytest.raises(ValueError, match="cyclic scalar"):
        radar_polar_frame_to_candidates(frame)


def test_radar_polar_keeps_masked_geometry_rows_filtered() -> None:
    masked = np.ma.array(5.0, mask=True)
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "range_m": pd.Series([masked, 10.0], dtype=object),
            "azimuth_deg": [0.0, 90.0],
        }
    )

    candidates = radar_polar_frame_to_candidates(frame)

    assert candidates.rows["time_s"].tolist() == [1.0]
    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0]],
        atol=1.0e-9,
    )
