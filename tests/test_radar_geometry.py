from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.radar_geometry import (
    build_radar_geometry_audit_frame,
    polar_offset_enu,
    summarize_radar_geometry_audit,
)


def test_polar_offset_enu_north_clockwise_axes() -> None:
    offsets = polar_offset_enu(
        [10.0, 10.0, 10.0],
        [0.0, 90.0, 0.0],
        [0.0, 0.0, 90.0],
        azimuth_convention="north-clockwise",
    )

    np.testing.assert_allclose(offsets[0], [0.0, 10.0, 0.0], atol=1.0e-9)
    np.testing.assert_allclose(offsets[1], [10.0, 0.0, 0.0], atol=1.0e-9)
    np.testing.assert_allclose(offsets[2], [0.0, 0.0, 10.0], atol=1.0e-9)


def test_geometry_audit_zero_delta_for_consistent_polar_and_lla() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [7, 7],
            "east_m": [0.0, 10.0],
            "north_m": [10.0, 0.0],
            "up_m": [0.0, 0.0],
            "range_m": [10.0, 10.0],
            "azimuth_deg": [0.0, 90.0],
            "elevation_deg": [0.0, 0.0],
        }
    )

    audit = build_radar_geometry_audit_frame(radar)

    np.testing.assert_allclose(audit["geometry_delta_3d_m"].to_numpy(), [0.0, 0.0], atol=1.0e-9)
    summary = summarize_radar_geometry_audit(audit)
    assert summary["rows"] == 2
    assert summary["track_ids"] == 1
    assert np.isclose(summary["geometry_delta_3d_m"]["max"], 0.0)


def test_geometry_summary_preserves_exact_integer_identifiers() -> None:
    large = 2**80
    audit = pd.DataFrame(
        {
            "track_id": [str(large), str(large + 1), "7.5", "8.5", "invalid", None],
            "frame_index": [str(large + 2), str(large + 3), 4.5, 5.5, np.inf, True],
        }
    )

    summary = summarize_radar_geometry_audit(audit)

    assert summary["rows"] == 6
    assert summary["track_ids"] == 2
    assert summary["frames"] == 2
