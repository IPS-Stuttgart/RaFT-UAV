from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.radar_geometry import (
    build_radar_geometry_audit_frame,
    summarize_radar_geometry_audit,
)


def test_radar_geometry_keeps_large_representable_distances_finite() -> None:
    radar = pd.DataFrame(
        {
            "east_m": [6.0e307, 4.8e307],
            "north_m": [8.0e307, 6.4e307],
            "up_m": [0.0, 0.0],
            "range_m": [0.0, 0.0],
            "azimuth_deg": [0.0, 0.0],
            "elevation_deg": [0.0, 0.0],
        }
    )

    audit = build_radar_geometry_audit_frame(radar)
    expected = np.array([1.0e308, 8.0e307])

    for column in (
        "geometry_delta_horizontal_m",
        "geometry_delta_3d_m",
        "lla_slant_range_from_radar_origin_m",
    ):
        actual = audit[column].to_numpy(dtype=float)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12)

    np.testing.assert_allclose(
        audit["range_minus_lla_slant_range_m"].to_numpy(dtype=float),
        -expected,
        rtol=1.0e-12,
    )

    summary = summarize_radar_geometry_audit(audit)
    distance_stats = summary["geometry_delta_3d_m"]
    assert distance_stats["count"] == 2
    for key in ("mean", "std", "p50", "p95", "max"):
        assert np.isfinite(float(distance_stats[key]))
    np.testing.assert_allclose(float(distance_stats["mean"]), 9.0e307, rtol=1.0e-12)
    np.testing.assert_allclose(
        float(distance_stats["std"]),
        np.sqrt(2.0) * 1.0e307,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(float(distance_stats["p50"]), 9.0e307, rtol=1.0e-12)
    np.testing.assert_allclose(float(distance_stats["p95"]), 9.9e307, rtol=1.0e-12)
    np.testing.assert_allclose(float(distance_stats["max"]), 1.0e308, rtol=1.0e-12)

    range_stats = summary["range_minus_lla_slant_range_m"]
    assert range_stats["count"] == 2
    np.testing.assert_allclose(float(range_stats["mean"]), -9.0e307, rtol=1.0e-12)
    np.testing.assert_allclose(float(range_stats["max"]), -8.0e307, rtol=1.0e-12)
