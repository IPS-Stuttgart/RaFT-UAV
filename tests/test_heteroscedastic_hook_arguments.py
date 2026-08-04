from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.heteroscedastic_cli import heteroscedastic_covariance_hooks


def _write_empty_model(path: Path) -> Path:
    path.write_text(
        '{"schema_version": 1, "metadata": {}, "heads": []}',
        encoding="utf-8",
    )
    return path


def test_hooked_rf_conversion_preserves_positional_default_std(tmp_path: Path) -> None:
    model_path = _write_empty_model(tmp_path / "uncertainty_model.json")
    frame = pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
        }
    )

    with heteroscedastic_covariance_hooks(model_path):
        from raft_uav import cli as legacy_cli

        [measurement] = legacy_cli.rf_measurements_to_enu(
            frame,
            None,
            None,
            4.0,
        )

    np.testing.assert_allclose(measurement.covariance, np.diag([16.0, 16.0]))


def test_hooked_radar_conversion_preserves_positional_velocity_options(
    tmp_path: Path,
) -> None:
    model_path = _write_empty_model(tmp_path / "uncertainty_model.json")
    frame = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "velocity_east_mps": [1.0],
            "velocity_north_mps": [2.0],
            "velocity_down_mps": [3.0],
        }
    )

    with heteroscedastic_covariance_hooks(model_path):
        from raft_uav import cli as legacy_cli

        [measurement] = legacy_cli.radar_measurements_to_enu(
            frame,
            None,
            None,
            2.0,
            3.0,
            4.0,
            True,
        )

    np.testing.assert_allclose(measurement.vector, [10.0, 20.0, 30.0, 1.0, 2.0, -3.0])
    np.testing.assert_allclose(
        measurement.covariance,
        np.diag([4.0, 4.0, 9.0, 16.0, 16.0, 16.0]),
    )
