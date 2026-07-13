from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.stress.perturbations import PerturbationConfig, perturb_radar, perturb_rf


def _radar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "frame_index": [0, 1],
            "track_id": [1, 1],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "velocity_east_mps": [1.0, 1.0],
            "cat_prob_uav": [0.9, 0.9],
        }
    )


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"radar_drop_rate": np.nan}, "radar_drop_rate"),
        ({"timestamp_jitter_std_s": np.inf}, "timestamp_jitter_std_s"),
        ({"velocity_noise_std_mps": np.nan}, "velocity_noise_std_mps"),
        ({"false_tracks_per_frame": 1.5}, "false_tracks_per_frame"),
        (
            {"false_tracks_per_frame": 1, "false_track_position_std_m": np.nan},
            "false_track_position_std_m",
        ),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_radar_stress_perturbation_rejects_invalid_numeric_controls(
    config_kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        perturb_radar(_radar_frame(), PerturbationConfig(name="invalid", **config_kwargs))


def test_rf_stress_perturbation_rejects_nonfinite_drop_rate() -> None:
    rf = pd.DataFrame({"time_s": [0.0, 1.0], "east_m": [0.0, 1.0], "north_m": [0.0, 0.0]})

    with pytest.raises(ValueError, match="rf_drop_burst_rate"):
        perturb_rf(rf, PerturbationConfig(name="invalid", rf_drop_burst_rate=np.nan))


def test_valid_stress_controls_preserve_existing_behavior() -> None:
    out = perturb_radar(
        _radar_frame(),
        PerturbationConfig(name="valid", false_tracks_per_frame=1, seed=7),
    )

    assert len(out) == 4
    assert out["stress_false_track"].sum() == 2
