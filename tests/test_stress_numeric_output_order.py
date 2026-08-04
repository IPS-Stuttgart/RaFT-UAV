from __future__ import annotations

import pandas as pd

from raft_uav.stress.perturbations import (
    PerturbationConfig,
    perturb_radar,
    perturb_rf,
)


def test_perturb_radar_sorts_numeric_string_keys_numerically() -> None:
    radar = pd.DataFrame(
        {
            "time_s": ["10", "2", "2"],
            "frame_index": ["10", "2", "2"],
            "track_id": ["1", "10", "2"],
            "marker": ["late", "second-track-ten", "second-track-two"],
        }
    )

    output = perturb_radar(radar, PerturbationConfig(name="serialized"))

    assert output["marker"].tolist() == [
        "second-track-two",
        "second-track-ten",
        "late",
    ]
    assert output["time_s"].tolist() == ["2", "2", "10"]
    assert output["frame_index"].tolist() == ["2", "2", "10"]
    assert output["track_id"].tolist() == ["2", "10", "1"]


def test_perturb_rf_sorts_numeric_string_timestamps_numerically() -> None:
    rf = pd.DataFrame(
        {
            "time_s": ["10", "2", "1"],
            "marker": ["late", "middle", "early"],
        }
    )

    output = perturb_rf(rf, PerturbationConfig(name="serialized"))

    assert output["marker"].tolist() == ["early", "middle", "late"]
    assert output["time_s"].tolist() == ["1", "2", "10"]


def test_perturb_radar_preserves_opaque_track_id_ordering() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [1.0, 1.0],
            "frame_index": [1, 1],
            "track_id": ["track-b", "track-a"],
        }
    )

    output = perturb_radar(radar, PerturbationConfig(name="opaque"))

    assert output["track_id"].tolist() == ["track-a", "track-b"]
