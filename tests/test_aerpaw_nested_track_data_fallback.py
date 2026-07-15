from __future__ import annotations

import json

import pytest

from raft_uav.io.aerpaw import read_radar_tracks_json


@pytest.mark.parametrize("top_level_track_data", [None, "not-a-list"])
def test_radar_reader_falls_back_to_valid_nested_track_data(
    tmp_path,
    top_level_track_data,
) -> None:
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "trackData": top_level_track_data,
                "params": {
                    "globalTime": 123.0,
                    "trackData": [
                        {
                            "id": 7,
                            "lla": [35.72749, -78.69621, 30.0],
                            "catProb": [0.9, 0.1],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert len(radar) == 1
    assert radar.loc[0, "track_id"] == 7
    assert radar.loc[0, "global_time_raw_s"] == 123.0
    assert radar.loc[0, "cat_prob_uav"] == 0.9
