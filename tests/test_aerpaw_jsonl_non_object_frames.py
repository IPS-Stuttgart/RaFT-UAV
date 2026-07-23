"""Regression tests for strict Fortem radar JSONL frame validation."""

from __future__ import annotations

import json

import pytest

from raft_uav.io.aerpaw import read_radar_tracks_json


def _tracked_frame() -> dict[str, object]:
    return {
        "params": {"globalTime": 1759866140.0},
        "trackData": [
            {
                "id": 4,
                "lla": [35.72749, -78.69621, 30.0],
                "globalTime": 1759866140.0,
                "catProb": [0.8, 0.1],
            }
        ],
    }


@pytest.mark.parametrize(
    "payloads",
    [
        [_tracked_frame(), ["not", "an", "object"]],
        [["not", "an", "object"], _tracked_frame()],
    ],
)
def test_radar_jsonl_rejects_non_object_frames_even_when_tracks_exist(
    tmp_path,
    payloads: list[object],
) -> None:
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        "\n".join(json.dumps(payload) for payload in payloads),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain a JSON object"):
        read_radar_tracks_json(radar_path)
