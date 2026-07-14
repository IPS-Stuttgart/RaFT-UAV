import json

import pytest

from raft_uav.io.aerpaw import read_radar_json, read_radar_tracks_json
from raft_uav.paper_selection import select_paper_strict_raw_radar_track


def _radar_frame(track_id: int | None, global_time_s: float) -> dict:
    track_data = []
    if track_id is not None:
        track_data.append(
            {
                "id": track_id,
                "lla": [35.72749, -78.69621, 30.0],
                "globalTime": global_time_s,
                "catProb": [0.8, 0.1],
            }
        )
    return {
        "params": {"globalTime": global_time_s},
        "trackData": track_data,
    }


def test_blank_jsonl_lines_do_not_split_continuous_radar_track(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        "\n".join(
            [
                json.dumps(_radar_frame(7, 100.0)),
                "",
                "   ",
                json.dumps(_radar_frame(7, 101.0)),
            ]
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)
    radar["time_s"] = [0.0, 1.0]
    selected = select_paper_strict_raw_radar_track(radar)

    assert radar["frame_index"].tolist() == [0, 1]
    assert read_radar_json(radar_path)["frame_index"].tolist() == [0, 1]
    assert selected["frame_index"].tolist() == [0, 1]


def test_empty_radar_frame_still_advances_logical_frame_index(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        "\n".join(
            [
                json.dumps(_radar_frame(7, 100.0)),
                json.dumps(_radar_frame(None, 101.0)),
                json.dumps(_radar_frame(7, 102.0)),
            ]
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert radar["frame_index"].tolist() == [0, 2]


def test_radar_json_parse_error_uses_physical_line_number(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text("\n{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2"):
        read_radar_tracks_json(radar_path)
