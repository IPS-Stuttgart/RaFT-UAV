import json

from raft_uav.io.aerpaw import read_radar_tracks_json


def _radar_frame(track_id: int | None) -> dict[str, object]:
    tracks = []
    if track_id is not None:
        tracks.append(
            {
                "id": track_id,
                "lla": [35.72749, -78.69621, 30.0],
                "globalTime": 1759866140.0 + track_id,
                "catProb": [0.8, 0.1],
            }
        )
    return {"params": {"globalTime": 1759866140.0}, "trackData": tracks}


def test_radar_jsonl_blank_lines_do_not_create_frame_gaps(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        "\n".join(
            [
                json.dumps(_radar_frame(1)),
                "",
                "   ",
                json.dumps(_radar_frame(2)),
            ]
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert radar["frame_index"].tolist() == [0, 1]


def test_radar_jsonl_empty_object_frame_still_advances_frame_index(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        "\n".join(
            [
                json.dumps(_radar_frame(None)),
                "",
                json.dumps(_radar_frame(3)),
            ]
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert radar["frame_index"].tolist() == [1]
