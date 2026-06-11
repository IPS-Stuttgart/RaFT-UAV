import json

import numpy as np
import pandas as pd
import pytest

from raft_uav.coordinates import LocalENUProjector
from raft_uav.io.aerpaw import (
    discover_flights,
    flight_file_manifest,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    radar_measurements_to_enu,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_radar_measurement_rows,
)


def test_enu_origin_maps_to_near_zero():
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)

    enu = projector.transform(35.7274895, -78.696216, 2.717)

    np.testing.assert_allclose(enu, np.zeros(3), atol=1e-6)


def test_timestamp_normalization_is_monotonic_and_truth_relative(tmp_path):
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        "\n".join(
            [
                '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:19.000000,4,27',
                '2,-78.6962159,35.7274895,2.727,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
            ]
        ),
        encoding="utf-8",
    )
    rf_path = tmp_path / "rf.csv"
    rf_path.write_text(
        "\n".join(
            [
                "Meas ID,Algorithm,Center Frequency,Sample Rate,Num Samples,Latitude,Longitude,Elevation,RHO,CEP,Total Sensors,Valid Sensors,Overloaded Sensors,Sensor Names,Time,Comment",
                "1,tdoa,1,1,1,35.72749,-78.69621,0,0,75,4,4,0,s,2025-10-07 19:42:20.000,",
            ]
        ),
        encoding="utf-8",
    )

    truth, projector, origin_time = normalize_truth(read_truth(truth_path))
    rf = normalize_rf(read_rf_csv(rf_path), projector, origin_time)

    assert truth["time_s"].tolist() == [0.0, 1.0]
    assert truth["time_s"].is_monotonic_increasing
    np.testing.assert_allclose(rf["time_s"].to_numpy(), np.array([1.0]), atol=1e-6)


def test_truth_normalization_accepts_second_precision_timestamps(tmp_path):
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        "\n".join(
            [
                '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:19,4,27',
                '2,-78.6962159,35.7274895,2.727,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20,4,27',
            ]
        ),
        encoding="utf-8",
    )

    truth, _projector, origin_time = normalize_truth(read_truth(truth_path))

    assert truth["time_s"].tolist() == [0.0, 1.0]
    assert origin_time == pd.Timestamp("2025-10-07 15:42:19")
    np.testing.assert_allclose(
        truth[["east_m", "north_m", "up_m"]].iloc[0].to_numpy(dtype=float),
        np.zeros(3),
        atol=1e-6,
    )


def test_rf_and_radar_clock_offsets_are_independent(tmp_path):
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
        encoding="utf-8",
    )
    rf_path = tmp_path / "rf.csv"
    rf_path.write_text(
        "\n".join(
            [
                "Meas ID,Algorithm,Center Frequency,Sample Rate,Num Samples,Latitude,Longitude,Elevation,RHO,CEP,Total Sensors,Valid Sensors,Overloaded Sensors,Sensor Names,Time,Comment",
                "1,tdoa,1,1,1,35.72749,-78.69621,0,0,75,4,4,0,s,2025-10-07 19:42:25.000,",
            ]
        ),
        encoding="utf-8",
    )
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "params": {"globalTime": 1759851745.0},
                "trackData": [
                    {
                        "id": 1,
                        "lla": [35.72749, -78.69621, 30.0],
                        "globalTime": 1759851745.0,
                        "catProb": [0.8, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _, projector, origin_time = normalize_truth(read_truth(truth_path))
    rf = normalize_rf(
        read_rf_csv(rf_path),
        projector,
        origin_time,
        clock_offset_s=-4.0 * 60.0 * 60.0,
    )
    radar = normalize_radar(
        read_radar_tracks_json(radar_path),
        projector,
        origin_time,
        clock_offset_s=0.0,
    )
    radar_with_rf_offset = normalize_radar(
        read_radar_tracks_json(radar_path),
        projector,
        origin_time,
        clock_offset_s=-4.0 * 60.0 * 60.0,
    )

    np.testing.assert_allclose(rf["time_s"].to_numpy(), np.array([5.0]), atol=1e-6)
    np.testing.assert_allclose(radar["time_s"].to_numpy(), np.array([5.0]), atol=1e-6)
    np.testing.assert_allclose(
        radar_with_rf_offset["time_s"].to_numpy(), np.array([-14395.0]), atol=1e-6
    )


def test_radar_jsonl_reader_rejects_non_object_frames(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        "\n".join(
            [
                json.dumps({"trackData": []}),
                json.dumps(["not", "an", "object"]),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain a JSON object"):
        read_radar_tracks_json(radar_path)


def test_radar_jsonl_reader_exposes_fortem_quality_fields(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "params": {"globalTime": 1759866140.0},
                "trackData": [
                    {
                        "id": 4,
                        "lla": [35.72749, -78.69621, 30.0],
                        "globalTime": 1759866140.0,
                        "range": 123.0,
                        "azimuth": 14.5,
                        "elevation": -2.0,
                        "rcsDbsm": -9.25,
                        "radialVelocity": 3.5,
                        "numInliers": 17,
                        "trackAge": 8,
                        "catProb": [0.8, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)
    row = radar.iloc[0]

    assert row["azimuth_deg"] == 14.5
    assert row["elevation_deg"] == -2.0
    assert row["rcs_dbsm"] == -9.25
    assert row["radial_velocity_mps"] == 3.5
    assert row["num_inliers"] == 17
    assert row["track_age"] == 8


def test_radar_jsonl_reader_falls_back_when_quality_alias_is_null(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "params": {"globalTime": 1759866140.0},
                "trackData": [
                    {
                        "id": 4,
                        "lla": [35.72749, -78.69621, 30.0],
                        "globalTime": 1759866140.0,
                        "confidence": None,
                        "confidenceScore": 0.9,
                        "trackAge": None,
                        "age": 8,
                        "status": None,
                        "trackStatus": "confirmed",
                        "azimuth": None,
                        "azimuthDeg": 14.5,
                        "elevation": None,
                        "elevationDeg": -2.0,
                        "rcsDbsm": None,
                        "radarCrossSection": -9.25,
                        "catProb": [0.8, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = read_radar_tracks_json(radar_path).iloc[0]

    assert row["confidence"] == 0.9
    assert row["track_age"] == 8
    assert row["track_status"] == "confirmed"
    assert row["azimuth_deg"] == 14.5
    assert row["elevation_deg"] == -2.0
    assert row["rcs_dbsm"] == -9.25


def test_radar_jsonl_reader_falls_back_to_frame_time_when_track_time_is_null(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "params": {
                    "globalTime": 1759866140.0,
                    "gpsWeek": 2388,
                    "gpsSeconds": 12345.0,
                },
                "trackData": [
                    {
                        "id": 4,
                        "lla": [35.72749, -78.69621, 30.0],
                        "globalTime": None,
                        "gpsWeek": None,
                        "gpsSeconds": None,
                        "catProb": [0.8, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert radar.loc[0, "global_time_raw_s"] == 1759866140.0
    assert radar.loc[0, "gps_week"] == 2388
    assert radar.loc[0, "gps_seconds"] == 12345.0


def test_radar_jsonl_reader_accepts_params_nested_track_data(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "params": {
                    "globalTime": 1759866140.0,
                    "trackData": [
                        {
                            "id": 4,
                            "lla": [35.72749, -78.69621, 30.0],
                            "globalTime": None,
                            "catProb": [0.8, 0.1],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert len(radar) == 1
    assert radar.loc[0, "track_id"] == 4
    assert radar.loc[0, "global_time_raw_s"] == 1759866140.0
    assert radar.loc[0, "cat_prob_uav"] == 0.8


def test_radar_jsonl_reader_preserves_explicit_zero_confidence(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(
        json.dumps(
            {
                "params": {"globalTime": 1759866140.0},
                "trackData": [
                    {
                        "id": 4,
                        "lla": [35.72749, -78.69621, 30.0],
                        "globalTime": 1759866140.0,
                        "confidence": 0.0,
                        "confidenceScore": 0.9,
                        "catProb": [0.8, 0.1],
                    },
                    {
                        "id": 5,
                        "lla": [35.72749, -78.69621, 31.0],
                        "globalTime": 1759866140.0,
                        "confidenceScore": 0.7,
                        "catProb": [0.7, 0.2],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert radar["confidence"].tolist() == [0.0, 0.7]


def test_discover_flights_can_select_original_or_rerun_variant(tmp_path):
    flight_dir = tmp_path / "RF Sensor and Radar" / "Opt1"
    flight_dir.mkdir(parents=True)
    (flight_dir / "AADM.csv").write_text("x\n", encoding="utf-8")
    (flight_dir / "AADM_rerun.csv").write_text("y\n", encoding="utf-8")
    (flight_dir / "radar_data.json").write_text("{}\n", encoding="utf-8")
    (flight_dir / "radar_data_rerun.json").write_text("{}\n", encoding="utf-8")
    (flight_dir / "date_time_vehicleOut.txt").write_text("truth\n", encoding="utf-8")
    (flight_dir / "date_time_vehicleOut_rerun.txt").write_text("truth\n", encoding="utf-8")

    auto = discover_flights(tmp_path)[0]
    original = discover_flights(tmp_path, variant="original")[0]
    rerun = discover_flights(tmp_path, variant="rerun")[0]

    assert auto.rf_csv is not None and "rerun" in auto.rf_csv.name
    assert original.rf_csv is not None and "rerun" not in original.rf_csv.name
    assert rerun.rf_csv is not None and "rerun" in rerun.rf_csv.name
    assert flight_file_manifest(rerun, dataset_root=tmp_path)["rf"]["variant"] == "rerun"


def test_radar_jsonl_reader_rejects_non_object_payload(tmp_path):
    radar_path = tmp_path / "radar.json"
    radar_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        read_radar_tracks_json(radar_path)


def test_radar_jsonl_reader_and_catprob_selection(tmp_path):
    radar_path = tmp_path / "radar.json"
    frames = [
        {"error": {"code": 200, "message": ""}},
        {
            "params": {"globalTime": 1759866140.0},
            "trackData": [
                {
                    "id": 1,
                    "lla": [35.72749, -78.69621, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.6, 0.1],
                },
                {
                    "id": 2,
                    "lla": [35.727, -78.696, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.1, 0.1],
                },
                {
                    "id": 3,
                    "lla": [35.7271, -78.6961, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.55, 0.1],
                },
            ],
        },
    ]
    radar_path.write_text("\n".join(json.dumps(frame) for frame in frames), encoding="utf-8")
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
        encoding="utf-8",
    )

    truth, projector, origin_time = normalize_truth(read_truth(truth_path))
    radar = normalize_radar(read_radar_tracks_json(radar_path), projector, origin_time)
    selected = select_radar_measurement_rows(radar, selection="catprob", catprob_threshold=0.5)
    selected_all = select_radar_measurement_rows(
        radar,
        selection="catprob-all",
        catprob_threshold=0.5,
    )

    assert len(radar) == 3
    assert selected["track_id"].tolist() == [1]
    assert selected_all["track_id"].tolist() == [1, 3]
    assert len(truth) == 1


def test_radar_measurement_converter_ignores_null_velocity_components():
    radar = pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "velocity_east_mps": [None],
            "velocity_north_mps": [None],
            "velocity_down_mps": [None],
        }
    )

    [measurement] = radar_measurements_to_enu(
        radar,
        include_velocity=True,
    )

    assert measurement.vector.shape == (3,)
    np.testing.assert_allclose(measurement.vector, np.array([10.0, 20.0, 30.0]))


def test_raw_radar_measurement_converter_accepts_rows_without_track_id():
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)
    origin_time = pd.Timestamp("2025-10-07 15:42:20")
    radar = pd.DataFrame(
        {
            "global_time_raw_s": [1759851741.0],
            "latitude": [35.72749],
            "longitude": [-78.69621],
            "altitude_m": [30.0],
        }
    )

    [measurement] = radar_measurements_to_enu(
        radar,
        projector=projector,
        truth_origin_time=origin_time,
        clock_offset_s=0.0,
    )

    assert measurement.time_s == pytest.approx(1.0)
    assert measurement.vector.shape == (3,)


def test_truth_gated_radar_selection_handles_unsorted_truth_times():
    radar = pd.DataFrame(
        {
            "track_id": [1, 2],
            "time_s": [0.05, 1.05],
            "east_m": [10.0, 20.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [1.0, 0.0],
            "east_m": [20.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    selected = select_radar_measurement_rows(
        radar,
        selection="truth-gated",
        truth=truth,
        truth_gate_m=1.0,
        truth_time_gate_s=0.2,
    )

    assert selected["track_id"].tolist() == [1, 2]


def test_radar_jsonl_reader_skips_malformed_records(tmp_path):
    radar_path = tmp_path / "radar.json"
    frames = [
        [],
        {"params": ["not", "a", "mapping"], "trackData": "not-a-list"},
        {
            "params": {"globalTime": 123.0},
            "trackData": [
                None,
                "not-a-track",
                {
                    "id": 7,
                    "lla": [35.72749, -78.69621, 30.0],
                    "catProb": [0.9, 0.1],
                },
            ],
        },
    ]
    radar_path.write_text(
        "\n".join(json.dumps(frame) for frame in frames),
        encoding="utf-8",
    )

    radar = read_radar_tracks_json(radar_path)

    assert len(radar) == 1
    row = radar.iloc[0]
    assert row["track_id"] == 7
    assert row["global_time_raw_s"] == 123.0
    assert row["cat_prob_uav"] == 0.9
