import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.radar_fingerprint import (
    _continuous_track_segments,
    _optional_int,
    _segment_range_source,
    _segment_ranges,
)


@pytest.mark.parametrize("invalid_frame_index", [np.nan, np.inf])
def test_partial_frame_indices_fall_back_to_timestamp_continuity(invalid_frame_index):
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "frame_index": [0.0, invalid_frame_index, 2.0],
            "time_s": [0.0, 1.0, 10.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert [segment["time_s"].tolist() for segment in segments] == [
        [0.0, 1.0],
        [10.0],
    ]
    assert all(
        float(segment["time_s"].iloc[-1]) >= float(segment["time_s"].iloc[0])
        for segment in segments
    )


def test_integer_timestamp_cadence_without_frame_indices_stays_continuous():
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "time_s": [0.0, 2.0, 4.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert len(segments) == 1
    assert segments[0]["time_s"].tolist() == [0.0, 2.0, 4.0]


def test_numeric_string_timestamps_sort_numerically():
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "time_s": ["10", "2", "6"],
        }
    )

    segments = _continuous_track_segments(radar)

    assert len(segments) == 1
    assert segments[0]["time_s"].tolist() == [2.0, 6.0, 10.0]


@pytest.mark.parametrize("invalid_timestamp", [np.nan, np.inf, -np.inf, "not-a-time"])
def test_invalid_timestamps_do_not_pollute_radar_segments(invalid_timestamp):
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7, 7],
            "frame_index": [0.0, 1.0, 2.0, 3.0],
            "time_s": [0.0, invalid_timestamp, 2.0, 3.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert [segment["time_s"].tolist() for segment in segments] == [
        [0.0],
        [2.0, 3.0],
    ]
    assert all(
        np.isfinite(segment["time_s"].to_numpy(dtype=float)).all()
        for segment in segments
    )


def test_complete_frame_indices_remain_authoritative_for_continuity():
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "frame_index": [0.0, 1.0, 2.0],
            "time_s": [0.0, 100.0, 200.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert len(segments) == 1
    assert segments[0]["time_s"].tolist() == [0.0, 100.0, 200.0]


def test_frame_index_gaps_stay_strict_when_timestamps_are_integer_valued():
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "frame_index": [0.0, 2.0, 4.0],
            "time_s": [0.0, 2.0, 4.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert [segment["time_s"].tolist() for segment in segments] == [
        [0.0],
        [2.0],
        [4.0],
    ]


def test_frame_counter_reset_starts_new_track_segment():
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7, 7],
            "frame_index": [0.0, 1.0, 0.0, 1.0],
            "time_s": [0.0, 1.0, 100.0, 101.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert [segment["time_s"].tolist() for segment in segments] == [
        [0.0, 1.0],
        [100.0, 101.0],
    ]


def test_radar_fingerprint_preserves_large_integer_metadata_exactly():
    assert _optional_int("9007199254740993") == 9007199254740993


@pytest.mark.parametrize("value", [7.5, "7.5", True, np.array([7])])
def test_radar_fingerprint_rejects_non_integer_metadata(value):
    assert _optional_int(value) is None


def test_range_source_matches_enu_fallback_when_sensor_ranges_are_unusable():
    segment = pd.DataFrame(
        {
            "range_m": [np.nan, "not-a-range"],
            "east_m": [3.0, 5.0],
            "north_m": [4.0, 12.0],
            "up_m": [0.0, 0.0],
        }
    )

    np.testing.assert_allclose(_segment_ranges(segment), np.array([5.0, 13.0]))
    assert _segment_range_source(segment) == "enu_norm"
