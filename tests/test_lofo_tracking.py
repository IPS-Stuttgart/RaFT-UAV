from raft_uav.calibration.lofo_tracking import records_to_frame


EXPECTED_ESTIMATE_COLUMNS = [
    "time_s",
    "source",
    "east_m",
    "north_m",
    "up_m",
    "v_east_mps",
    "v_north_mps",
    "v_up_mps",
]


def test_records_to_frame_preserves_schema_for_empty_records() -> None:
    frame = records_to_frame([])

    assert frame.empty
    assert frame.columns.tolist() == EXPECTED_ESTIMATE_COLUMNS


def test_records_to_frame_sorts_nonempty_records_by_time() -> None:
    frame = records_to_frame(
        [
            {"time_s": 2.0, "source": "radar", "state": [2, 3, 4, 5, 6, 7]},
            {"time_s": 1.0, "source": "rf", "state": [1, 2, 3, 4, 5, 6]},
        ]
    )

    assert frame.columns.tolist() == EXPECTED_ESTIMATE_COLUMNS
    assert frame["time_s"].tolist() == [1.0, 2.0]
    assert frame["source"].tolist() == ["rf", "radar"]


def test_records_to_frame_preserves_order_within_duplicate_timestamps() -> None:
    records = []
    for index in range(24):
        time_s = (2.0, 1.0, 3.0, 1.0)[index % 4]
        records.append(
            {
                "time_s": time_s,
                "source": f"source-{index}",
                "state": [float(index)] * 6,
            }
        )

    frame = records_to_frame(records)

    expected_sources = [
        record["source"]
        for time_s in (1.0, 2.0, 3.0)
        for record in records
        if record["time_s"] == time_s
    ]
    assert frame["source"].tolist() == expected_sources
