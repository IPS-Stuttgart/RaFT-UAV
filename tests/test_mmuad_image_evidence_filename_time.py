from pathlib import Path

import pytest

from raft_uav.mmuad.image_evidence import _image_file_rows, _timestamp_from_filename


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("frame-001.25.png", 1.25),
        ("livox-42.jpg", 42.0),
        ("frame-1e-3.png", 1.0e-3),
        ("-1.25.png", -1.25),
    ],
)
def test_timestamp_from_filename_uses_separator_safe_parser(
    filename: str,
    expected: float,
) -> None:
    assert _timestamp_from_filename(Path(filename)) == pytest.approx(expected)


def test_timestamp_from_filename_without_number_returns_none() -> None:
    assert _timestamp_from_filename(Path("frame.png")) is None


def test_image_file_rows_sorts_hyphenated_timestamps_as_positive_values() -> None:
    rows = _image_file_rows([Path("frame-2.0.png"), Path("frame-1.0.png")])

    assert rows["image_time_s"].tolist() == [1.0, 2.0]
