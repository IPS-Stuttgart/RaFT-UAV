from __future__ import annotations

from dataclasses import dataclass

import pytest

from raft_uav.mmuad import native_ros


@pytest.mark.parametrize(
    "nanoseconds",
    (
        -1,
        -0.5,
        0.5,
        1_000_000_000,
        1_000_000_000.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        "",
        "not-a-number",
        True,
    ),
)
def test_time_field_to_s_rejects_invalid_nanoseconds(nanoseconds: object) -> None:
    assert native_ros._time_field_to_s({"sec": 5, "nanosec": nanoseconds}) is None


@pytest.mark.parametrize(
    ("nanoseconds", "expected"),
    (
        (0, 5.0),
        (999_999_999, 5.999_999_999),
        (500_000_000.0, 5.5),
        ("500000000", 5.5),
    ),
)
def test_time_field_to_s_accepts_valid_nanoseconds(
    nanoseconds: object,
    expected: float,
) -> None:
    assert native_ros._time_field_to_s(
        {"sec": 5, "nanosec": nanoseconds}
    ) == pytest.approx(expected)


@pytest.mark.parametrize("seconds", (float("nan"), float("inf"), float("-inf"), True))
def test_time_field_to_s_rejects_invalid_seconds(seconds: object) -> None:
    assert native_ros._time_field_to_s({"sec": seconds, "nanosec": 0}) is None


@dataclass
class _RosStamp:
    sec: object
    nanosec: object


def test_time_field_to_s_validates_object_stamps() -> None:
    assert native_ros._time_field_to_s(_RosStamp(2, 250_000_000)) == pytest.approx(2.25)
    assert native_ros._time_field_to_s(_RosStamp(2, 1_250_000_000)) is None


def test_time_field_to_s_preserves_scalar_fallback() -> None:
    assert native_ros._time_field_to_s("2.5") == pytest.approx(2.5)
    assert native_ros._time_field_to_s(float("nan")) is None
    assert native_ros._time_field_to_s(True) is None
