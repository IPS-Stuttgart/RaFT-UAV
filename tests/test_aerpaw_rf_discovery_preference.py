from pathlib import Path

import pytest

from raft_uav.io.aerpaw import discover_flights


@pytest.mark.parametrize(
    ("variant", "expected_name"),
    [
        ("auto", "AADM_rerun.csv"),
        ("original", "AADM.csv"),
        ("rerun", "AADM_rerun.csv"),
    ],
)
def test_discover_flights_prefers_aadm_rf_exports(
    tmp_path: Path,
    variant: str,
    expected_name: str,
) -> None:
    flight_dir = tmp_path / "RF Sensor and Radar" / "Opt1"
    flight_dir.mkdir(parents=True)
    for name in (
        "000_diagnostics.csv",
        "000_diagnostics_rerun.csv",
        "AADM-archive.csv",
        "AADM-archive-rerun.csv",
        "AADM.csv",
        "AADM_rerun.csv",
    ):
        (flight_dir / name).write_text("value\n", encoding="utf-8")

    [flight] = discover_flights(tmp_path, variant=variant)

    assert flight.rf_csv == flight_dir / expected_name


@pytest.mark.parametrize(
    ("variant", "canonical_name", "legacy_name", "expected_name"),
    [
        (
            "auto",
            "AADM.csv",
            "rf_measurements_rerun.csv",
            "rf_measurements_rerun.csv",
        ),
        (
            "rerun",
            "AADM.csv",
            "rf_measurements_rerun.csv",
            "rf_measurements_rerun.csv",
        ),
        (
            "original",
            "AADM_rerun.csv",
            "rf_measurements.csv",
            "rf_measurements.csv",
        ),
    ],
)
def test_discover_flights_does_not_cross_rf_variants(
    tmp_path: Path,
    variant: str,
    canonical_name: str,
    legacy_name: str,
    expected_name: str,
) -> None:
    flight_dir = tmp_path / "RF Sensor and Radar" / "Opt1"
    flight_dir.mkdir(parents=True)
    for name in (canonical_name, legacy_name):
        (flight_dir / name).write_text("value\n", encoding="utf-8")

    [flight] = discover_flights(tmp_path, variant=variant)

    assert flight.rf_csv == flight_dir / expected_name


def test_discover_flights_keeps_legacy_rf_filename_fallback(tmp_path: Path) -> None:
    flight_dir = tmp_path / "RF Sensor and Radar" / "Opt1"
    flight_dir.mkdir(parents=True)
    legacy_rf = flight_dir / "rf_measurements.csv"
    legacy_rf.write_text("value\n", encoding="utf-8")

    [flight] = discover_flights(tmp_path)

    assert flight.rf_csv == legacy_rf
