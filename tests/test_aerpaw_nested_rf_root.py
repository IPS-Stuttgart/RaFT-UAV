from pathlib import Path

from raft_uav.io.aerpaw import discover_flights


def _write_minimal_flight(rf_root: Path) -> tuple[Path, Path]:
    flight_dir = rf_root / "Opt1"
    flight_dir.mkdir(parents=True)
    rf_csv = flight_dir / "AADM.csv"
    rf_csv.write_text("value\n", encoding="utf-8")
    return flight_dir, rf_csv


def test_discover_flights_finds_nested_underscored_rf_root(
    tmp_path: Path,
) -> None:
    rf_root = (
        tmp_path
        / "extracted"
        / "AADM2025Dryad"
        / "RF_Sensor_and_Radar"
    )
    flight_dir, rf_csv = _write_minimal_flight(rf_root)

    [flight] = discover_flights(tmp_path, variant="original")

    assert flight.name == "Opt1"
    assert flight.root == flight_dir
    assert flight.rf_csv == rf_csv


def test_discover_flights_ignores_file_with_rf_root_name(tmp_path: Path) -> None:
    (tmp_path / "RF Sensor and Radar").write_text(
        "not a directory\n",
        encoding="utf-8",
    )
    rf_root = tmp_path / "nested" / "RF_Sensor_and_Radar"
    flight_dir, rf_csv = _write_minimal_flight(rf_root)

    [flight] = discover_flights(tmp_path, variant="original")

    assert flight.root == flight_dir
    assert flight.rf_csv == rf_csv
