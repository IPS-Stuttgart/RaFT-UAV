from pathlib import Path

from raft_uav.io.aerpaw import discover_flights


def test_discover_flights_finds_nested_underscored_rf_root(
    tmp_path: Path,
) -> None:
    flight_dir = (
        tmp_path
        / "extracted"
        / "AADM2025Dryad"
        / "RF_Sensor_and_Radar"
        / "Opt1"
    )
    flight_dir.mkdir(parents=True)
    rf_csv = flight_dir / "AADM.csv"
    rf_csv.write_text("value\n", encoding="utf-8")

    [flight] = discover_flights(tmp_path, variant="original")

    assert flight.name == "Opt1"
    assert flight.root == flight_dir
    assert flight.rf_csv == rf_csv
