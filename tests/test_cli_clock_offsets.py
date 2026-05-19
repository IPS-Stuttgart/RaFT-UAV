from raft_uav import cli


def test_inspect_parser_forwards_independent_clock_offsets(monkeypatch, tmp_path):
    captured = {}

    def fake_inspect(
        dataset_root,
        requested_flights,
        rf_clock_offset_s,
        radar_clock_offset_s,
    ):
        captured["dataset_root"] = dataset_root
        captured["requested_flights"] = requested_flights
        captured["rf_clock_offset_s"] = rf_clock_offset_s
        captured["radar_clock_offset_s"] = radar_clock_offset_s
        return 0

    monkeypatch.setattr(cli, "_inspect", fake_inspect)

    assert cli.main(
        [
            "inspect",
            str(tmp_path),
            "--rf-clock-offset-s",
            "-14400",
            "--radar-clock-offset-s",
            "0",
        ]
    ) == 0

    assert captured["dataset_root"] == tmp_path
    assert captured["requested_flights"] is None
    assert captured["rf_clock_offset_s"] == -14400.0
    assert captured["radar_clock_offset_s"] == 0.0


def test_run_baseline_parser_forwards_independent_clock_offsets(monkeypatch, tmp_path):
    captured = {}

    def fake_run_baseline(*args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    assert cli.main(
        [
            "run-baseline",
            str(tmp_path),
            "--flight",
            "flight1",
            "--rf-clock-offset-s",
            "-14400",
            "--radar-clock-offset-s",
            "0",
        ]
    ) == 0

    assert captured["args"][4] == -14400.0
    assert captured["args"][5] == 0.0
