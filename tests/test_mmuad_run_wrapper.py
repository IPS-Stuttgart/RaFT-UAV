from __future__ import annotations

from raft_uav.mmuad import run as mmuad_run


def test_mmuad_run_unrecognized_long_option_does_not_hide_sequence_root(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_track_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(mmuad_run, "track_main", fake_track_main)

    result = mmuad_run.main(["--typo-flag", "seq-root", "--output-dir", "out"])

    assert result == 7
    assert captured["argv"] == [
        "--sequence-root",
        "seq-root",
        "--typo-flag",
        "--output-dir",
        "out",
    ]


def test_mmuad_run_known_value_option_still_consumes_its_value(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_track_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(mmuad_run, "track_main", fake_track_main)

    result = mmuad_run.main(["--output-dir", "out", "seq-root"])

    assert result == 0
    assert captured["argv"] == [
        "--sequence-root",
        "seq-root",
        "--output-dir",
        "out",
    ]
