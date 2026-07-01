from __future__ import annotations

from raft_uav.mmuad import run


def test_mmuad_run_entrypoint_preserves_mmuad_value_options(monkeypatch) -> None:
    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert (
        run.main(
            [
                "--mmuad-selection-mode",
                "viterbi",
                "--mmuad-viterbi-motion-weight",
                "0.75",
                "--mmuad-source-calibration-json",
                "calibration.json",
                "data/mmuad",
                "--output-dir",
                "outputs/mmuad",
            ]
        )
        == 0
    )
    assert forwarded == [
        "--sequence-root",
        "data/mmuad",
        "--mmuad-selection-mode",
        "viterbi",
        "--mmuad-viterbi-motion-weight",
        "0.75",
        "--mmuad-source-calibration-json",
        "calibration.json",
        "--output-dir",
        "outputs/mmuad",
    ]


def test_mmuad_run_entrypoint_preserves_assignment_form_value_options(monkeypatch) -> None:
    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert (
        run.main(
            [
                "--evaluation-class-map-file=classes.csv",
                "data/mmuad",
                "--output-dir",
                "outputs/mmuad",
            ]
        )
        == 0
    )
    assert forwarded == [
        "--sequence-root",
        "data/mmuad",
        "--evaluation-class-map-file=classes.csv",
        "--output-dir",
        "outputs/mmuad",
    ]
