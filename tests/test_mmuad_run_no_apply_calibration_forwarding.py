from __future__ import annotations


def test_mmuad_run_entrypoint_preserves_calibration_disable_flag(monkeypatch) -> None:
    from raft_uav.mmuad import run

    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert (
        run.main(
            [
                "--" "no-" "apply-calibration",
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
        "--" "no-" "apply-calibration",
        "--output-dir",
        "outputs/mmuad",
    ]
