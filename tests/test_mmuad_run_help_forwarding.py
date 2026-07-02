from __future__ import annotations


def test_mmuad_run_entrypoint_forwards_help_without_sequence_root_rewrite(monkeypatch) -> None:
    from raft_uav.mmuad import run

    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert run.main(["--help", "data/mmuad", "--output-dir", "outputs/mmuad"]) == 0
    assert forwarded == ["--help", "data/mmuad", "--output-dir", "outputs/mmuad"]
