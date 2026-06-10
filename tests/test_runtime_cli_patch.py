from __future__ import annotations

import os

from raft_uav import runtime_cli_patch


def test_runtime_cli_patch_restores_runtime_environment_between_run_baseline_calls(
    monkeypatch,
) -> None:
    key = "RAFT_UAV_TRACKLET_TRACK_SWITCH_COST"
    monkeypatch.delenv(key, raising=False)
    calls: list[tuple[list[str], str | None]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append((list(argv or []), os.environ.get(key)))
        return 0

    monkeypatch.setattr(runtime_cli_patch, "_ORIGINAL_MAIN", fake_main)
    monkeypatch.setattr(runtime_cli_patch, "_CURRENT_RUNTIME_CONFIG", None)

    assert (
        runtime_cli_patch._main_with_runtime_config(
            [
                "run-baseline",
                "/data/aerpaw",
                "--tracklet-track-switch-cost",
                "99",
            ]
        )
        == 0
    )
    assert os.environ.get(key) is None

    assert runtime_cli_patch._main_with_runtime_config(["run-baseline", "/data/aerpaw"]) == 0
    assert os.environ.get(key) is None
    assert runtime_cli_patch._CURRENT_RUNTIME_CONFIG is None
    assert calls == [
        (["run-baseline", "/data/aerpaw"], "99.0"),
        (["run-baseline", "/data/aerpaw"], "8.0"),
    ]
