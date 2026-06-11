from __future__ import annotations

import os

from raft_uav import cli as base_cli
from raft_uav import runtime_cli_patch
from raft_uav import tracklet_viterbi_cli


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


def test_tracklet_cli_restores_base_dispatch_bindings(monkeypatch) -> None:
    """The tracklet wrapper must not permanently monkey-patch the base CLI."""

    sentinel_modes = ("legacy-only",)

    def sentinel_runner(**_kwargs: object) -> None:
        raise AssertionError("sentinel runner should not be called")

    calls: list[list[str]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append(list(argv or []))
        assert "tracklet-viterbi" in base_cli.RADAR_ASSOCIATION_MODES
        assert (
            base_cli.run_async_cv_baseline_with_radar_association
            is tracklet_viterbi_cli.run_async_cv_baseline_with_radar_association
        )
        return 0

    fake_main.__module__ = "raft_uav.runtime_cli_patch"
    monkeypatch.setattr(base_cli, "RADAR_ASSOCIATION_MODES", sentinel_modes)
    monkeypatch.setattr(
        base_cli,
        "run_async_cv_baseline_with_radar_association",
        sentinel_runner,
    )
    monkeypatch.setattr(base_cli, "main", fake_main)

    assert tracklet_viterbi_cli.main(["run-baseline", "/data/aerpaw"]) == 0

    assert calls == [["run-baseline", "/data/aerpaw"]]
    assert base_cli.RADAR_ASSOCIATION_MODES is sentinel_modes
    assert base_cli.run_async_cv_baseline_with_radar_association is sentinel_runner
