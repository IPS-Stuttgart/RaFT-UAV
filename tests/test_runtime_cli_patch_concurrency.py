from __future__ import annotations

import os
from threading import Event, Thread

from raft_uav import runtime_cli_patch


def test_runtime_cli_patch_serializes_overlapping_invocations(monkeypatch) -> None:
    key = "RAFT_UAV_TRACKLET_TRACK_SWITCH_COST"
    monkeypatch.delenv(key, raising=False)
    first_entered = Event()
    release_first = Event()
    second_attempted = Event()
    second_entered = Event()
    calls: list[tuple[str | None, float]] = []
    results: list[int] = []
    errors: list[BaseException] = []

    def fake_main(argv: list[str] | None = None) -> int:
        del argv
        value = os.environ.get(key)
        config = runtime_cli_patch._CURRENT_RUNTIME_CONFIG
        assert config is not None
        cost = float(config["tracklet_viterbi"]["track_switch_cost"])
        if value == "99.0":
            first_entered.set()
            release_first.wait(timeout=2.0)
        elif value == "17.0":
            second_entered.set()
        calls.append((value, cost))
        return 0

    def invoke(argv: list[str]) -> None:
        try:
            results.append(runtime_cli_patch._main_with_runtime_config(argv))
        except BaseException as exc:
            errors.append(exc)

    first = Thread(
        target=invoke,
        args=(
            [
                "run-baseline",
                "/data/aerpaw",
                "--tracklet-track-switch-cost",
                "99",
            ],
        ),
    )

    def second_worker() -> None:
        second_attempted.set()
        invoke(
            [
                "run-baseline",
                "/data/aerpaw",
                "--tracklet-track-switch-cost",
                "17",
            ]
        )

    second = Thread(target=second_worker)
    monkeypatch.setattr(runtime_cli_patch, "_ORIGINAL_MAIN", fake_main)
    monkeypatch.setattr(runtime_cli_patch, "_CURRENT_RUNTIME_CONFIG", None)

    first.start()
    assert first_entered.wait(timeout=2.0)
    second.start()
    assert second_attempted.wait(timeout=2.0)

    entered_while_first_active = second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not entered_while_first_active
    assert second_entered.is_set()
    assert errors == []
    assert results == [0, 0]
    assert calls == [("99.0", 99.0), ("17.0", 17.0)]
    assert os.environ.get(key) is None
    assert runtime_cli_patch._CURRENT_RUNTIME_CONFIG is None
