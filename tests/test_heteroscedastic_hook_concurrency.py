from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import raft_uav.heteroscedastic_cli as heteroscedastic_cli


def test_hooks_serialize_overlapping_contexts(monkeypatch) -> None:
    monkeypatch.setattr(
        heteroscedastic_cli,
        "load_uncertainty_model",
        lambda _path: object(),
    )
    first_entered = Event()
    release_first = Event()
    second_attempted = Event()
    second_entered = Event()
    errors: list[BaseException] = []

    def first_worker() -> None:
        try:
            with heteroscedastic_cli.heteroscedastic_covariance_hooks(
                Path("first.json")
            ):
                first_entered.set()
                if not release_first.wait(2.0):
                    raise TimeoutError("first hook context was not released")
        except BaseException as exc:
            errors.append(exc)

    def second_worker() -> None:
        try:
            if not first_entered.wait(2.0):
                raise TimeoutError("first hook context was not entered")
            second_attempted.set()
            with heteroscedastic_cli.heteroscedastic_covariance_hooks(
                Path("second.json")
            ):
                second_entered.set()
        except BaseException as exc:
            errors.append(exc)

    first_thread = Thread(target=first_worker)
    second_thread = Thread(target=second_worker)
    first_thread.start()
    second_thread.start()
    try:
        assert first_entered.wait(2.0)
        assert second_attempted.wait(2.0)
        assert not second_entered.wait(0.1)
    finally:
        release_first.set()
        first_thread.join(2.0)
        second_thread.join(2.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()
    assert not errors
