import argparse
from threading import Event, Thread

from raft_uav import robust_cli


def test_robust_cli_patch_serializes_overlapping_contexts():
    original_add_argument = argparse.ArgumentParser.add_argument
    first_entered = Event()
    release_first = Event()
    second_attempted = Event()
    second_entered = Event()
    release_second = Event()

    def first_worker() -> None:
        with robust_cli.expose_heavy_tailed_robust_update_modes():
            first_entered.set()
            release_first.wait(timeout=2.0)

    def second_worker() -> None:
        second_attempted.set()
        with robust_cli.expose_heavy_tailed_robust_update_modes():
            second_entered.set()
            release_second.wait(timeout=2.0)

    first = Thread(target=first_worker)
    second = Thread(target=second_worker)
    first.start()
    assert first_entered.wait(timeout=2.0)
    second.start()
    assert second_attempted.wait(timeout=2.0)

    entered_while_first_active = second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2.0)
    assert not first.is_alive()
    assert second_entered.wait(timeout=2.0)
    release_second.set()
    second.join(timeout=2.0)
    assert not second.is_alive()

    final_method_was_restored = argparse.ArgumentParser.add_argument is original_add_argument
    argparse.ArgumentParser.add_argument = original_add_argument

    assert not entered_while_first_active
    assert final_method_was_restored
