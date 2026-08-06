from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import raft_uav.mmuad.track5_scorecard_compare as compare_module


def test_scorecard_compare_main_serializes_pandas_proxy(monkeypatch) -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()

    def fake_main(argv: list[str] | None = None) -> int:
        marker = None if not argv else argv[0]
        if marker == "first":
            first_entered.set()
            if not release_first.wait(timeout=5.0):
                raise AssertionError("first scorecard invocation was not released")
        elif marker == "second":
            second_entered.set()
            if not release_second.wait(timeout=5.0):
                raise AssertionError("second scorecard invocation was not released")
        return 0

    def run_second() -> int:
        second_started.set()
        return compare_module.main(["second"])

    monkeypatch.setattr(compare_module, "_ORIGINAL_MAIN", fake_main)
    original_pandas = compare_module._IMPL.pd
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        first = executor.submit(compare_module.main, ["first"])
        assert first_entered.wait(timeout=5.0)

        second = executor.submit(run_second)
        assert second_started.wait(timeout=5.0)
        overlapped = second_entered.wait(timeout=0.5)

        release_first.set()
        assert first.result(timeout=5.0) == 0
        if not overlapped:
            assert second_entered.wait(timeout=5.0)
        release_second.set()
        assert second.result(timeout=5.0) == 0

        assert not overlapped
        assert compare_module._IMPL.pd is original_pandas
    finally:
        release_first.set()
        release_second.set()
        executor.shutdown(wait=True)
        compare_module._IMPL.pd = original_pandas
