from __future__ import annotations

import runpy

import pytest

import raft_uav.mmuad.candidate_pair_group_correction as group_correction


def test_pair_group_correction_supports_python_m(monkeypatch) -> None:
    calls: list[bool] = []

    monkeypatch.setattr(
        group_correction,
        "main",
        lambda: calls.append(True) or 0,
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(
            "raft_uav.mmuad.candidate_pair_group_correction.__main__",
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert calls == [True]
