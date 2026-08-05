from __future__ import annotations

import runpy
import sys

import pytest


def test_nis_reliability_package_supports_python_m_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m raft_uav.diagnostics.nis_reliability", "--help"],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("raft_uav.diagnostics.nis_reliability", run_name="__main__")

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "raft-uav-nis-reliability" in captured.out
