from __future__ import annotations

import runpy
import sys

import pytest


def test_nis_reliability_package_supports_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "raft_uav.diagnostics.nis_reliability"
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
