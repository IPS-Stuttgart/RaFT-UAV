from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


def test_mmuad_track5_spread_guard_search_entrypoint_is_exposed_and_importable() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"]["raft-uav-mmuad-track5-spread-guard-search"]

    assert target == "raft_uav.mmuad.track5_spread_guard_search:main"

    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
