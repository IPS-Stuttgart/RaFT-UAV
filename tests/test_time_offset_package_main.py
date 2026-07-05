from __future__ import annotations

import importlib


def test_time_offset_package_main_forwards_to_cli_main() -> None:
    time_offset = importlib.import_module("raft_uav.diagnostics.time_offset")
    package_main = importlib.import_module("raft_uav.diagnostics.time_offset.__main__")

    assert package_main.main is time_offset.main
