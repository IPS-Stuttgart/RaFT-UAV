from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_mmuad_import_does_not_require_optional_pyrecest_runtime_hook():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys


class OptionalPyrecestBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyrecest.filters.adaptive_process_noise":
            raise ModuleNotFoundError(
                "simulated missing optional pyrecest runtime hook",
                name=fullname,
            )
        return None


sys.meta_path.insert(0, OptionalPyrecestBlocker())
from raft_uav.mmuad.submission import parse_official_position_cell

assert parse_official_position_cell("[1,2,3]") == (1.0, 2.0, 3.0)
"""
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
