#!/usr/bin/env python
"""Run the MMUAD source-calibration path candidate ensemble."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.source_calibration_path_ensemble import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
