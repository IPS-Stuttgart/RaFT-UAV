#!/usr/bin/env python
"""Source-tree wrapper for the MMUAD constant-velocity bridge reservoir."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.candidate_constant_velocity_bridge_quota import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
