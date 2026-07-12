#!/usr/bin/env python3
"""Run agreement-adaptive MMUAD pair forward-backward candidate inference."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raft_uav.mmuad.candidate_pair_forward_backward_agreement import main


if __name__ == "__main__":
    raise SystemExit(main())
