#!/usr/bin/env python
"""Run reliability-weighted multi-anchor MMUAD group selection."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
