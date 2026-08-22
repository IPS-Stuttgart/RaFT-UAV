#!/usr/bin/env python3
"""Run the guarded LTS tournament with the swarm-relative edge ablation."""

from __future__ import annotations

import run_multi_uav_lts_improved_evidence as improved

SWARM_RELATIVE_CANDIDATE = (
    "graph_delayed_swarm_relative",
    (
        "--enable-delayed-path-cover",
        "--delayed-max-gap",
        "0",
        "--delayed-lookahead-frames",
        "2",
        "--delayed-successors-per-frame",
        "3",
        "--delayed-continuation-weight",
        "0.75",
        "--enable-common-motion",
        "--common-motion-min-pairs",
        "4",
        "--common-motion-max-normalized-step",
        "8.0",
        "--common-motion-max-normalized-residual",
        "1.5",
        "--swarm-relative-weight",
        "0.75",
        "--swarm-relative-clip",
        "4.0",
        "--swarm-neighbors",
        "4",
        "--swarm-radius-scale",
        "12.0",
    ),
)


def main() -> int:
    names = {name for name, _arguments in improved.IMPROVED_CANDIDATES}
    if SWARM_RELATIVE_CANDIDATE[0] not in names:
        improved.IMPROVED_CANDIDATES = (
            *improved.IMPROVED_CANDIDATES,
            SWARM_RELATIVE_CANDIDATE,
        )
    return improved.main()


if __name__ == "__main__":
    raise SystemExit(main())
