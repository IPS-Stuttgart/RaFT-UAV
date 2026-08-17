#!/usr/bin/env python3
"""Run the public evidence gate with guarded association-improvement candidates."""

from __future__ import annotations

import run_multi_uav_lts_public_evidence as evidence

IMPROVED_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "graph_common_motion",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
        ),
    ),
    (
        "graph_interpolate_one",
        ("--interpolate-max-gap", "1"),
    ),
    (
        "graph_common_motion_interpolate",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--interpolate-max-gap",
            "1",
        ),
    ),
    (
        "graph_guarded_border_birth",
        (
            "--birth-require-border-entry",
            "--birth-min-inward-motion",
            "0.0",
        ),
    ),
    (
        "graph_common_motion_guarded_birth",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--birth-require-border-entry",
            "--birth-min-inward-motion",
            "0.0",
        ),
    ),
)


def main() -> int:
    evidence.CANDIDATES = (*evidence.CANDIDATES, *IMPROVED_CANDIDATES)
    return evidence.main()


if __name__ == "__main__":
    raise SystemExit(main())
