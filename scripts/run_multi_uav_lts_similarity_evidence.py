#!/usr/bin/env python3
"""Run the guarded LTS evidence suite with similarity-motion candidates."""

from __future__ import annotations

import run_multi_uav_lts_improved_evidence as improved

SIMILARITY_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "graph_similarity_common_motion",
        (
            "--enable-common-motion",
            "--common-motion-model",
            "similarity",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--similarity-min-pairs",
            "4",
            "--similarity-max-scale-change",
            "0.12",
            "--similarity-max-rotation-deg",
            "10.0",
            "--similarity-max-normalized-residual",
            "1.0",
            "--similarity-min-normalized-spread",
            "2.0",
            "--similarity-min-residual-improvement",
            "0.05",
        ),
    ),
    (
        "graph_delayed_similarity_beam",
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
            "--enable-ambiguity-beam",
            "--ambiguity-beam-width",
            "8",
            "--enable-common-motion",
            "--common-motion-model",
            "similarity",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--similarity-min-pairs",
            "4",
            "--similarity-max-scale-change",
            "0.12",
            "--similarity-max-rotation-deg",
            "10.0",
            "--similarity-max-normalized-residual",
            "1.0",
            "--similarity-min-normalized-spread",
            "2.0",
            "--similarity-min-residual-improvement",
            "0.05",
        ),
    ),
)


def main() -> int:
    improved.IMPROVED_CANDIDATES = (
        *improved.IMPROVED_CANDIDATES,
        *SIMILARITY_CANDIDATES,
    )
    return improved.main()


if __name__ == "__main__":
    raise SystemExit(main())
