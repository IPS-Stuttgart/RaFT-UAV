"""Proposal-graph CLI with guarded experimental association improvements."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import _proposal_graph_core as graph_core
from . import proposal_graph_tracker
from ._proposal_common_motion import estimate_common_motion
from ._proposal_graph_sparse_matching import solve_link_component
from ._proposal_seed_calibration import calibrate_proposals


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enable-seed-calibration", action="store_true")
    parser.add_argument("--seed-calibration-min-pairs", type=int, default=2)
    custom, remaining = parser.parse_known_args(arguments)
    if custom.seed_calibration_min_pairs <= 0:
        raise ValueError("--seed-calibration-min-pairs must be positive")

    original_motion = graph_core._estimate_common_motion
    original_link_solver = graph_core._solve_link_component
    original_track_sequence = proposal_graph_tracker.track_sequence
    graph_core._estimate_common_motion = estimate_common_motion
    graph_core._solve_link_component = solve_link_component

    if custom.enable_seed_calibration:

        def calibrated_track_sequence(seeds, proposals, parameters):
            calibrated, _summary = calibrate_proposals(
                seeds,
                proposals,
                min_pairs=custom.seed_calibration_min_pairs,
            )
            return original_track_sequence(seeds, calibrated, parameters)

        proposal_graph_tracker.track_sequence = calibrated_track_sequence
    try:
        return proposal_graph_tracker.main(remaining)
    finally:
        graph_core._estimate_common_motion = original_motion
        graph_core._solve_link_component = original_link_solver
        proposal_graph_tracker.track_sequence = original_track_sequence


if __name__ == "__main__":
    raise SystemExit(main())
