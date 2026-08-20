"""Proposal-graph CLI with guarded experimental association improvements."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

from . import _proposal_graph_core as graph_core
from . import proposal_graph_tracker
from ._proposal_common_motion import estimate_common_motion
from ._proposal_delayed_path_cover import (
    DelayedPathCoverConfig,
    track_sequence_delayed_path_cover,
)
from ._proposal_edge_likelihood import load_edge_likelihood_model
from ._proposal_graph_sparse_matching import solve_link_component
from ._proposal_seed_calibration import calibrate_proposals


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enable-seed-calibration", action="store_true")
    parser.add_argument("--seed-calibration-min-pairs", type=int, default=2)
    parser.add_argument("--enable-delayed-path-cover", action="store_true")
    parser.add_argument("--delayed-max-gap", type=int, default=0)
    parser.add_argument("--delayed-lookahead-frames", type=int, default=2)
    parser.add_argument("--delayed-successors-per-frame", type=int, default=3)
    parser.add_argument("--delayed-continuation-weight", type=float, default=0.75)
    parser.add_argument("--delayed-continuation-clip", type=float, default=4.0)
    parser.add_argument("--edge-model-json", type=Path)
    parser.add_argument("--edge-model-weight", type=float, default=1.0)
    parser.add_argument("--edge-model-clip", type=float, default=4.0)
    parser.add_argument("--swarm-relative-weight", type=float, default=0.0)
    parser.add_argument("--swarm-relative-clip", type=float, default=4.0)
    parser.add_argument("--swarm-neighbors", type=int, default=4)
    parser.add_argument("--swarm-radius-scale", type=float, default=12.0)
    parser.add_argument("--swarm-unmatched-penalty", type=float, default=2.0)
    custom, remaining = parser.parse_known_args(arguments)
    if custom.seed_calibration_min_pairs <= 0:
        raise ValueError("--seed-calibration-min-pairs must be positive")
    if (
        custom.edge_model_json is not None or custom.swarm_relative_weight > 0.0
    ) and not custom.enable_delayed_path_cover:
        raise ValueError(
            "edge likelihoods and swarm-relative costs require delayed path cover"
        )

    delayed_config = DelayedPathCoverConfig(
        max_gap=custom.delayed_max_gap,
        lookahead_frames=custom.delayed_lookahead_frames,
        successors_per_frame=custom.delayed_successors_per_frame,
        continuation_weight=custom.delayed_continuation_weight,
        continuation_clip=custom.delayed_continuation_clip,
        edge_model_weight=custom.edge_model_weight,
        edge_model_clip=custom.edge_model_clip,
        swarm_relative_weight=custom.swarm_relative_weight,
        swarm_relative_clip=custom.swarm_relative_clip,
        swarm_neighbors=custom.swarm_neighbors,
        swarm_radius_scale=custom.swarm_radius_scale,
        swarm_unmatched_penalty=custom.swarm_unmatched_penalty,
    )
    delayed_config.validate()
    edge_model = (
        load_edge_likelihood_model(custom.edge_model_json)
        if custom.edge_model_json is not None
        else None
    )
    if custom.edge_model_json is not None:
        model_path = custom.edge_model_json.expanduser().resolve()
        model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        print(f"edge_model_json={model_path}")
        print(f"edge_model_sha256={model_digest}")

    original_motion = graph_core._estimate_common_motion
    original_link_solver = graph_core._solve_link_component
    original_track_sequence = proposal_graph_tracker.track_sequence
    graph_core._estimate_common_motion = estimate_common_motion
    graph_core._solve_link_component = solve_link_component

    selected_track_sequence = original_track_sequence
    if custom.enable_delayed_path_cover:

        def delayed_track_sequence(seeds, proposals, parameters):
            return track_sequence_delayed_path_cover(
                seeds,
                proposals,
                parameters,
                config=delayed_config,
                edge_model=edge_model,
            )

        selected_track_sequence = delayed_track_sequence

    if custom.enable_seed_calibration:
        uncalibrated_track_sequence = selected_track_sequence

        def calibrated_track_sequence(seeds, proposals, parameters):
            calibrated, _summary = calibrate_proposals(
                seeds,
                proposals,
                min_pairs=custom.seed_calibration_min_pairs,
            )
            return uncalibrated_track_sequence(seeds, calibrated, parameters)

        selected_track_sequence = calibrated_track_sequence

    proposal_graph_tracker.track_sequence = selected_track_sequence
    try:
        return proposal_graph_tracker.main(remaining)
    finally:
        graph_core._estimate_common_motion = original_motion
        graph_core._solve_link_component = original_link_solver
        proposal_graph_tracker.track_sequence = original_track_sequence


if __name__ == "__main__":
    raise SystemExit(main())
