"""Proposal-graph CLI with guarded experimental association improvements."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import _proposal_ambiguity_beam as ambiguity_beam
from . import _proposal_common_motion as proposal_common_motion
from . import _proposal_delayed_path_cover as delayed_path_cover
from . import _proposal_edge_likelihood as proposal_edge_likelihood
from . import _proposal_graph_core as graph_core
from . import _proposal_graph_sparse_matching as sparse_matching
from . import _proposal_seed_calibration as seed_calibration
from . import _proposal_sequence_cache as sequence_cache
from . import proposal_graph_tracker


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-sequence-cache", action="store_true")
    parser.add_argument("--sequence-cache-dir", type=Path)
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
    parser.add_argument("--enable-ambiguity-beam", action="store_true")
    parser.add_argument("--ambiguity-beam-width", type=int, default=8)
    parser.add_argument(
        "--ambiguity-beam-max-component-nodes",
        type=int,
        default=16,
    )
    parser.add_argument("--ambiguity-beam-margin", type=float, default=1.0)
    parser.add_argument(
        "--ambiguity-acceleration-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ambiguity-acceleration-clip",
        type=float,
        default=4.0,
    )
    parser.add_argument("--ambiguity-beam-expansion-factor", type=int, default=4)
    custom, remaining = parser.parse_known_args(arguments)
    if custom.seed_calibration_min_pairs <= 0:
        raise ValueError("--seed-calibration-min-pairs must be positive")
    if (
        custom.edge_model_json is not None or custom.swarm_relative_weight > 0.0
    ) and not custom.enable_delayed_path_cover:
        raise ValueError(
            "edge likelihoods and swarm-relative costs require delayed path cover"
        )
    if custom.enable_ambiguity_beam and not custom.enable_delayed_path_cover:
        raise ValueError("ambiguity beam reranking requires delayed path cover")

    delayed_config = delayed_path_cover.DelayedPathCoverConfig(
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
    beam_config = ambiguity_beam.AmbiguityBeamConfig(
        width=custom.ambiguity_beam_width,
        max_component_nodes=custom.ambiguity_beam_max_component_nodes,
        margin=custom.ambiguity_beam_margin,
        acceleration_weight=custom.ambiguity_acceleration_weight,
        acceleration_clip=custom.ambiguity_acceleration_clip,
        expansion_factor=custom.ambiguity_beam_expansion_factor,
    )
    beam_config.validate()
    edge_model = (
        proposal_edge_likelihood.load_edge_likelihood_model(custom.edge_model_json)
        if custom.edge_model_json is not None
        else None
    )
    edge_model_digest: str | None = None
    if custom.edge_model_json is not None:
        model_path = custom.edge_model_json.expanduser().resolve()
        edge_model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        print(f"edge_model_json={model_path}")
        print(f"edge_model_sha256={edge_model_digest}")

    original_motion = graph_core._estimate_common_motion
    original_link_solver = graph_core._solve_link_component
    original_track_sequence = proposal_graph_tracker.track_sequence
    graph_core._estimate_common_motion = proposal_common_motion.estimate_common_motion
    graph_core._solve_link_component = sparse_matching.solve_link_component

    selected_track_sequence = original_track_sequence
    if custom.enable_delayed_path_cover:
        if custom.enable_ambiguity_beam:

            def delayed_track_sequence(seeds, proposals, parameters):
                return ambiguity_beam.track_sequence_ambiguity_beam(
                    seeds,
                    proposals,
                    parameters,
                    delayed_config=delayed_config,
                    beam_config=beam_config,
                    edge_model=edge_model,
                )

        else:

            def delayed_track_sequence(seeds, proposals, parameters):
                return delayed_path_cover.track_sequence_delayed_path_cover(
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
            calibrated, _summary = seed_calibration.calibrate_proposals(
                seeds,
                proposals,
                min_pairs=custom.seed_calibration_min_pairs,
            )
            return uncalibrated_track_sequence(seeds, calibrated, parameters)

        selected_track_sequence = calibrated_track_sequence

    proposal_graph_tracker.track_sequence = selected_track_sequence
    try:
        if custom.no_sequence_cache:
            return proposal_graph_tracker.main(remaining)
        return sequence_cache.run_cached(
            remaining,
            tracker_main=proposal_graph_tracker.main,
            cache_salt=_cache_salt(custom, edge_model_digest=edge_model_digest),
            cache_dir=custom.sequence_cache_dir,
        )
    finally:
        graph_core._estimate_common_motion = original_motion
        graph_core._solve_link_component = original_link_solver
        proposal_graph_tracker.track_sequence = original_track_sequence


def _cache_salt(
    custom: argparse.Namespace,
    *,
    edge_model_digest: str | None,
) -> str:
    controls = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(custom).items()
        if key not in {"no_sequence_cache", "sequence_cache_dir"}
    }
    source_paths = (
        Path(__file__),
        Path(ambiguity_beam.__file__),
        Path(proposal_common_motion.__file__),
        Path(delayed_path_cover.__file__),
        Path(proposal_edge_likelihood.__file__),
        Path(graph_core.__file__),
        Path(sparse_matching.__file__),
        Path(seed_calibration.__file__),
        Path(sequence_cache.__file__),
        Path(proposal_graph_tracker.__file__),
    )
    source_digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    return json.dumps(
        {
            "schema": "raft-uav-lts-experimental-proposal-graph-cache-salt-v2",
            "controls": controls,
            "edge_model_sha256": edge_model_digest,
            "source_sha256": source_digests,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
