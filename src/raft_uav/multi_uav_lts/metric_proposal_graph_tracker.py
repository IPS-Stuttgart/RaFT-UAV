"""Run the experimental proposal graph with a metric-aligned multi-head edge model."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import experimental_proposal_graph_tracker as experimental
from ._proposal_metric_edge_likelihood import load_metric_edge_likelihood_model


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metric-edge-model-json")
    custom, remaining = parser.parse_known_args(arguments)
    if custom.metric_edge_model_json is None:
        return experimental.main(remaining)
    if "--edge-model-json" in remaining:
        raise ValueError("metric and legacy edge models are mutually exclusive")

    from pathlib import Path

    model_path = Path(custom.metric_edge_model_json).expanduser().resolve()
    model = load_metric_edge_likelihood_model(model_path)
    original_loader = experimental.proposal_edge_likelihood.load_edge_likelihood_model

    def load_metric(_path):
        return model

    experimental.proposal_edge_likelihood.load_edge_likelihood_model = load_metric
    delegated = [*remaining, "--edge-model-json", str(model_path)]
    if "--no-sequence-cache" not in delegated:
        delegated.append("--no-sequence-cache")
    try:
        return experimental.main(delegated)
    finally:
        experimental.proposal_edge_likelihood.load_edge_likelihood_model = original_loader


if __name__ == "__main__":
    raise SystemExit(main())
