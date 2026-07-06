"""Command-line entry point for class-probability candidate context."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.class_probability_context import (
    DEFAULT_INTERACTION_COLUMNS,
    FILL_MISSING_POLICIES,
    attach_class_probability_context,
    write_class_probability_context,
)
from raft_uav.mmuad.class_probability_csv import read_class_probability_csv
from raft_uav.mmuad.io import load_candidate_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-class-prob-context",
        description="attach sequence-level class probabilities to candidates",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--class-probabilities-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--interaction-column", action="append", default=[])
    parser.add_argument("--fill-missing", choices=FILL_MISSING_POLICIES, default="uniform")
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidate_csv)
    probabilities = read_class_probability_csv(args.class_probabilities_csv)
    interaction_columns = tuple(args.interaction_column) or DEFAULT_INTERACTION_COLUMNS
    augmented = attach_class_probability_context(
        candidates,
        probabilities,
        interaction_columns=interaction_columns,
        fill_missing=args.fill_missing,
    )
    write_class_probability_context(
        augmented,
        output_csv=args.output_csv,
        provenance_json=args.provenance_json,
        provenance={
            "candidate_csv": str(args.candidate_csv),
            "class_probabilities_csv": str(args.class_probabilities_csv),
            "fill_missing": str(args.fill_missing),
            "requested_interaction_columns": list(interaction_columns),
        },
    )
    print("mmuad_class_probability_context=ok")
    print(f"output_csv={args.output_csv}")
    if args.provenance_json is not None:
        print(f"provenance_json={args.provenance_json}")
    return 0
