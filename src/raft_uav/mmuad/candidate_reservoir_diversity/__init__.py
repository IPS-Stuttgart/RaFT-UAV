"""Compatibility wrapper for the diversity-preserving candidate reservoir cap.

The maintained implementation lives in the sibling
``candidate_reservoir_diversity.py`` module. This package preserves the public
import path while fixing custom ``--top-k`` handling in the CLI.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_diversity.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_diversity_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate reservoir diversity implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

_DEFAULT_TOP_K_VALUES = (1, 3, 5, 10, 20)


def main(argv: list[str] | None = None) -> int:
    """Run the diversity-cap CLI with replace-on-use ``--top-k`` semantics."""

    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-diversity-cap-reservoir",
        description="apply a diversity-preserving final cap to an MMUAD candidate reservoir",
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--min-per-source", type=int, default=1)
    parser.add_argument("--min-per-branch", type=int, default=1)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    top_k_values = _DEFAULT_TOP_K_VALUES if args.top_k is None else tuple(args.top_k)
    rows = pd.read_csv(args.input_csv)
    capped = diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=args.max_candidates_per_frame,
        min_per_source=args.min_per_source,
        min_per_branch=args.min_per_branch,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        branch_column=args.branch_column,
    )
    write_diversity_cap_outputs(
        capped,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        input_rows=rows,
    )
    print("mmuad_diversity_cap_reservoir=ok")
    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(capped)}")
    print(f"output_csv={args.output_csv}")

    if args.truth_csv is not None:
        truth = normalize_truth_columns(pd.read_csv(args.truth_csv))
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            capped,
            truth,
            top_k_values=top_k_values,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        if args.oracle_frame_csv is not None:
            args.oracle_frame_csv.parent.mkdir(parents=True, exist_ok=True)
            frame_rows.to_csv(args.oracle_frame_csv, index=False)
        if args.oracle_summary_csv is not None:
            args.oracle_summary_csv.parent.mkdir(parents=True, exist_ok=True)
            pooled.to_csv(args.oracle_summary_csv, index=False)
        if args.oracle_by_sequence_csv is not None:
            args.oracle_by_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
            by_sequence.to_csv(args.oracle_by_sequence_csv, index=False)
        if not pooled.empty:
            print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
    return 0


_IMPL.main = main
globals()["main"] = main
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
