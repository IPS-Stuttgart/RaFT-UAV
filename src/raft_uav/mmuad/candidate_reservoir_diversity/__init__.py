"""Compatibility fixes for candidate-reservoir diversity reporting.

The maintained implementation lives in the sibling
``candidate_reservoir_diversity.py`` module. This package preserves the public
import path while aligning summary diagnostics with the configured branch
column and retaining replace-on-use ``--top-k`` CLI semantics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_diversity.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_diversity_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load candidate reservoir diversity implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_DEFAULT_TOP_K_VALUES = (1, 3, 5, 10, 20)


def _frame_label_coverage(
    input_rows: pd.DataFrame,
    output_rows: pd.DataFrame,
    *,
    branch_column: str = "candidate_branch",
) -> tuple[list[float], list[float]]:
    """Return per-frame coverage for the configured branch and source labels."""

    input_frame_map = {
        (str(sequence_id), float(time_s)): group
        for (sequence_id, time_s), group in pd.DataFrame(input_rows).groupby(
            ["sequence_id", "time_s"],
            sort=False,
        )
    }
    output_frame_map = {
        (str(sequence_id), float(time_s)): group
        for (sequence_id, time_s), group in pd.DataFrame(output_rows).groupby(
            ["sequence_id", "time_s"],
            sort=False,
        )
    }
    branch_coverage: list[float] = []
    source_coverage: list[float] = []
    for key, frame in input_frame_map.items():
        output = output_frame_map.get(key, pd.DataFrame())
        branch_coverage.append(
            _IMPL._label_coverage_fraction(frame, output, branch_column)
        )
        source_coverage.append(_IMPL._label_coverage_fraction(frame, output, "source"))
    return branch_coverage, source_coverage


def diversity_cap_summary(
    input_rows: pd.DataFrame,
    output_rows: pd.DataFrame,
    *,
    branch_column: str = "candidate_branch",
) -> dict[str, Any]:
    """Build a summary using the branch dimension selected by the caller."""

    branch_coverage, source_coverage = _frame_label_coverage(
        input_rows,
        output_rows,
        branch_column=branch_column,
    )
    return {
        "input_rows": int(len(input_rows)),
        "output_rows": int(len(output_rows)),
        "input_frame_count": int(_IMPL._frame_counts(input_rows).size),
        "output_frame_count": int(_IMPL._frame_counts(output_rows).size),
        "input_candidates_per_frame_mean": _IMPL._mean(
            _IMPL._frame_counts(input_rows)
        ),
        "output_candidates_per_frame_mean": _IMPL._mean(
            _IMPL._frame_counts(output_rows)
        ),
        "output_candidates_per_frame_p95": _IMPL._quantile(
            _IMPL._frame_counts(output_rows),
            0.95,
        ),
        "output_candidates_per_frame_max": _IMPL._max(
            _IMPL._frame_counts(output_rows)
        ),
        "mean_branch_coverage_fraction": _IMPL._mean(
            pd.Series(branch_coverage, dtype=float)
        ),
        "mean_source_coverage_fraction": _IMPL._mean(
            pd.Series(source_coverage, dtype=float)
        ),
        "frames_all_branches_preserved_fraction": _IMPL._all_preserved_fraction(
            branch_coverage
        ),
        "frames_all_sources_preserved_fraction": _IMPL._all_preserved_fraction(
            source_coverage
        ),
        "source_counts": _IMPL._value_counts(output_rows, "source"),
        "branch_counts": _IMPL._value_counts(output_rows, branch_column),
        "branch_column": str(branch_column),
        "diversity_cap_reason_counts": _IMPL._reason_counts(output_rows),
    }


def write_diversity_cap_outputs(
    capped: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    input_rows: pd.DataFrame | None = None,
    branch_column: str = "candidate_branch",
) -> None:
    """Write outputs whose summary follows the configured branch column."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    capped.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                diversity_cap_summary(
                    input_rows if input_rows is not None else capped,
                    capped,
                    branch_column=branch_column,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    """Run the diversity cap with aligned branch summaries and top-k parsing."""

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
    capped = _IMPL.diversity_cap_reservoir(
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
        branch_column=args.branch_column,
    )
    print("mmuad_diversity_cap_reservoir=ok")
    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(capped)}")
    print(f"output_csv={args.output_csv}")

    if args.truth_csv is not None:
        truth = _IMPL.normalize_truth_columns(pd.read_csv(args.truth_csv))
        frame_rows, pooled, by_sequence = _IMPL.build_oracle_recall_tables(
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


_IMPL._frame_label_coverage = _frame_label_coverage
_IMPL.diversity_cap_summary = diversity_cap_summary
_IMPL.write_diversity_cap_outputs = write_diversity_cap_outputs
_IMPL.main = main

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_frame_label_coverage"] = _frame_label_coverage
globals()["diversity_cap_summary"] = diversity_cap_summary
globals()["write_diversity_cap_outputs"] = write_diversity_cap_outputs
globals()["main"] = main
globals()["_DEFAULT_TOP_K_VALUES"] = _DEFAULT_TOP_K_VALUES

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
