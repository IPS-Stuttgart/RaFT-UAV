"""Console wrapper preserving textual Track 5 sequence identifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.track5_estimate_sequence_gate_fit import DEFAULT_WEIGHT_GRID
from raft_uav.mmuad.track5_estimate_sequence_gate_fit import fit_estimate_sequence_gate_weights
from raft_uav.mmuad.track5_estimate_sequence_gate_fit import write_estimate_sequence_gate_fit_outputs
from raft_uav.mmuad.track5_estimate_sequence_gate_fit import _weight_grid_text


def _read_estimate_csv(path: Path) -> pd.DataFrame:
    """Read estimate CSVs without coercing opaque sequence IDs to numbers or NA."""

    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows.columns = [str(column).strip() for column in rows.columns]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-sequence-gate-fit",
        description="fit estimate-level Track 5 sequence-gate weights",
    )
    parser.add_argument("--base-estimates", type=Path, required=True)
    parser.add_argument("--alternate-estimates", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-grid", default=",".join(f"{value:g}" for value in DEFAULT_WEIGHT_GRID))
    parser.add_argument("--apply-base-estimates", type=Path)
    parser.add_argument("--apply-alternate-estimates", type=Path)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    args = parser.parse_args(argv)

    template = load_official_track5_template_file(args.template)
    grid = _weight_grid_text(args.weight_grid)
    result = fit_estimate_sequence_gate_weights(
        base_estimates=_read_estimate_csv(args.base_estimates),
        alternate_estimates=_read_estimate_csv(args.alternate_estimates),
        template=template,
        truth=load_evaluation_truth_file(args.truth_csv).rows,
        weight_grid=grid,
        apply_base_estimates=None
        if args.apply_base_estimates is None
        else _read_estimate_csv(args.apply_base_estimates),
        apply_alternate_estimates=None
        if args.apply_alternate_estimates is None
        else _read_estimate_csv(args.apply_alternate_estimates),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    paths = write_estimate_sequence_gate_fit_outputs(
        result=result,
        output_dir=args.output_dir,
        base_estimates_path=args.base_estimates,
        alternate_estimates_path=args.alternate_estimates,
        apply_base_estimates_path=args.apply_base_estimates,
        apply_alternate_estimates_path=args.apply_alternate_estimates,
        template_path=args.template,
        truth_path=args.truth_csv,
        weight_grid=grid,
    )
    print("mmuad_track5_estimate_sequence_gate_fit=ok")
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
