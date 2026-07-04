"""Console entry point for MMUAD temporal-consensus train-CV selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_temporal_consensus_train_cv import (
    _DEFAULT_SELECTION_METRIC,
    _DEFAULT_TOP_K,
    _jsonable,
    _parse_float_grid,
    select_temporal_consensus_config_by_sequence_cv,
)
from raft_uav.mmuad.io import load_candidate_file


def main(argv: list[str] | None = None) -> int:
    """Run train-only temporal-consensus grid selection with replacement top-k semantics."""

    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-temporal-consensus-train-cv",
        description="select MMUAD temporal-consensus weights on train sequences",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-score-weight-grid", default="0.1,0.25,0.5")
    parser.add_argument("--support-weight-grid", default="0.5,1.0,1.5")
    parser.add_argument("--bidirectional-bonus-grid", default="0,0.75")
    parser.add_argument("--interpolation-weight-grid", default="0,0.75")
    parser.add_argument("--acceleration-weight-grid", default="0,0.5")
    parser.add_argument("--max-time-gap-s", type=float, default=2.0)
    parser.add_argument("--max-speed-mps", type=float, default=60.0)
    parser.add_argument("--distance-scale-m", type=float, default=5.0)
    parser.add_argument("--acceleration-scale-mps2", type=float, default=20.0)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--source-diversity-bonus", type=float, default=0.25)
    parser.add_argument("--branch-diversity-bonus", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--selection-metric", default=_DEFAULT_SELECTION_METRIC)
    parser.add_argument("--write-selected-candidates", action="store_true")
    args = parser.parse_args(argv)

    top_k_values = tuple(args.top_k) if args.top_k is not None else _DEFAULT_TOP_K
    candidates = load_candidate_file(args.candidate_csv).rows
    truth = pd.read_csv(args.truth_csv)
    selected, folds, grid, selected_candidates = select_temporal_consensus_config_by_sequence_cv(
        candidates,
        truth,
        base_score_weights=_parse_float_grid(args.base_score_weight_grid),
        support_weights=_parse_float_grid(args.support_weight_grid),
        bidirectional_bonuses=_parse_float_grid(args.bidirectional_bonus_grid),
        interpolation_weights=_parse_float_grid(args.interpolation_weight_grid),
        acceleration_weights=_parse_float_grid(args.acceleration_weight_grid),
        max_time_gap_s=args.max_time_gap_s,
        max_speed_mps=args.max_speed_mps,
        distance_scale_m=args.distance_scale_m,
        acceleration_scale_mps2=args.acceleration_scale_mps2,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        source_diversity_bonus=args.source_diversity_bonus,
        branch_diversity_bonus=args.branch_diversity_bonus,
        top_k_values=top_k_values,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        selection_metric=args.selection_metric,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_json = output_dir / "mmuad_temporal_consensus_train_selected_config.json"
    fold_csv = output_dir / "mmuad_temporal_consensus_train_cv_folds.csv"
    grid_csv = output_dir / "mmuad_temporal_consensus_train_grid_summary.csv"
    config_json.write_text(json.dumps(_jsonable(selected), indent=2), encoding="utf-8")
    folds.to_csv(fold_csv, index=False)
    grid.to_csv(grid_csv, index=False)
    selected_csv: Path | None = None
    if args.write_selected_candidates:
        selected_csv = output_dir / "mmuad_temporal_consensus_train_selected_candidates.csv"
        selected_candidates.to_csv(selected_csv, index=False)

    print("mmuad_temporal_consensus_train_cv=ok")
    print(f"selected_config_json={config_json}")
    print(f"fold_summary_csv={fold_csv}")
    print(f"grid_summary_csv={grid_csv}")
    if selected_csv is not None:
        print(f"selected_candidates_csv={selected_csv}")
    print(f"selected_config_id={selected['selected_config_id']}")
    print(f"selected_metric_value={selected['selected_metric_value']}")
    return 0


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
