"""Train and apply MMUAD candidate uncertainty with branch-consensus context.

The branch-consensus stage adds truth-free cross-sensor agreement features, but
candidate uncertainty historically consumed only cluster, image, reservoir,
diversity, and dynamic prefixes.  This module provides an inference-safe bridge:
it augments candidates with cross-sensor consensus, exposes numeric consensus
features through the existing ``candidate_reservoir_*`` feature namespace, and
then trains/applies the existing per-candidate sigma model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_branch_consensus import (
    attach_candidate_branch_consensus,
    branch_consensus_summary,
)
from raft_uav.mmuad.candidate_uncertainty import (
    CandidateUncertaintyModel,
    apply_candidate_uncertainty,
    candidate_uncertainty_training_summary,
    load_candidate_uncertainty_model,
    save_candidate_uncertainty_model,
    train_candidate_uncertainty,
)
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

CONSENSUS_INPUT_PREFIX = "branch_consensus_"
CONSENSUS_UNCERTAINTY_PREFIX = "candidate_reservoir_consensus_"


def attach_consensus_uncertainty_features(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    time_window_s: float = 0.05,
    time_scale_s: float | None = None,
    distance_gate_m: float = 5.0,
    distance_scale_m: float = 5.0,
    base_score_column: str = "ranker_score",
    base_score_weight: float = 1.0,
    consensus_weight: float = 1.0,
    pair_advantage_weight: float = 0.25,
    branch_column: str | None = None,
    origin_column: str | None = None,
    exclude_same_origin_support: bool = True,
) -> tuple[CandidateFrame, list[str]]:
    """Attach consensus features and uncertainty-model aliases.

    Numeric ``branch_consensus_*`` columns are copied to the
    ``candidate_reservoir_consensus_*`` namespace.  The existing uncertainty
    feature selector consumes numeric ``candidate_reservoir_*`` columns, so the
    saved model remains portable and validation/test application is truth-free.
    """

    augmented = attach_candidate_branch_consensus(
        candidates,
        time_window_s=time_window_s,
        time_scale_s=time_scale_s,
        distance_gate_m=distance_gate_m,
        distance_scale_m=distance_scale_m,
        base_score_column=base_score_column,
        base_score_weight=base_score_weight,
        consensus_weight=consensus_weight,
        pair_advantage_weight=pair_advantage_weight,
        branch_column=branch_column,
        origin_column=origin_column,
        exclude_same_origin_support=exclude_same_origin_support,
    )
    rows = augmented.rows.copy()
    aliases: list[str] = []
    for column in sorted(str(value) for value in rows.columns):
        if not column.startswith(CONSENSUS_INPUT_PREFIX):
            continue
        numeric = pd.to_numeric(rows[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).any():
            continue
        suffix = column[len(CONSENSUS_INPUT_PREFIX) :]
        alias = f"{CONSENSUS_UNCERTAINTY_PREFIX}{suffix}"
        rows[alias] = numeric
        aliases.append(alias)
    return CandidateFrame(normalize_candidate_columns(rows)), aliases


def train_consensus_conditioned_uncertainty(
    candidates: CandidateFrame | pd.DataFrame,
    truth: pd.DataFrame,
    *,
    model_type: str = "hist-gradient-boosting",
    target_transform: str = "log1p",
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 30.0,
    ridge_alpha: float = 1.0,
    random_state: int = 13,
    n_estimators: int = 300,
    max_truth_time_delta_s: float = 0.5,
    time_window_s: float = 0.05,
    time_scale_s: float | None = None,
    distance_gate_m: float = 5.0,
    distance_scale_m: float = 5.0,
    base_score_column: str = "ranker_score",
    base_score_weight: float = 1.0,
    consensus_weight: float = 1.0,
    pair_advantage_weight: float = 0.25,
    branch_column: str | None = None,
    origin_column: str | None = None,
    exclude_same_origin_support: bool = True,
) -> tuple[CandidateUncertaintyModel, CandidateFrame, pd.DataFrame, dict[str, Any]]:
    """Fit an uncertainty model using cross-sensor consensus context."""

    augmented, aliases = attach_consensus_uncertainty_features(
        candidates,
        time_window_s=time_window_s,
        time_scale_s=time_scale_s,
        distance_gate_m=distance_gate_m,
        distance_scale_m=distance_scale_m,
        base_score_column=base_score_column,
        base_score_weight=base_score_weight,
        consensus_weight=consensus_weight,
        pair_advantage_weight=pair_advantage_weight,
        branch_column=branch_column,
        origin_column=origin_column,
        exclude_same_origin_support=exclude_same_origin_support,
    )
    features = build_cluster_feature_table(
        augmented,
        truth=truth,
        max_truth_time_delta_s=max_truth_time_delta_s,
    )
    model = train_candidate_uncertainty(
        features,
        model_type=model_type,
        target_transform=target_transform,
        sigma_min_m=sigma_min_m,
        sigma_max_m=sigma_max_m,
        ridge_alpha=ridge_alpha,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    summary = candidate_uncertainty_training_summary(features, model)
    summary.update(
        {
            "consensus_feature_aliases": aliases,
            "consensus_feature_count": int(len(aliases)),
            "consensus_features_used_by_model": sorted(
                column
                for column in model.feature_columns
                if column.startswith(CONSENSUS_UNCERTAINTY_PREFIX)
            ),
            "consensus_config": {
                "time_window_s": float(time_window_s),
                "time_scale_s": None if time_scale_s is None else float(time_scale_s),
                "distance_gate_m": float(distance_gate_m),
                "distance_scale_m": float(distance_scale_m),
                "base_score_column": str(base_score_column),
                "base_score_weight": float(base_score_weight),
                "consensus_weight": float(consensus_weight),
                "pair_advantage_weight": float(pair_advantage_weight),
                "branch_column": branch_column,
                "origin_column": origin_column,
                "exclude_same_origin_support": bool(exclude_same_origin_support),
            },
            "consensus_summary": branch_consensus_summary(augmented),
        }
    )
    return model, augmented, features, summary


def apply_consensus_conditioned_uncertainty(
    candidates: CandidateFrame | pd.DataFrame,
    model: CandidateUncertaintyModel,
    *,
    output_column: str = "predicted_sigma_m",
    replace_covariance: bool = False,
    z_scale: float = 1.0,
    time_window_s: float = 0.05,
    time_scale_s: float | None = None,
    distance_gate_m: float = 5.0,
    distance_scale_m: float = 5.0,
    base_score_column: str = "ranker_score",
    base_score_weight: float = 1.0,
    consensus_weight: float = 1.0,
    pair_advantage_weight: float = 0.25,
    branch_column: str | None = None,
    origin_column: str | None = None,
    exclude_same_origin_support: bool = True,
) -> CandidateFrame:
    """Apply a saved consensus-conditioned uncertainty model without truth."""

    augmented, _ = attach_consensus_uncertainty_features(
        candidates,
        time_window_s=time_window_s,
        time_scale_s=time_scale_s,
        distance_gate_m=distance_gate_m,
        distance_scale_m=distance_scale_m,
        base_score_column=base_score_column,
        base_score_weight=base_score_weight,
        consensus_weight=consensus_weight,
        pair_advantage_weight=pair_advantage_weight,
        branch_column=branch_column,
        origin_column=origin_column,
        exclude_same_origin_support=exclude_same_origin_support,
    )
    return apply_candidate_uncertainty(
        augmented,
        model,
        output_column=output_column,
        replace_covariance=replace_covariance,
        z_scale=z_scale,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-consensus-uncertainty",
        description="train/apply candidate uncertainty with cross-sensor consensus context",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--candidates-csv", type=Path, required=True)
    train.add_argument("--truth-csv", type=Path, required=True)
    train.add_argument("--model-json", type=Path, required=True)
    train.add_argument("--augmented-candidates-csv", type=Path)
    train.add_argument("--features-csv", type=Path)
    train.add_argument("--summary-json", type=Path)
    train.add_argument(
        "--model-type",
        choices=("ridge", "random-forest", "hist-gradient-boosting"),
        default="hist-gradient-boosting",
    )
    train.add_argument("--target-transform", choices=("identity", "log1p"), default="log1p")
    train.add_argument("--sigma-min-m", type=float, default=1.0)
    train.add_argument("--sigma-max-m", type=float, default=30.0)
    train.add_argument("--ridge-alpha", type=float, default=1.0)
    train.add_argument("--random-state", type=int, default=13)
    train.add_argument("--n-estimators", type=int, default=300)
    train.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    _add_consensus_arguments(train)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--candidates-csv", type=Path, required=True)
    apply_parser.add_argument("--model-json", type=Path, required=True)
    apply_parser.add_argument("--output-csv", type=Path, required=True)
    apply_parser.add_argument("--output-column", default="predicted_sigma_m")
    apply_parser.add_argument("--replace-covariance", action="store_true")
    apply_parser.add_argument("--z-scale", type=float, default=1.0)
    _add_consensus_arguments(apply_parser)

    args = parser.parse_args(argv)
    consensus_kwargs = _consensus_kwargs(args)
    if args.command == "train":
        candidates = load_candidate_file(args.candidates_csv)
        truth = load_evaluation_truth_file(args.truth_csv).rows
        model, augmented, features, summary = train_consensus_conditioned_uncertainty(
            candidates,
            truth,
            model_type=args.model_type,
            target_transform=args.target_transform,
            sigma_min_m=args.sigma_min_m,
            sigma_max_m=args.sigma_max_m,
            ridge_alpha=args.ridge_alpha,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
            **consensus_kwargs,
        )
        save_candidate_uncertainty_model(model, args.model_json)
        if args.augmented_candidates_csv is not None:
            args.augmented_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
            augmented.rows.to_csv(args.augmented_candidates_csv, index=False)
        if args.features_csv is not None:
            args.features_csv.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.features_csv, index=False)
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("mmuad_consensus_uncertainty_train=ok")
        print(f"model_json={args.model_json}")
        print(f"training_rows={summary.get('row_count', 0)}")
        print(f"consensus_feature_count={summary.get('consensus_feature_count', 0)}")
        return 0

    model = load_candidate_uncertainty_model(args.model_json)
    scored = apply_consensus_conditioned_uncertainty(
        load_candidate_file(args.candidates_csv),
        model,
        output_column=args.output_column,
        replace_covariance=args.replace_covariance,
        z_scale=args.z_scale,
        **consensus_kwargs,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.rows.to_csv(args.output_csv, index=False)
    print("mmuad_consensus_uncertainty_apply=ok")
    print(f"output_csv={args.output_csv}")
    print(f"output_rows={len(scored.rows)}")
    return 0


def _add_consensus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--consensus-time-window-s", type=float, default=0.05)
    parser.add_argument("--consensus-time-scale-s", type=float)
    parser.add_argument("--consensus-distance-gate-m", type=float, default=5.0)
    parser.add_argument("--consensus-distance-scale-m", type=float, default=5.0)
    parser.add_argument("--consensus-base-score-column", default="ranker_score")
    parser.add_argument("--consensus-base-score-weight", type=float, default=1.0)
    parser.add_argument("--consensus-weight", type=float, default=1.0)
    parser.add_argument("--consensus-pair-advantage-weight", type=float, default=0.25)
    parser.add_argument("--consensus-branch-column")
    parser.add_argument("--consensus-origin-column")
    parser.add_argument("--allow-same-origin-support", action="store_true")


def _consensus_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "time_window_s": float(args.consensus_time_window_s),
        "time_scale_s": args.consensus_time_scale_s,
        "distance_gate_m": float(args.consensus_distance_gate_m),
        "distance_scale_m": float(args.consensus_distance_scale_m),
        "base_score_column": str(args.consensus_base_score_column),
        "base_score_weight": float(args.consensus_base_score_weight),
        "consensus_weight": float(args.consensus_weight),
        "pair_advantage_weight": float(args.consensus_pair_advantage_weight),
        "branch_column": args.consensus_branch_column,
        "origin_column": args.consensus_origin_column,
        "exclude_same_origin_support": not bool(args.allow_same_origin_support),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
