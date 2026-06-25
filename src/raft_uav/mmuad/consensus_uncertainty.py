"""Cross-sensor-consensus conditioning for MMUAD candidate uncertainty.

The branch-preserving MMUAD pipeline can contain raw, dynamic, calibrated, and
merged hypotheses whose score scales are not directly comparable.  The existing
candidate-uncertainty model learns a per-candidate position scale, while the
branch-consensus module provides a truth-free signal that a hypothesis is
supported by an independent sensor.  This module composes those two pieces:

* attach cross-sensor branch-consensus features;
* mirror the numeric consensus diagnostics into the feature namespace consumed
  by the uncertainty model;
* train/apply the existing uncertainty estimator on those features; and
* optionally reduce predicted sigma monotonically for independently supported
  candidates without replacing the original ranker score.

Validation/test application does not require truth.  Truth is used only by the
``train`` subcommand to fit the uncertainty model on the training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_branch_consensus import (
    DEFAULT_SCORE_OUTPUT_COLUMN,
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

CONSENSUS_FEATURE_COLUMNS = (
    "branch_consensus_base_score",
    "branch_consensus_base_score_normalized",
    "branch_consensus_nearest_cross_source_distance_m",
    "branch_consensus_nearest_cross_source_time_delta_s",
    "branch_consensus_neighbor_count",
    "branch_consensus_unique_source_count",
    "branch_consensus_unique_branch_count",
    "branch_consensus_distance_score",
    "branch_consensus_time_score",
    "branch_consensus_support_score",
    "branch_consensus_score",
    "branch_consensus_pair_advantage_m",
    "branch_consensus_pair_preference",
    DEFAULT_SCORE_OUTPUT_COLUMN,
    "branch_consensus_rank_percentile",
)
CONSENSUS_FEATURE_PREFIX = "image_"
DEFAULT_SIGMA_FACTOR_COLUMN = "candidate_uncertainty_consensus_factor"


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
) -> CandidateFrame:
    """Attach consensus diagnostics and uncertainty-model feature aliases.

    ``candidate_uncertainty`` already consumes numeric ``image_*`` columns as
    optional context.  Mirroring the truth-free consensus diagnostics into that
    namespace keeps backward compatibility with existing saved uncertainty
    models while allowing newly trained models to learn cross-sensor support.
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
    for column in CONSENSUS_FEATURE_COLUMNS:
        if column not in rows.columns:
            continue
        numeric = pd.to_numeric(rows[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).any():
            continue
        rows[f"{CONSENSUS_FEATURE_PREFIX}{column}"] = numeric
    return CandidateFrame(normalize_candidate_columns(rows))


def apply_consensus_conditioned_uncertainty(
    candidates: CandidateFrame | pd.DataFrame,
    model: CandidateUncertaintyModel,
    *,
    output_column: str = "predicted_sigma_m",
    replace_covariance: bool = False,
    z_scale: float = 1.0,
    consensus_score_column: str = "branch_consensus_score",
    consensus_sigma_weight: float = 0.0,
    consensus_sigma_min_factor: float = 0.25,
    sigma_factor_column: str = DEFAULT_SIGMA_FACTOR_COLUMN,
    **consensus_kwargs: Any,
) -> CandidateFrame:
    """Apply learned uncertainty and optional monotonic consensus shrinkage.

    The adjustment is deliberately conservative and opt-in.  For a consensus
    score clipped to ``[0, 1]`` the multiplicative factor is

    ``max(min_factor, exp(-weight * score))``.

    A candidate without cross-sensor support therefore keeps its learned sigma;
    an independently supported candidate receives a smaller measurement scale.
    The raw learned prediction is retained as ``raw_<output_column>``.
    """

    if float(consensus_sigma_weight) < 0.0:
        raise ValueError("consensus_sigma_weight must be non-negative")
    if not 0.0 < float(consensus_sigma_min_factor) <= 1.0:
        raise ValueError("consensus_sigma_min_factor must be in (0, 1]")
    if float(z_scale) <= 0.0:
        raise ValueError("z_scale must be positive")

    augmented = attach_consensus_uncertainty_features(candidates, **consensus_kwargs)
    scored = apply_candidate_uncertainty(
        augmented,
        model,
        output_column=output_column,
        replace_covariance=False,
    )
    rows = scored.rows.copy()
    raw_sigma = pd.to_numeric(rows[output_column], errors="coerce")
    rows[f"raw_{output_column}"] = raw_sigma

    if consensus_score_column not in rows.columns:
        if float(consensus_sigma_weight) > 0.0:
            raise ValueError(
                f"consensus score column {consensus_score_column!r} is missing"
            )
        consensus_score = pd.Series(0.0, index=rows.index, dtype=float)
    else:
        consensus_score = pd.to_numeric(rows[consensus_score_column], errors="coerce")
        consensus_score = consensus_score.fillna(0.0).clip(lower=0.0, upper=1.0)

    factor = np.exp(-float(consensus_sigma_weight) * consensus_score.to_numpy(float))
    factor = np.clip(factor, float(consensus_sigma_min_factor), 1.0)
    rows[sigma_factor_column] = factor
    adjusted = raw_sigma.to_numpy(float) * factor
    adjusted = np.nan_to_num(
        adjusted,
        nan=float(model.fallback_sigma_m),
        posinf=float(model.sigma_max_m),
        neginf=float(model.sigma_min_m),
    )
    rows[output_column] = np.clip(
        adjusted,
        float(model.sigma_min_m),
        float(model.sigma_max_m),
    )

    if replace_covariance:
        rows["raw_std_xy_m"] = pd.to_numeric(rows.get("std_xy_m"), errors="coerce")
        rows["raw_std_z_m"] = pd.to_numeric(rows.get("std_z_m"), errors="coerce")
        rows["std_xy_m"] = rows[output_column]
        rows["std_z_m"] = rows[output_column] * float(z_scale)
    return CandidateFrame(normalize_candidate_columns(rows))


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
    **consensus_kwargs: Any,
) -> tuple[CandidateUncertaintyModel, pd.DataFrame, CandidateFrame]:
    """Fit uncertainty on train truth after adding consensus features."""

    augmented = attach_consensus_uncertainty_features(candidates, **consensus_kwargs)
    features = build_cluster_feature_table(
        augmented,
        truth=truth,
        max_truth_time_delta_s=float(max_truth_time_delta_s),
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
    return model, features, augmented


def consensus_uncertainty_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    output_column: str = "predicted_sigma_m",
    raw_output_column: str | None = None,
    factor_column: str = DEFAULT_SIGMA_FACTOR_COLUMN,
) -> dict[str, Any]:
    """Return compact diagnostics for a consensus-conditioned candidate table."""

    rows = candidates.rows if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates)
    raw_column = raw_output_column or f"raw_{output_column}"
    predicted = pd.to_numeric(rows.get(output_column), errors="coerce")
    raw = pd.to_numeric(rows.get(raw_column), errors="coerce")
    factor = pd.to_numeric(rows.get(factor_column), errors="coerce")
    finite_predicted = predicted[np.isfinite(predicted.to_numpy(float))]
    finite_raw = raw[np.isfinite(raw.to_numpy(float))]
    finite_factor = factor[np.isfinite(factor.to_numpy(float))]
    return {
        "row_count": int(len(rows)),
        "consensus": branch_consensus_summary(rows),
        "output_column": str(output_column),
        "raw_output_column": str(raw_column),
        "factor_column": str(factor_column),
        "predicted_sigma_mean_m": _safe_mean(finite_predicted),
        "raw_predicted_sigma_mean_m": _safe_mean(finite_raw),
        "consensus_factor_mean": _safe_mean(finite_factor),
        "consensus_factor_min": _safe_min(finite_factor),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-consensus-uncertainty",
        description=(
            "train or apply candidate uncertainty conditioned on truth-free "
            "cross-sensor branch consensus"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_consensus_arguments(train)
    train.add_argument("--candidates-csv", type=Path, required=True)
    train.add_argument("--truth-csv", type=Path, required=True)
    train.add_argument("--model-json", type=Path, required=True)
    train.add_argument("--features-csv", type=Path)
    train.add_argument("--augmented-candidates-csv", type=Path)
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

    apply_parser = subparsers.add_parser("apply")
    _add_consensus_arguments(apply_parser)
    apply_parser.add_argument("--candidates-csv", type=Path, required=True)
    apply_parser.add_argument("--model-json", type=Path, required=True)
    apply_parser.add_argument("--output-csv", type=Path, required=True)
    apply_parser.add_argument("--provenance-json", type=Path)
    apply_parser.add_argument("--output-column", default="predicted_sigma_m")
    apply_parser.add_argument("--replace-covariance", action="store_true")
    apply_parser.add_argument("--z-scale", type=float, default=1.0)
    apply_parser.add_argument("--consensus-score-column", default="branch_consensus_score")
    apply_parser.add_argument("--consensus-sigma-weight", type=float, default=0.0)
    apply_parser.add_argument("--consensus-sigma-min-factor", type=float, default=0.25)

    args = parser.parse_args(argv)
    consensus_kwargs = _consensus_kwargs(args)
    if args.command == "train":
        candidates = load_candidate_file(args.candidates_csv)
        truth = load_evaluation_truth_file(args.truth_csv).rows
        model, features, augmented = train_consensus_conditioned_uncertainty(
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
        if args.features_csv is not None:
            args.features_csv.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.features_csv, index=False)
        if args.augmented_candidates_csv is not None:
            args.augmented_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
            augmented.rows.to_csv(args.augmented_candidates_csv, index=False)
        summary = candidate_uncertainty_training_summary(features, model)
        summary["consensus"] = branch_consensus_summary(augmented)
        summary["consensus_feature_columns"] = [
            column for column in model.feature_columns if column.startswith("image_branch_consensus_")
        ]
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("mmuad_consensus_uncertainty_train=ok")
        print(f"model_json={args.model_json}")
        print(f"training_rows={summary.get('row_count', 0)}")
        print(f"consensus_feature_count={len(summary['consensus_feature_columns'])}")
        return 0

    model = load_candidate_uncertainty_model(args.model_json)
    scored = apply_consensus_conditioned_uncertainty(
        load_candidate_file(args.candidates_csv),
        model,
        output_column=args.output_column,
        replace_covariance=args.replace_covariance,
        z_scale=args.z_scale,
        consensus_score_column=args.consensus_score_column,
        consensus_sigma_weight=args.consensus_sigma_weight,
        consensus_sigma_min_factor=args.consensus_sigma_min_factor,
        **consensus_kwargs,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.rows.to_csv(args.output_csv, index=False)
    summary = consensus_uncertainty_summary(scored, output_column=args.output_column)
    if args.provenance_json is not None:
        args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("mmuad_consensus_uncertainty_apply=ok")
    print(f"output_csv={args.output_csv}")
    print(f"output_rows={len(scored.rows)}")
    return 0


def _add_consensus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--time-window-s", type=float, default=0.05)
    parser.add_argument("--time-scale-s", type=float)
    parser.add_argument("--distance-gate-m", type=float, default=5.0)
    parser.add_argument("--distance-scale-m", type=float, default=5.0)
    parser.add_argument("--base-score-column", default="ranker_score")
    parser.add_argument("--base-score-weight", type=float, default=1.0)
    parser.add_argument("--consensus-weight", type=float, default=1.0)
    parser.add_argument("--pair-advantage-weight", type=float, default=0.25)
    parser.add_argument("--branch-column")
    parser.add_argument("--origin-column")
    parser.add_argument("--allow-same-origin-support", action="store_true")


def _consensus_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "time_window_s": args.time_window_s,
        "time_scale_s": args.time_scale_s,
        "distance_gate_m": args.distance_gate_m,
        "distance_scale_m": args.distance_scale_m,
        "base_score_column": args.base_score_column,
        "base_score_weight": args.base_score_weight,
        "consensus_weight": args.consensus_weight,
        "pair_advantage_weight": args.pair_advantage_weight,
        "branch_column": args.branch_column,
        "origin_column": args.origin_column,
        "exclude_same_origin_support": not args.allow_same_origin_support,
    }


def _safe_mean(values: pd.Series) -> float | None:
    if len(values) == 0:
        return None
    return float(values.mean())


def _safe_min(values: pd.Series) -> float | None:
    if len(values) == 0:
        return None
    return float(values.min())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
