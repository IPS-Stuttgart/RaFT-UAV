"""Branch-aware, class-conditioned candidate uncertainty for MMUAD.

The existing learned-sigma model consumes numeric candidate features, but a
branch-preserving reservoir can still present candidates from raw, dynamic,
source-translated, calibrated, and merged branches with very different error
statistics.  This module adds stable semantic branch features, translation
magnitude diagnostics, within-branch score ranks, and soft UAV-class
interactions before fitting or applying the maintained candidate-uncertainty
model.

All fitting uses training truth only.  Application requires candidates and
sequence-level class probabilities, but no validation/test truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_uncertainty import (
    CandidateUncertaintyModel,
    apply_candidate_uncertainty,
    candidate_uncertainty_training_summary,
    load_candidate_uncertainty_model,
    save_candidate_uncertainty_model,
    train_candidate_uncertainty,
)
from raft_uav.mmuad.class_probability_context import (
    DEFAULT_INTERACTION_COLUMNS,
    attach_class_probability_context,
)
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

DEFAULT_SCORE_COLUMNS = (
    "candidate_pair_forward_backward_score",
    "candidate_forward_backward_score",
    "candidate_reservoir_grid_score",
    "candidate_risk_adjusted_score",
    "ranker_score",
    "confidence",
)

BRANCH_CONTEXT_COLUMNS = (
    "candidate_reservoir_branch_is_raw",
    "candidate_reservoir_branch_is_static",
    "candidate_reservoir_branch_is_dynamic",
    "candidate_reservoir_branch_is_translated",
    "candidate_reservoir_branch_is_calibrated",
    "candidate_reservoir_branch_is_merged",
    "candidate_reservoir_translation_dx_m",
    "candidate_reservoir_translation_dy_m",
    "candidate_reservoir_translation_dz_m",
    "candidate_reservoir_translation_distance_m",
    "candidate_reservoir_frame_branch_count",
    "candidate_reservoir_branch_candidate_count",
    "candidate_reservoir_branch_fraction",
    "candidate_reservoir_source_branch_candidate_count",
    "candidate_reservoir_source_branch_fraction",
    "candidate_reservoir_branch_score_rank",
    "candidate_reservoir_branch_score_gap",
    "candidate_reservoir_source_branch_score_rank",
    "candidate_reservoir_source_branch_score_gap",
)

_ORIGINAL_COORDINATE_SETS = (
    ("original_x_m", "original_y_m", "original_z_m"),
    ("raw_x_m", "raw_y_m", "raw_z_m"),
    ("uncalibrated_x_m", "uncalibrated_y_m", "uncalibrated_z_m"),
)


def attach_branch_uncertainty_context(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    score_columns: Sequence[str] = DEFAULT_SCORE_COLUMNS,
) -> CandidateFrame:
    """Attach inference-safe branch and translation features to candidates."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)
    out = rows.copy().reset_index(drop=True)
    branch = _branch_values(out)
    source = out.get("source", pd.Series("candidate", index=out.index)).fillna("candidate").astype(str)
    out["candidate_branch"] = branch

    lowered = branch.str.lower()
    out["candidate_reservoir_branch_is_raw"] = lowered.str.contains("raw", regex=False).astype(float)
    out["candidate_reservoir_branch_is_static"] = lowered.str.contains("static", regex=False).astype(float)
    out["candidate_reservoir_branch_is_dynamic"] = lowered.str.contains("dynamic", regex=False).astype(float)
    out["candidate_reservoir_branch_is_translated"] = lowered.str.contains(
        "translat", regex=False
    ).astype(float)
    out["candidate_reservoir_branch_is_calibrated"] = lowered.str.contains(
        "calibrat", regex=False
    ).astype(float)
    out["candidate_reservoir_branch_is_merged"] = (
        lowered.str.contains("merged", regex=False)
        | lowered.str.contains("fusion", regex=False)
        | lowered.str.contains("consensus", regex=False)
    ).astype(float)

    _add_translation_features(out)
    out["candidate_reservoir_score_for_context"] = _coalesced_numeric(out, score_columns)
    out["candidate_reservoir_frame_branch_count"] = 0.0
    out["candidate_reservoir_branch_candidate_count"] = 0.0
    out["candidate_reservoir_branch_fraction"] = 0.0
    out["candidate_reservoir_source_branch_candidate_count"] = 0.0
    out["candidate_reservoir_source_branch_fraction"] = 0.0
    out["candidate_reservoir_branch_score_rank"] = np.nan
    out["candidate_reservoir_branch_score_gap"] = np.nan
    out["candidate_reservoir_source_branch_score_rank"] = np.nan
    out["candidate_reservoir_source_branch_score_gap"] = np.nan

    group_columns = ["sequence_id", "time_s"]
    for _, frame in out.groupby(group_columns, sort=False, dropna=False):
        frame_index = frame.index
        frame_count = max(int(len(frame)), 1)
        out.loc[frame_index, "candidate_reservoir_frame_branch_count"] = float(
            frame["candidate_branch"].astype(str).nunique()
        )
        for _, branch_rows in frame.groupby("candidate_branch", sort=False, dropna=False):
            _assign_group_score_features(
                out,
                branch_rows,
                count_column="candidate_reservoir_branch_candidate_count",
                fraction_column="candidate_reservoir_branch_fraction",
                rank_column="candidate_reservoir_branch_score_rank",
                gap_column="candidate_reservoir_branch_score_gap",
                denominator=frame_count,
            )
        frame_with_source = frame.assign(_context_source=source.loc[frame_index].to_numpy())
        for _, source_branch_rows in frame_with_source.groupby(
            ["_context_source", "candidate_branch"],
            sort=False,
            dropna=False,
        ):
            _assign_group_score_features(
                out,
                source_branch_rows,
                count_column="candidate_reservoir_source_branch_candidate_count",
                fraction_column="candidate_reservoir_source_branch_fraction",
                rank_column="candidate_reservoir_source_branch_score_rank",
                gap_column="candidate_reservoir_source_branch_score_gap",
                denominator=frame_count,
            )

    return CandidateFrame(normalize_candidate_columns(out))


def attach_branch_class_uncertainty_context(
    candidates: CandidateFrame | pd.DataFrame,
    class_probabilities: pd.DataFrame,
    *,
    score_columns: Sequence[str] = DEFAULT_SCORE_COLUMNS,
    fill_missing: str = "uniform",
    extra_interaction_columns: Iterable[str] = (),
) -> CandidateFrame:
    """Attach branch features and their soft class-probability interactions."""

    contextual = attach_branch_uncertainty_context(candidates, score_columns=score_columns)
    interactions = tuple(
        dict.fromkeys(
            (
                *DEFAULT_INTERACTION_COLUMNS,
                *BRANCH_CONTEXT_COLUMNS,
                *tuple(str(value) for value in extra_interaction_columns),
            )
        )
    )
    return attach_class_probability_context(
        contextual,
        pd.DataFrame(class_probabilities),
        interaction_columns=interactions,
        fill_missing=fill_missing,
    )


def train_branch_aware_candidate_uncertainty(
    candidates: CandidateFrame | pd.DataFrame,
    truth: pd.DataFrame,
    class_probabilities: pd.DataFrame,
    *,
    score_columns: Sequence[str] = DEFAULT_SCORE_COLUMNS,
    fill_missing: str = "uniform",
    model_type: str = "hist-gradient-boosting",
    target_transform: str = "log1p",
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 30.0,
    ridge_alpha: float = 1.0,
    random_state: int = 13,
    n_estimators: int = 300,
    max_truth_time_delta_s: float = 0.5,
) -> tuple[CandidateUncertaintyModel, pd.DataFrame, dict[str, Any]]:
    """Fit learned sigma after adding branch/class context on training data."""

    contextual = attach_branch_class_uncertainty_context(
        candidates,
        class_probabilities,
        score_columns=score_columns,
        fill_missing=fill_missing,
    )
    features = build_cluster_feature_table(
        contextual,
        truth=pd.DataFrame(truth),
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
    summary = candidate_uncertainty_training_summary(features, model)
    summary.update(
        {
            "protocol": "train-only branch-aware class-conditioned candidate uncertainty",
            "branch_context_columns": [
                column for column in BRANCH_CONTEXT_COLUMNS if column in features.columns
            ],
            "branch_class_interaction_columns": [
                column
                for column in model.feature_columns
                if column.startswith("image_class_prob_")
                and "_x_candidate_reservoir_" in column
            ],
            "score_columns": list(score_columns),
            "fill_missing_class_probabilities": str(fill_missing),
        }
    )
    return model, features, summary


def apply_branch_aware_candidate_uncertainty(
    candidates: CandidateFrame | pd.DataFrame,
    model: CandidateUncertaintyModel,
    class_probabilities: pd.DataFrame,
    *,
    score_columns: Sequence[str] = DEFAULT_SCORE_COLUMNS,
    fill_missing: str = "uniform",
    output_column: str = "predicted_sigma_m_branch_class",
    replace_covariance: bool = False,
    z_scale: float = 1.0,
) -> CandidateFrame:
    """Apply a branch-aware uncertainty model without target truth."""

    contextual = attach_branch_class_uncertainty_context(
        candidates,
        class_probabilities,
        score_columns=score_columns,
        fill_missing=fill_missing,
    )
    return apply_candidate_uncertainty(
        contextual,
        model,
        output_column=output_column,
        replace_covariance=replace_covariance,
        z_scale=z_scale,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.candidate_branch_uncertainty",
        description="train/apply branch-aware class-conditioned MMUAD candidate uncertainty",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--candidates-csv", type=Path, required=True)
    train.add_argument("--truth-csv", type=Path, required=True)
    train.add_argument("--class-probabilities-csv", type=Path, required=True)
    train.add_argument("--model-json", type=Path, required=True)
    train.add_argument("--features-csv", type=Path)
    train.add_argument("--summary-json", type=Path)
    train.add_argument("--score-column", action="append", default=[])
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
    train.add_argument("--fill-missing", choices=("uniform", "zero", "error"), default="uniform")

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--candidates-csv", type=Path, required=True)
    apply_parser.add_argument("--class-probabilities-csv", type=Path, required=True)
    apply_parser.add_argument("--model-json", type=Path, required=True)
    apply_parser.add_argument("--output-csv", type=Path, required=True)
    apply_parser.add_argument("--score-column", action="append", default=[])
    apply_parser.add_argument("--output-column", default="predicted_sigma_m_branch_class")
    apply_parser.add_argument("--replace-covariance", action="store_true")
    apply_parser.add_argument("--z-scale", type=float, default=1.0)
    apply_parser.add_argument("--fill-missing", choices=("uniform", "zero", "error"), default="uniform")

    args = parser.parse_args(argv)
    score_columns = tuple(args.score_column) or DEFAULT_SCORE_COLUMNS
    class_probabilities = pd.read_csv(args.class_probabilities_csv, dtype=str)

    if args.command == "train":
        candidates = load_candidate_file(args.candidates_csv)
        truth = load_evaluation_truth_file(args.truth_csv).rows
        model, features, summary = train_branch_aware_candidate_uncertainty(
            candidates,
            truth,
            class_probabilities,
            score_columns=score_columns,
            fill_missing=args.fill_missing,
            model_type=args.model_type,
            target_transform=args.target_transform,
            sigma_min_m=args.sigma_min_m,
            sigma_max_m=args.sigma_max_m,
            ridge_alpha=args.ridge_alpha,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        save_candidate_uncertainty_model(model, args.model_json)
        if args.features_csv is not None:
            args.features_csv.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.features_csv, index=False)
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        print("mmuad_branch_aware_candidate_uncertainty_train=ok")
        print(f"model_json={args.model_json}")
        return 0

    model = load_candidate_uncertainty_model(args.model_json)
    scored = apply_branch_aware_candidate_uncertainty(
        load_candidate_file(args.candidates_csv),
        model,
        class_probabilities,
        score_columns=score_columns,
        fill_missing=args.fill_missing,
        output_column=args.output_column,
        replace_covariance=args.replace_covariance,
        z_scale=args.z_scale,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.rows.to_csv(args.output_csv, index=False)
    print("mmuad_branch_aware_candidate_uncertainty_apply=ok")
    print(f"output_csv={args.output_csv}")
    print(f"output_rows={len(scored.rows)}")
    return 0


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates).copy()
    rows = normalize_candidate_columns(rows)
    if not rows.empty:
        rows["sequence_id"] = rows["sequence_id"].astype(str)
    return rows


def _branch_values(rows: pd.DataFrame) -> pd.Series:
    if "candidate_branch" in rows.columns:
        branch = rows["candidate_branch"].fillna("").astype(str).str.strip()
    else:
        branch = pd.Series("", index=rows.index, dtype=str)
    source = rows.get("source", pd.Series("candidate", index=rows.index)).fillna("candidate").astype(str)
    return branch.where(branch.str.len() > 0, source)


def _add_translation_features(rows: pd.DataFrame) -> None:
    original = _original_coordinate_columns(rows)
    if original is None:
        dx = np.zeros(len(rows), dtype=float)
        dy = np.zeros(len(rows), dtype=float)
        dz = np.zeros(len(rows), dtype=float)
    else:
        current = rows[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        baseline = rows[list(original)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        displacement = current - baseline
        displacement = np.where(np.isfinite(displacement), displacement, 0.0)
        dx, dy, dz = displacement.T
    rows["candidate_reservoir_translation_dx_m"] = dx
    rows["candidate_reservoir_translation_dy_m"] = dy
    rows["candidate_reservoir_translation_dz_m"] = dz
    rows["candidate_reservoir_translation_distance_m"] = np.sqrt(dx**2 + dy**2 + dz**2)


def _original_coordinate_columns(rows: pd.DataFrame) -> tuple[str, str, str] | None:
    for columns in _ORIGINAL_COORDINATE_SETS:
        if set(columns).issubset(rows.columns):
            return columns
    return None


def _coalesced_numeric(rows: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    score = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        score = score.where(score.notna(), values)
    return score.fillna(0.0)


def _assign_group_score_features(
    out: pd.DataFrame,
    group: pd.DataFrame,
    *,
    count_column: str,
    fraction_column: str,
    rank_column: str,
    gap_column: str,
    denominator: int,
) -> None:
    indices = group.index
    scores = pd.to_numeric(
        out.loc[indices, "candidate_reservoir_score_for_context"], errors="coerce"
    ).fillna(0.0)
    count = int(len(group))
    top_score = float(scores.max()) if count else 0.0
    out.loc[indices, count_column] = float(count)
    out.loc[indices, fraction_column] = float(count / max(int(denominator), 1))
    out.loc[indices, rank_column] = scores.rank(method="average", ascending=False).to_numpy(float)
    out.loc[indices, gap_column] = top_score - scores.to_numpy(float)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
