"""Train and apply an MMUAD cluster ranker with soft class context.

This module couples the sequence classifier and point-cloud candidate ranker
without hard-branching on a predicted UAV type.  Per-sequence class
probabilities are attached to every candidate as ``image_*`` features, including
probability-by-geometry interactions, and are then consumed by the existing
cluster-ranker model.

For a non-leaky train-to-validation experiment, provide out-of-fold class
probabilities for training sequences and train-only classifier predictions for
validation/test sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from raft_uav.mmuad.class_probability_context import (
    DEFAULT_INTERACTION_COLUMNS,
    attach_class_probability_context,
)
from raft_uav.mmuad.cluster_ranker import (
    ClusterRankerModel,
    build_cluster_feature_table,
    load_cluster_ranker_model,
    save_cluster_ranker_model,
    score_cluster_candidates,
    train_cluster_ranker,
    write_ranker_diagnostics,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame

MODEL_TYPES = (
    "logistic",
    "sklearn-logistic",
    "random-forest-classifier",
    "hist-gradient-boosting-classifier",
    "random-forest-regressor",
    "hist-gradient-boosting-regressor",
)


def train_class_conditioned_ranker(
    *,
    candidate_csv: Path,
    truth_file: Path,
    class_probabilities_csv: Path,
    model_json: Path,
    context_candidates_csv: Path | None = None,
    train_features_csv: Path | None = None,
    model_type: str = "logistic",
    target_column: str = "good_cluster",
    interaction_columns: Iterable[str] = DEFAULT_INTERACTION_COLUMNS,
    fill_missing: str = "uniform",
    good_threshold_m: float = 5.0,
    max_truth_time_delta_s: float = 0.5,
    learning_rate: float = 0.05,
    iterations: int = 600,
    random_state: int = 13,
    n_estimators: int = 200,
    score_distance_scale_m: float | None = None,
) -> tuple[ClusterRankerModel, pd.DataFrame]:
    """Train a cluster ranker using soft sequence-class context."""

    candidates = load_candidate_file(candidate_csv)
    probabilities = pd.read_csv(class_probabilities_csv)
    contextual = attach_class_probability_context(
        candidates,
        probabilities,
        interaction_columns=tuple(interaction_columns),
        fill_missing=fill_missing,
    )
    if context_candidates_csv is not None:
        _write_frame(contextual.rows, context_candidates_csv)
    truth = load_evaluation_truth_file(truth_file).rows
    features = build_cluster_feature_table(
        contextual,
        truth=truth,
        good_threshold_m=good_threshold_m,
        max_truth_time_delta_s=max_truth_time_delta_s,
    )
    model = train_cluster_ranker(
        features,
        model_type=model_type,
        target_column=target_column,
        learning_rate=learning_rate,
        iterations=iterations,
        random_state=random_state,
        n_estimators=n_estimators,
        score_distance_scale_m=(
            good_threshold_m if score_distance_scale_m is None else score_distance_scale_m
        ),
    )
    save_cluster_ranker_model(model, model_json)
    if train_features_csv is not None:
        write_ranker_diagnostics(features, train_features_csv)
    return model, features


def score_class_conditioned_candidates(
    *,
    candidate_csv: Path,
    class_probabilities_csv: Path,
    model: ClusterRankerModel,
    scored_candidates_csv: Path,
    context_candidates_csv: Path | None = None,
    score_features_csv: Path | None = None,
    interaction_columns: Iterable[str] = DEFAULT_INTERACTION_COLUMNS,
    fill_missing: str = "uniform",
) -> CandidateFrame:
    """Apply a trained cluster ranker to class-conditioned candidates."""

    candidates = load_candidate_file(candidate_csv)
    probabilities = pd.read_csv(class_probabilities_csv)
    contextual = attach_class_probability_context(
        candidates,
        probabilities,
        interaction_columns=tuple(interaction_columns),
        fill_missing=fill_missing,
    )
    if context_candidates_csv is not None:
        _write_frame(contextual.rows, context_candidates_csv)
    scored = score_cluster_candidates(contextual, model)
    _write_frame(scored.rows, scored_candidates_csv)
    if score_features_csv is not None:
        write_ranker_diagnostics(scored.rows, score_features_csv)
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-class-conditioned-ranker",
        description="train/apply an MMUAD cluster ranker with soft class probabilities",
    )
    parser.add_argument("--train-candidates", type=Path)
    parser.add_argument("--train-truth", type=Path)
    parser.add_argument("--train-class-probabilities-csv", type=Path)
    parser.add_argument("--score-candidates", type=Path)
    parser.add_argument("--score-class-probabilities-csv", type=Path)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--train-context-candidates-csv", type=Path)
    parser.add_argument("--score-context-candidates-csv", type=Path)
    parser.add_argument("--train-features-csv", type=Path)
    parser.add_argument("--score-features-csv", type=Path)
    parser.add_argument("--scored-candidates-csv", type=Path)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="logistic")
    parser.add_argument("--target-column", default="good_cluster")
    parser.add_argument("--good-threshold-m", type=float, default=5.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--score-distance-scale-m", type=float)
    parser.add_argument(
        "--interaction-column",
        action="append",
        default=[],
        help="candidate numeric column to interact with class probabilities; may be repeated",
    )
    parser.add_argument(
        "--fill-missing",
        choices=("uniform", "zero", "error"),
        default="uniform",
    )
    parser.add_argument("--provenance-json", type=Path)
    args = parser.parse_args(argv)

    interaction_columns = tuple(args.interaction_column) or DEFAULT_INTERACTION_COLUMNS
    training_requested = args.train_candidates is not None
    scoring_requested = args.score_candidates is not None
    if not training_requested and not scoring_requested:
        raise SystemExit("provide --train-candidates and/or --score-candidates")

    model: ClusterRankerModel
    train_features: pd.DataFrame | None = None
    if training_requested:
        if args.train_truth is None or args.train_class_probabilities_csv is None:
            raise SystemExit(
                "--train-candidates requires --train-truth and "
                "--train-class-probabilities-csv"
            )
        model, train_features = train_class_conditioned_ranker(
            candidate_csv=args.train_candidates,
            truth_file=args.train_truth,
            class_probabilities_csv=args.train_class_probabilities_csv,
            model_json=args.model_json,
            context_candidates_csv=args.train_context_candidates_csv,
            train_features_csv=args.train_features_csv,
            model_type=args.model_type,
            target_column=args.target_column,
            interaction_columns=interaction_columns,
            fill_missing=args.fill_missing,
            good_threshold_m=args.good_threshold_m,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            score_distance_scale_m=args.score_distance_scale_m,
        )
    else:
        model = load_cluster_ranker_model(args.model_json)

    scored: CandidateFrame | None = None
    if scoring_requested:
        if args.score_class_probabilities_csv is None or args.scored_candidates_csv is None:
            raise SystemExit(
                "--score-candidates requires --score-class-probabilities-csv and "
                "--scored-candidates-csv"
            )
        scored = score_class_conditioned_candidates(
            candidate_csv=args.score_candidates,
            class_probabilities_csv=args.score_class_probabilities_csv,
            model=model,
            scored_candidates_csv=args.scored_candidates_csv,
            context_candidates_csv=args.score_context_candidates_csv,
            score_features_csv=args.score_features_csv,
            interaction_columns=interaction_columns,
            fill_missing=args.fill_missing,
        )

    if args.provenance_json is not None:
        payload: dict[str, Any] = {
            "protocol": "soft sequence-class probabilities as candidate-ranker context",
            "model_json": str(args.model_json),
            "model_type": model.model_type,
            "target_column": model.target_column,
            "feature_columns": model.feature_columns,
            "class_context_feature_columns": [
                column for column in model.feature_columns if column.startswith("image_class_")
            ],
            "interaction_columns": list(interaction_columns),
            "fill_missing": str(args.fill_missing),
            "train_candidates": _optional_path(args.train_candidates),
            "train_class_probabilities_csv": _optional_path(
                args.train_class_probabilities_csv
            ),
            "score_candidates": _optional_path(args.score_candidates),
            "score_class_probabilities_csv": _optional_path(
                args.score_class_probabilities_csv
            ),
            "train_rows": None if train_features is None else int(len(train_features)),
            "score_rows": None if scored is None else int(len(scored.rows)),
        }
        args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if training_requested:
        print("mmuad_class_conditioned_ranker_train=ok")
        print(f"model_json={args.model_json}")
        print(f"train_rows={len(train_features) if train_features is not None else 0}")
    if scoring_requested:
        print("mmuad_class_conditioned_ranker_score=ok")
        print(f"scored_candidates_csv={args.scored_candidates_csv}")
        print(f"score_rows={len(scored.rows) if scored is not None else 0}")
    return 0


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _optional_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
