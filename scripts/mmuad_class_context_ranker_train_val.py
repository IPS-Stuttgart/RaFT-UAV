#!/usr/bin/env python
"""Train and apply an MMUAD cluster ranker with class-probability context.

This runner composes the sequence-level fused classifier with the existing
candidate ranker.  It attaches per-sequence class probabilities as soft
``image_*`` candidate features, trains a cluster ranker on train truth, and
scores a second candidate table.  The intent is to let UAV type evidence inform
pose candidate reliability without hard-branching on a predicted class label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.class_probability_context import (  # noqa: E402
    DEFAULT_INTERACTION_COLUMNS,
    attach_class_probability_context,
)
from raft_uav.mmuad.cluster_ranker import (  # noqa: E402
    build_cluster_feature_table,
    predict_cluster_scores,
    save_cluster_ranker_model,
    score_cluster_candidates,
    train_cluster_ranker,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file  # noqa: E402
from raft_uav.mmuad.io import load_candidate_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--score-candidates", type=Path, required=True)
    parser.add_argument("--train-truth", type=Path, required=True)
    parser.add_argument("--train-class-probabilities-csv", type=Path, required=True)
    parser.add_argument("--score-class-probabilities-csv", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--scored-candidates-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--train-context-candidates-csv", type=Path)
    parser.add_argument("--score-context-candidates-csv", type=Path)
    parser.add_argument("--train-features-csv", type=Path)
    parser.add_argument("--score-features-csv", type=Path)
    parser.add_argument("--model-type", default="random-forest-classifier")
    parser.add_argument("--target-column", default="good_cluster_5m")
    parser.add_argument("--good-threshold-m", type=float, default=5.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--score-distance-scale-m", type=float, default=10.0)
    parser.add_argument(
        "--interaction-column",
        action="append",
        default=[],
        help="candidate numeric column to multiply by each class probability; may be repeated",
    )
    parser.add_argument(
        "--fill-missing-class-probabilities",
        choices=("uniform", "zero", "error"),
        default="uniform",
    )
    args = parser.parse_args(argv)

    interaction_columns = tuple(args.interaction_column) or DEFAULT_INTERACTION_COLUMNS
    train_context = attach_class_probability_context(
        load_candidate_file(args.train_candidates),
        pd.read_csv(args.train_class_probabilities_csv),
        interaction_columns=interaction_columns,
        fill_missing=args.fill_missing_class_probabilities,
    )
    score_context = attach_class_probability_context(
        load_candidate_file(args.score_candidates),
        pd.read_csv(args.score_class_probabilities_csv),
        interaction_columns=interaction_columns,
        fill_missing=args.fill_missing_class_probabilities,
    )
    _maybe_write_frame(train_context.rows, args.train_context_candidates_csv)
    _maybe_write_frame(score_context.rows, args.score_context_candidates_csv)

    truth = load_evaluation_truth_file(args.train_truth).rows
    train_features = build_cluster_feature_table(
        train_context,
        truth=truth,
        good_threshold_m=float(args.good_threshold_m),
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
    )
    _maybe_write_frame(train_features, args.train_features_csv)

    model = train_cluster_ranker(
        train_features,
        model_type=str(args.model_type),
        target_column=str(args.target_column),
        random_state=int(args.random_state),
        n_estimators=int(args.n_estimators),
        score_distance_scale_m=float(args.score_distance_scale_m),
    )
    save_cluster_ranker_model(model, args.model_json)

    scored = score_cluster_candidates(score_context, model)
    _write_frame(scored.rows, args.scored_candidates_csv)
    score_features = build_cluster_feature_table(score_context)
    score_features["ranker_score"] = predict_cluster_scores(score_features, model)
    _maybe_write_frame(score_features, args.score_features_csv)

    summary = {
        "train_candidates": str(args.train_candidates),
        "score_candidates": str(args.score_candidates),
        "train_class_probabilities_csv": str(args.train_class_probabilities_csv),
        "score_class_probabilities_csv": str(args.score_class_probabilities_csv),
        "model_json": str(args.model_json),
        "scored_candidates_csv": str(args.scored_candidates_csv),
        "model_type": model.model_type,
        "target_column": model.target_column,
        "feature_count": len(model.feature_columns),
        "class_context_feature_count": int(
            sum(str(column).startswith("image_class_") for column in model.feature_columns)
        ),
        "interaction_columns": list(interaction_columns),
        "train_context_rows": int(len(train_context.rows)),
        "score_context_rows": int(len(score_context.rows)),
        "train_feature_rows": int(len(train_features)),
        "scored_rows": int(len(scored.rows)),
        "feature_columns": model.feature_columns,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print("mmuad_class_context_ranker_train_val=ok")
    print(f"model_json={args.model_json}")
    print(f"scored_candidates_csv={args.scored_candidates_csv}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    print(f"class_context_feature_count={summary['class_context_feature_count']}")
    return 0


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _maybe_write_frame(frame: pd.DataFrame, path: Path | None) -> None:
    if path is not None:
        _write_frame(frame, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
