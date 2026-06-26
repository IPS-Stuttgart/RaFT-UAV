"""Train-only class-conditioned calibration for MMUAD candidate scores.

The MMUAD pose pipeline already exposes strong sequence-level UAV type
probabilities, but candidate reservoirs still rely on a single generic ranker
score.  This module learns additive logit corrections for candidate branches
and sensor sources, conditioned softly on class probabilities.  The model is
fit from training candidates, training truth, and out-of-fold training class
probabilities, then applied to validation/test without truth labels.

The calibrated score is intended as the ``--score-column`` for the existing
branch-preserving candidate reservoir and robust candidate-mixture smoother.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_context import (
    OFFICIAL_CLASS_LABELS,
    attach_class_probability_context,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns, normalize_truth_columns

MODEL_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_SCORE_COLUMN = "candidate_class_calibrated_score"
_EPS = 1.0e-6


def fit_candidate_score_calibration(
    candidates: CandidateFrame | pd.DataFrame,
    truth: pd.DataFrame,
    class_probabilities: pd.DataFrame,
    *,
    score_column: str = "ranker_score",
    fallback_score_column: str = "confidence",
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
    good_threshold_m: float = 5.0,
    max_truth_time_delta_s: float = 0.5,
    min_group_weight: float = 20.0,
    l2_penalty: float = 10.0,
    max_abs_logit_offset: float = 4.0,
    include_branch_source_interactions: bool = True,
    fill_missing_class_probabilities: str = "uniform",
) -> tuple[dict[str, Any], CandidateFrame, pd.DataFrame]:
    """Fit a soft class-conditioned branch/source score calibration model.

    Training class probabilities should be out-of-fold predictions.  Ground
    truth is used only to create the binary candidate-goodness target.
    """

    contextual = attach_class_probability_context(
        _candidate_frame(candidates),
        pd.DataFrame(class_probabilities),
        fill_missing=fill_missing_class_probabilities,
    )
    rows = _prepare_candidate_context(contextual.rows)
    labelled = _attach_truth_targets(
        rows,
        normalize_truth_columns(pd.DataFrame(truth).copy()),
        good_threshold_m=float(good_threshold_m),
        max_truth_time_delta_s=float(max_truth_time_delta_s),
    )
    matched = labelled.loc[labelled["candidate_truth_matched"]].copy()
    if matched.empty:
        raise ValueError("no candidate rows matched training truth within the requested tolerance")

    transform = _infer_score_transform(matched, score_column, fallback_score_column)
    base_probability = _base_probability(
        matched,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        transform=transform,
    )
    base_logit = _logit(base_probability)
    target = matched["candidate_good_target"].to_numpy(float)
    unit_weights = np.ones(len(matched), dtype=float)

    global_offset = _fit_logit_offset(
        base_logit,
        target,
        unit_weights,
        l2_penalty=float(l2_penalty),
        max_abs_offset=float(max_abs_logit_offset),
    )
    class_columns = [f"image_class_prob_{label}" for label in OFFICIAL_CLASS_LABELS]

    diagnostics: list[dict[str, Any]] = []
    branch_offsets = _fit_group_class_offsets(
        matched,
        base_logit + global_offset,
        target,
        group_column="candidate_branch",
        class_columns=class_columns,
        min_group_weight=float(min_group_weight),
        l2_penalty=float(l2_penalty),
        max_abs_logit_offset=float(max_abs_logit_offset),
        level="branch",
        diagnostics=diagnostics,
    )
    branch_contribution = _soft_group_contribution(
        matched,
        group_column="candidate_branch",
        offsets=branch_offsets,
        class_columns=class_columns,
    )
    source_offsets = _fit_group_class_offsets(
        matched,
        base_logit + global_offset + branch_contribution,
        target,
        group_column="source",
        class_columns=class_columns,
        min_group_weight=float(min_group_weight),
        l2_penalty=float(l2_penalty),
        max_abs_logit_offset=float(max_abs_logit_offset),
        level="source",
        diagnostics=diagnostics,
    )
    source_contribution = _soft_group_contribution(
        matched,
        group_column="source",
        offsets=source_offsets,
        class_columns=class_columns,
    )

    interaction_offsets: dict[str, dict[str, float]] = {}
    if include_branch_source_interactions:
        matched = matched.copy()
        matched["candidate_branch_source"] = (
            matched["candidate_branch"].astype(str) + "||" + matched["source"].astype(str)
        )
        interaction_offsets = _fit_group_class_offsets(
            matched,
            base_logit + global_offset + branch_contribution + source_contribution,
            target,
            group_column="candidate_branch_source",
            class_columns=class_columns,
            min_group_weight=float(min_group_weight),
            l2_penalty=float(l2_penalty),
            max_abs_logit_offset=float(max_abs_logit_offset),
            level="branch_source",
            diagnostics=diagnostics,
        )

    model: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "protocol": "train-only soft-class-conditioned candidate-score logit calibration",
        "score_column": str(score_column),
        "fallback_score_column": str(fallback_score_column),
        "output_score_column": str(output_score_column),
        "score_transform": transform,
        "good_threshold_m": float(good_threshold_m),
        "max_truth_time_delta_s": float(max_truth_time_delta_s),
        "min_group_weight": float(min_group_weight),
        "l2_penalty": float(l2_penalty),
        "max_abs_logit_offset": float(max_abs_logit_offset),
        "include_branch_source_interactions": bool(include_branch_source_interactions),
        "class_labels": list(OFFICIAL_CLASS_LABELS),
        "global_logit_offset": float(global_offset),
        "branch_class_logit_offsets": branch_offsets,
        "source_class_logit_offsets": source_offsets,
        "branch_source_class_logit_offsets": interaction_offsets,
        "fit_candidate_rows": int(len(labelled)),
        "fit_matched_rows": int(len(matched)),
        "fit_positive_rate": float(np.mean(target)),
        "fill_missing_class_probabilities": str(fill_missing_class_probabilities),
    }
    calibrated = apply_candidate_score_calibration(
        contextual,
        model,
        class_probabilities=None,
        fill_missing_class_probabilities=fill_missing_class_probabilities,
    )
    matched_calibrated = calibrated.rows.loc[labelled["candidate_truth_matched"].to_numpy(bool)].copy()
    diagnostics_frame = pd.DataFrame.from_records(
        [
            {
                "level": "global",
                "group": "__global__",
                "class_label": "__all__",
                "effective_weight": float(len(matched)),
                "positive_rate": float(np.mean(target)),
                "logit_offset": float(global_offset),
            },
            *diagnostics,
            {
                "level": "summary",
                "group": "__all__",
                "class_label": "__all__",
                "effective_weight": float(len(matched)),
                "positive_rate": float(np.mean(target)),
                "base_brier": float(np.mean((base_probability - target) ** 2)),
                "calibrated_brier": float(
                    np.mean(
                        (
                            pd.to_numeric(
                                matched_calibrated[output_score_column], errors="coerce"
                            ).to_numpy(float)
                            - target
                        )
                        ** 2
                    )
                ),
                "logit_offset": float("nan"),
            },
        ]
    )
    return model, calibrated, diagnostics_frame


def apply_candidate_score_calibration(
    candidates: CandidateFrame | pd.DataFrame,
    model: Mapping[str, Any],
    *,
    class_probabilities: pd.DataFrame | None,
    fill_missing_class_probabilities: str | None = None,
) -> CandidateFrame:
    """Apply a fitted candidate-score calibration without truth labels."""

    _validate_model(model)
    if class_probabilities is None:
        rows = _candidate_frame(candidates).rows.copy()
        required = [f"image_class_prob_{label}" for label in model["class_labels"]]
        if not set(required).issubset(rows.columns):
            raise ValueError(
                "class_probabilities is required unless candidate rows already contain "
                "image_class_prob_0..3 columns"
            )
        contextual = CandidateFrame(normalize_candidate_columns(rows))
    else:
        contextual = attach_class_probability_context(
            _candidate_frame(candidates),
            pd.DataFrame(class_probabilities),
            fill_missing=(
                str(fill_missing_class_probabilities)
                if fill_missing_class_probabilities is not None
                else str(model.get("fill_missing_class_probabilities", "uniform"))
            ),
        )
    rows = _prepare_candidate_context(contextual.rows)
    base_probability = _base_probability(
        rows,
        score_column=str(model["score_column"]),
        fallback_score_column=str(model["fallback_score_column"]),
        transform=str(model["score_transform"]),
    )
    class_columns = [f"image_class_prob_{label}" for label in model["class_labels"]]
    branch_contribution = _soft_group_contribution(
        rows,
        group_column="candidate_branch",
        offsets=_nested_float_dict(model.get("branch_class_logit_offsets", {})),
        class_columns=class_columns,
    )
    source_contribution = _soft_group_contribution(
        rows,
        group_column="source",
        offsets=_nested_float_dict(model.get("source_class_logit_offsets", {})),
        class_columns=class_columns,
    )
    interaction_contribution = np.zeros(len(rows), dtype=float)
    if bool(model.get("include_branch_source_interactions", False)):
        interaction_rows = rows.copy()
        interaction_rows["candidate_branch_source"] = (
            interaction_rows["candidate_branch"].astype(str)
            + "||"
            + interaction_rows["source"].astype(str)
        )
        interaction_contribution = _soft_group_contribution(
            interaction_rows,
            group_column="candidate_branch_source",
            offsets=_nested_float_dict(model.get("branch_source_class_logit_offsets", {})),
            class_columns=class_columns,
        )
    global_offset = float(model.get("global_logit_offset", 0.0))
    calibrated_logit = (
        _logit(base_probability)
        + global_offset
        + branch_contribution
        + source_contribution
        + interaction_contribution
    )
    output_column = str(model.get("output_score_column", DEFAULT_OUTPUT_SCORE_COLUMN))
    out = rows.copy()
    out["candidate_class_calibration_base_probability"] = base_probability
    out["candidate_class_calibration_global_logit_offset"] = global_offset
    out["candidate_class_calibration_branch_logit_offset"] = branch_contribution
    out["candidate_class_calibration_source_logit_offset"] = source_contribution
    out["candidate_class_calibration_interaction_logit_offset"] = interaction_contribution
    out[output_column] = _sigmoid(calibrated_logit)
    return CandidateFrame(normalize_candidate_columns(out))


def save_candidate_score_calibration_model(model: Mapping[str, Any], path: Path) -> None:
    """Write a fitted calibration model as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(model), indent=2), encoding="utf-8")


def load_candidate_score_calibration_model(path: Path) -> dict[str, Any]:
    """Load and validate a fitted calibration model."""

    model = json.loads(path.read_text(encoding="utf-8"))
    _validate_model(model)
    return model


def fit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-fit-candidate-score-calibration",
        description="fit train-only class-conditioned MMUAD candidate score calibration",
    )
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--train-truth", type=Path, required=True)
    parser.add_argument("--train-class-probabilities-csv", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--calibrated-train-candidates-csv", type=Path)
    parser.add_argument("--diagnostics-csv", type=Path)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--output-score-column", default=DEFAULT_OUTPUT_SCORE_COLUMN)
    parser.add_argument("--good-threshold-m", type=float, default=5.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--min-group-weight", type=float, default=20.0)
    parser.add_argument("--l2-penalty", type=float, default=10.0)
    parser.add_argument("--max-abs-logit-offset", type=float, default=4.0)
    parser.add_argument("--no-branch-source-interactions", action="store_true")
    parser.add_argument("--fill-missing", choices=("uniform", "zero", "error"), default="uniform")
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.train_candidates)
    truth = load_evaluation_truth_file(args.train_truth).rows
    probabilities = pd.read_csv(args.train_class_probabilities_csv)
    model, calibrated, diagnostics = fit_candidate_score_calibration(
        candidates,
        truth,
        probabilities,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        output_score_column=args.output_score_column,
        good_threshold_m=args.good_threshold_m,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        min_group_weight=args.min_group_weight,
        l2_penalty=args.l2_penalty,
        max_abs_logit_offset=args.max_abs_logit_offset,
        include_branch_source_interactions=not args.no_branch_source_interactions,
        fill_missing_class_probabilities=args.fill_missing,
    )
    save_candidate_score_calibration_model(model, args.model_json)
    if args.calibrated_train_candidates_csv is not None:
        _write_csv(calibrated.rows, args.calibrated_train_candidates_csv)
    if args.diagnostics_csv is not None:
        _write_csv(diagnostics, args.diagnostics_csv)
    print("mmuad_fit_candidate_score_calibration=ok")
    print(f"model_json={args.model_json}")
    print(f"fit_matched_rows={model['fit_matched_rows']}")
    print(f"output_score_column={model['output_score_column']}")
    return 0


def apply_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-apply-candidate-score-calibration",
        description="apply train-fitted class-conditioned MMUAD candidate score calibration",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--class-probabilities-csv", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--fill-missing", choices=("uniform", "zero", "error"))
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidate_csv)
    probabilities = pd.read_csv(args.class_probabilities_csv)
    model = load_candidate_score_calibration_model(args.model_json)
    calibrated = apply_candidate_score_calibration(
        candidates,
        model,
        class_probabilities=probabilities,
        fill_missing_class_probabilities=args.fill_missing,
    )
    _write_csv(calibrated.rows, args.output_csv)
    if args.provenance_json is not None:
        args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_json.write_text(
            json.dumps(
                {
                    "protocol": model["protocol"],
                    "model_json": str(args.model_json),
                    "candidate_csv": str(args.candidate_csv),
                    "class_probabilities_csv": str(args.class_probabilities_csv),
                    "output_csv": str(args.output_csv),
                    "output_score_column": model["output_score_column"],
                    "row_count": int(len(calibrated.rows)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print("mmuad_apply_candidate_score_calibration=ok")
    print(f"output_csv={args.output_csv}")
    print(f"row_count={len(calibrated.rows)}")
    return 0


def _candidate_frame(candidates: CandidateFrame | pd.DataFrame) -> CandidateFrame:
    if isinstance(candidates, CandidateFrame):
        return CandidateFrame(normalize_candidate_columns(candidates.rows.copy()))
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame(candidates).copy()))


def _prepare_candidate_context(rows: pd.DataFrame) -> pd.DataFrame:
    out = normalize_candidate_columns(pd.DataFrame(rows).copy())
    if out.empty:
        return out
    out["sequence_id"] = out["sequence_id"].astype(str)
    if "source" not in out.columns:
        out["source"] = "unknown"
    out["source"] = out["source"].fillna("unknown").astype(str)
    if "candidate_branch" not in out.columns:
        out["candidate_branch"] = out["source"]
    out["candidate_branch"] = out["candidate_branch"].fillna(out["source"]).astype(str)
    return out


def _attach_truth_targets(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    good_threshold_m: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    if candidates.empty or truth.empty:
        return candidates.assign(
            candidate_truth_matched=False,
            candidate_truth_distance_3d_m=np.nan,
            candidate_good_target=False,
        )
    truth_rows = normalize_truth_columns(truth.copy())
    truth_rows["sequence_id"] = truth_rows["sequence_id"].astype(str)
    parts: list[pd.DataFrame] = []
    for sequence_id, candidate_group in candidates.groupby("sequence_id", sort=False):
        truth_group = truth_rows.loc[truth_rows["sequence_id"] == str(sequence_id)].copy()
        candidate_group = candidate_group.copy()
        candidate_group["_candidate_row_order"] = np.arange(len(candidate_group), dtype=int)
        if truth_group.empty:
            candidate_group["candidate_truth_matched"] = False
            candidate_group["candidate_truth_time_delta_s"] = np.nan
            candidate_group["candidate_truth_distance_3d_m"] = np.nan
            candidate_group["candidate_good_target"] = False
            parts.append(candidate_group)
            continue
        right = truth_group[["time_s", "x_m", "y_m", "z_m"]].rename(
            columns={
                "time_s": "truth_time_s",
                "x_m": "truth_x_m",
                "y_m": "truth_y_m",
                "z_m": "truth_z_m",
            }
        )
        merged = pd.merge_asof(
            candidate_group.sort_values("time_s"),
            right.sort_values("truth_time_s"),
            left_on="time_s",
            right_on="truth_time_s",
            direction="nearest",
            tolerance=float(max_truth_time_delta_s),
        )
        matched = merged["truth_time_s"].notna()
        delta = merged[["x_m", "y_m", "z_m"]].to_numpy(float) - merged[
            ["truth_x_m", "truth_y_m", "truth_z_m"]
        ].to_numpy(float)
        distance = np.linalg.norm(delta, axis=1)
        distance[~matched.to_numpy(bool)] = np.nan
        merged["candidate_truth_matched"] = matched
        merged["candidate_truth_time_delta_s"] = merged["time_s"] - merged["truth_time_s"]
        merged["candidate_truth_distance_3d_m"] = distance
        merged["candidate_good_target"] = matched & (distance <= float(good_threshold_m))
        parts.append(merged.sort_values("_candidate_row_order"))
    return pd.concat(parts, ignore_index=True).drop(columns=["_candidate_row_order"], errors="ignore")


def _infer_score_transform(rows: pd.DataFrame, score_column: str, fallback_column: str) -> str:
    score = _raw_score(rows, score_column, fallback_column)
    finite = score[np.isfinite(score)]
    if finite.empty or (finite.min() >= 0.0 and finite.max() <= 1.0):
        return "probability"
    return "sigmoid"


def _raw_score(rows: pd.DataFrame, score_column: str, fallback_column: str) -> pd.Series:
    primary = (
        pd.to_numeric(rows[score_column], errors="coerce")
        if score_column in rows.columns
        else pd.Series(np.nan, index=rows.index, dtype=float)
    )
    fallback = (
        pd.to_numeric(rows[fallback_column], errors="coerce")
        if fallback_column in rows.columns
        else pd.Series(0.5, index=rows.index, dtype=float)
    )
    return primary.fillna(fallback).fillna(0.5).astype(float)


def _base_probability(
    rows: pd.DataFrame,
    *,
    score_column: str,
    fallback_score_column: str,
    transform: str,
) -> np.ndarray:
    score = _raw_score(rows, score_column, fallback_score_column).to_numpy(float)
    if transform == "probability":
        return np.clip(score, _EPS, 1.0 - _EPS)
    if transform == "sigmoid":
        return np.clip(_sigmoid(score), _EPS, 1.0 - _EPS)
    raise ValueError(f"unsupported score transform: {transform}")


def _fit_group_class_offsets(
    rows: pd.DataFrame,
    base_logit: np.ndarray,
    target: np.ndarray,
    *,
    group_column: str,
    class_columns: list[str],
    min_group_weight: float,
    l2_penalty: float,
    max_abs_logit_offset: float,
    level: str,
    diagnostics: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    offsets: dict[str, dict[str, float]] = {}
    group_values = rows[group_column].fillna("unknown").astype(str)
    for group in sorted(group_values.unique()):
        group_mask = group_values.eq(group).to_numpy(float)
        class_map: dict[str, float] = {}
        for class_column in class_columns:
            class_label = class_column.rsplit("_", 1)[-1]
            class_probability = pd.to_numeric(rows[class_column], errors="coerce").fillna(0.0).to_numpy(float)
            weights = group_mask * class_probability
            effective_weight = float(np.sum(weights))
            if effective_weight < float(min_group_weight):
                continue
            offset = _fit_logit_offset(
                base_logit,
                target,
                weights,
                l2_penalty=l2_penalty,
                max_abs_offset=max_abs_logit_offset,
            )
            class_map[class_label] = float(offset)
            diagnostics.append(
                {
                    "level": level,
                    "group": group,
                    "class_label": class_label,
                    "effective_weight": effective_weight,
                    "positive_rate": float(np.sum(weights * target) / max(effective_weight, _EPS)),
                    "logit_offset": float(offset),
                }
            )
        if class_map:
            offsets[group] = class_map
    return offsets


def _soft_group_contribution(
    rows: pd.DataFrame,
    *,
    group_column: str,
    offsets: Mapping[str, Mapping[str, float]],
    class_columns: list[str],
) -> np.ndarray:
    contribution = np.zeros(len(rows), dtype=float)
    groups = rows[group_column].fillna("unknown").astype(str).to_numpy(object)
    for index, group in enumerate(groups):
        class_offsets = offsets.get(str(group), {})
        if not class_offsets:
            continue
        total = 0.0
        for class_column in class_columns:
            label = class_column.rsplit("_", 1)[-1]
            probability = float(pd.to_numeric(pd.Series([rows.iloc[index][class_column]]), errors="coerce").fillna(0.0).iloc[0])
            total += probability * float(class_offsets.get(label, 0.0))
        contribution[index] = total
    return contribution


def _fit_logit_offset(
    base_logit: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    l2_penalty: float,
    max_abs_offset: float,
) -> float:
    logits = np.asarray(base_logit, dtype=float)
    labels = np.asarray(target, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(logits) & np.isfinite(labels) & np.isfinite(sample_weights) & (sample_weights > 0.0)
    if not np.any(valid):
        return 0.0
    logits = logits[valid]
    labels = labels[valid]
    sample_weights = sample_weights[valid]

    def gradient(offset: float) -> float:
        predicted = _sigmoid(logits + float(offset))
        return float(np.sum(sample_weights * (predicted - labels)) + float(l2_penalty) * offset)

    lower = -abs(float(max_abs_offset))
    upper = abs(float(max_abs_offset))
    lower_gradient = gradient(lower)
    upper_gradient = gradient(upper)
    if lower_gradient >= 0.0:
        return lower
    if upper_gradient <= 0.0:
        return upper
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if gradient(midpoint) > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return float(0.5 * (lower + upper))


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    out = np.empty_like(array, dtype=float)
    positive = array >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def _nested_float_dict(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(group): {str(label): float(offset) for label, offset in dict(class_map).items()}
        for group, class_map in value.items()
        if isinstance(class_map, Mapping)
    }


def _validate_model(model: Mapping[str, Any]) -> None:
    if int(model.get("schema_version", -1)) != MODEL_SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate score calibration schema: {model.get('schema_version')}")
    for key in ("score_column", "fallback_score_column", "score_transform", "class_labels"):
        if key not in model:
            raise ValueError(f"candidate score calibration model missing {key}")


def _write_csv(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fit_main())
