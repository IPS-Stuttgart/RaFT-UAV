"""Attach sequence-classifier context to MMUAD candidate rows.

The sequence classifier is useful for more than the final Track 5
``Classification`` column: UAV type probabilities can condition candidate
ranking, learned uncertainty, and mixture-MAP pose inference.  This module
merges train/validation sequence-level class probabilities onto candidate rows
and emits numeric ``image_*`` features so the existing cluster-ranker feature
selection path can consume them without a separate schema change.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns

CLASS_IDS = (0, 1, 2, 3)
SEQUENCE_ALIASES = ("sequence_id", "sequence", "Sequence", "heldout_sequence")
PREDICTED_CLASS_ALIASES = ("predicted_class", "uav_type", "Classification")
INTERACTION_BASE_COLUMNS = (
    "cluster_point_count",
    "cluster_extent_3d_m",
    "cluster_extent_xy_m",
    "cluster_density_points_per_m3",
    "cross_sensor_neighbor_count",
    "nearest_cross_sensor_score",
    "frame_candidate_count",
    "frame_source_count",
)


def add_class_context_features(
    candidates: pd.DataFrame,
    class_predictions: pd.DataFrame,
    *,
    missing_policy: str = "uniform",
) -> pd.DataFrame:
    """Return candidates augmented with sequence-level class-context features.

    ``class_predictions`` may use the LOSO sequence-classifier schema with
    ``predicted_probability_0..3`` columns, a class-map-like schema with only a
    predicted class, or already-normalized ``class_prob_0..3`` columns.  Missing
    candidate sequences get a uniform prior by default, which keeps test-set
    inference safe when a classifier omits a sequence.
    """

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows
    class_rows = _class_probability_rows(class_predictions)
    out = rows.merge(class_rows, on="sequence_id", how="left")
    missing = out["image_sequence_class_context_available"].isna()
    if missing.any():
        missing_policy = str(missing_policy).lower().replace("_", "-")
        if missing_policy == "error":
            missing_sequences = sorted(out.loc[missing, "sequence_id"].astype(str).unique())
            raise ValueError(f"missing class predictions for sequences: {missing_sequences}")
        if missing_policy == "nan":
            out.loc[missing, "image_sequence_class_context_available"] = 0.0
        elif missing_policy == "uniform":
            out.loc[missing, "image_sequence_class_context_available"] = 0.0
            for class_id in CLASS_IDS:
                out.loc[missing, f"image_sequence_class_prob_{class_id}"] = 1.0 / len(CLASS_IDS)
            out.loc[missing, "image_sequence_predicted_class_id"] = -1.0
            out.loc[missing, "image_sequence_class_max_prob"] = 1.0 / len(CLASS_IDS)
            out.loc[missing, "image_sequence_class_entropy"] = float(np.log(len(CLASS_IDS)))
        else:
            raise ValueError("missing_policy must be one of: uniform, nan, error")
    out["image_sequence_class_context_available"] = pd.to_numeric(
        out["image_sequence_class_context_available"],
        errors="coerce",
    ).fillna(0.0)
    out = _add_class_interactions(out)
    return out.sort_values(["sequence_id", "time_s", "source"]).reset_index(drop=True)


def _class_probability_rows(class_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(class_predictions).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "image_sequence_class_context_available"])
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("class predictions must include a sequence id column")
    out = pd.DataFrame({"sequence_id": rows[sequence_column].astype(str).str.strip()})
    probability_columns = [_probability_column(rows, class_id) for class_id in CLASS_IDS]
    if all(column is not None for column in probability_columns):
        for class_id, column in zip(CLASS_IDS, probability_columns, strict=True):
            out[f"image_sequence_class_prob_{class_id}"] = pd.to_numeric(
                rows[column],
                errors="coerce",
            )
    else:
        predicted_column = _first_present(rows, PREDICTED_CLASS_ALIASES)
        if predicted_column is None:
            raise ValueError(
                "class predictions must include predicted_probability_0..3, "
                "class_prob_0..3, or a predicted class column",
            )
        predicted = pd.to_numeric(rows[predicted_column].map(_parse_class_cell), errors="coerce")
        for class_id in CLASS_IDS:
            out[f"image_sequence_class_prob_{class_id}"] = (predicted == class_id).astype(float)
    out = _normalize_probability_columns(out)
    probs = out[[f"image_sequence_class_prob_{class_id}" for class_id in CLASS_IDS]].to_numpy(float)
    out["image_sequence_predicted_class_id"] = np.argmax(probs, axis=1).astype(float)
    out["image_sequence_class_max_prob"] = np.max(probs, axis=1)
    entropy = -np.sum(np.where(probs > 0.0, probs * np.log(np.maximum(probs, 1.0e-12)), 0.0), axis=1)
    out["image_sequence_class_entropy"] = entropy.astype(float)
    out["image_sequence_class_context_available"] = 1.0
    return (
        out.groupby("sequence_id", as_index=False)
        .mean(numeric_only=True)
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )


def _normalize_probability_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    prob_cols = [f"image_sequence_class_prob_{class_id}" for class_id in CLASS_IDS]
    probs = out[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    sums = probs.sum(axis=1).replace(0.0, np.nan)
    probs = probs.div(sums, axis=0).fillna(1.0 / len(CLASS_IDS))
    out[prob_cols] = probs
    return out


def _probability_column(rows: pd.DataFrame, class_id: int) -> str | None:
    aliases = (
        f"predicted_probability_{class_id}",
        f"probability_{class_id}",
        f"class_prob_{class_id}",
        f"p_class_{class_id}",
        f"prob_class_{class_id}",
    )
    return _first_present(rows, aliases)


def _first_present(rows: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower_to_column = {str(column).lower(): str(column) for column in rows.columns}
    for alias in aliases:
        column = lower_to_column.get(alias.lower())
        if column is not None:
            return column
    return None


def _parse_class_cell(value: Any) -> float | None:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text.strip("[]").split(",", 1)[0]
        if text.startswith("(") and text.endswith(")"):
            text = text.strip("()").split(",", 1)[0]
        value = text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_class_interactions(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    dynamic_text = (
        out.get("candidate_branch", out.get("source", ""))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    out["image_candidate_branch_dynamic_flag"] = dynamic_text.str.contains("dynamic").astype(float)
    for class_id in CLASS_IDS:
        prob_col = f"image_sequence_class_prob_{class_id}"
        prob = pd.to_numeric(out.get(prob_col), errors="coerce")
        if prob.isna().all():
            continue
        prob = prob.fillna(0.0)
        for base_col in INTERACTION_BASE_COLUMNS:
            if base_col not in out.columns:
                continue
            base = pd.to_numeric(out[base_col], errors="coerce").fillna(0.0)
            out[f"image_class{class_id}_x_{base_col}"] = prob * base
        out[f"image_class{class_id}_x_dynamic_candidate"] = (
            prob * out["image_candidate_branch_dynamic_flag"]
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-class-context-features",
        description="merge sequence classifier probabilities into MMUAD candidates",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--class-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--missing-policy",
        choices=("uniform", "nan", "error"),
        default="uniform",
    )
    args = parser.parse_args(argv)

    candidates = pd.read_csv(args.candidates_csv)
    predictions = pd.read_csv(args.class_predictions_csv)
    output = add_class_context_features(candidates, predictions, missing_policy=args.missing_policy)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print("mmuad_class_context_features=ok")
    print(f"candidate_rows={len(candidates)}")
    print(f"output_rows={len(output)}")
    print(f"output_csv={args.output_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
