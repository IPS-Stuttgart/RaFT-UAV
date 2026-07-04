"""Attach sequence-level class probabilities as candidate-ranking context.

The MMUAD sequence classifier can produce one probability vector per sequence.
This module joins those probabilities onto candidate rows and emits ``image_*``
feature columns that are already consumed by the cluster ranker. It is intended
for pose experiments where type classification should inform candidate
reliability without hard-branching on a possibly wrong class label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

OFFICIAL_CLASS_LABELS = ("0", "1", "2", "3")
SOURCE_CONTEXT_NAMES = (
    "lidar_360",
    "livox_avia",
    "radar_enhance_pcl",
    "cross_sensor_merged",
)
DEFAULT_INTERACTION_COLUMNS = (
    "confidence",
    "cluster_point_count",
    "cluster_extent_xy_m",
    "cluster_extent_3d_m",
    "cluster_density_points_per_m3",
    "cluster_range_3d_m",
    "cluster_height_m",
    "image_candidate_branch_dynamic_flag",
    "image_source_is_lidar_360",
    "image_source_is_livox_avia",
    "image_source_is_radar_enhance_pcl",
    "image_source_is_cross_sensor_merged",
)
SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq", "scene_id")
PREDICTED_CLASS_ALIASES = ("predicted_class", "Classification", "uav_type", "class_id")


def attach_class_probability_context(
    candidates: CandidateFrame | pd.DataFrame,
    class_probabilities: pd.DataFrame,
    *,
    interaction_columns: Iterable[str] = DEFAULT_INTERACTION_COLUMNS,
    fill_missing: str = "uniform",
) -> CandidateFrame:
    """Return candidate rows with sequence-level class-probability features.

    ``class_probabilities`` may use classifier output columns such as
    ``predicted_probability_0`` or already-normalized ``class_prob_0`` columns.
    The emitted probability/context columns are prefixed with ``image_`` so that
    existing MMUAD cluster-ranker feature selection consumes them without a new
    model schema.
    """

    candidate_rows = _candidate_rows(candidates)
    probability_rows = _probability_rows(class_probabilities)
    out = candidate_rows.merge(probability_rows, on="sequence_id", how="left")
    out = _fill_missing_probabilities(out, fill_missing=fill_missing)
    out = _add_probability_summaries(out)
    out = _add_candidate_context_summaries(out)
    out = _add_probability_interactions(out, interaction_columns=tuple(interaction_columns))
    return CandidateFrame(normalize_candidate_columns(out))


def write_class_probability_context(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    provenance_json: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Write augmented candidates and optional provenance."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if provenance_json is not None:
        provenance_json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(provenance or {})
        payload.update(
            {
                "output_csv": str(output_csv),
                "row_count": int(len(candidates.rows)),
                "class_probability_columns": [
                    column
                    for column in candidates.rows.columns
                    if str(column).startswith("image_class_prob_")
                ],
                "interaction_columns": [
                    column
                    for column in candidates.rows.columns
                    if str(column).startswith("image_class_prob_") and "_x_" in str(column)
                ],
                "source_context_columns": [
                    column
                    for column in candidates.rows.columns
                    if str(column).startswith("image_source_is_")
                ],
            }
        )
        provenance_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-class-prob-context",
        description="attach sequence-level class probabilities to MMUAD candidates",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--class-probabilities-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument(
        "--interaction-column",
        action="append",
        default=[],
        help="candidate numeric column to multiply by each class probability; may be repeated",
    )
    parser.add_argument(
        "--fill-missing",
        choices=("uniform", "zero", "error"),
        default="uniform",
        help="how to handle sequences missing from the probability table",
    )
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidate_csv)
    probabilities = pd.read_csv(args.class_probabilities_csv)
    interaction_columns = tuple(args.interaction_column) or DEFAULT_INTERACTION_COLUMNS
    augmented = attach_class_probability_context(
        candidates,
        probabilities,
        interaction_columns=interaction_columns,
        fill_missing=args.fill_missing,
    )
    write_class_probability_context(
        augmented,
        output_csv=args.output_csv,
        provenance_json=args.provenance_json,
        provenance={
            "candidate_csv": str(args.candidate_csv),
            "class_probabilities_csv": str(args.class_probabilities_csv),
            "fill_missing": str(args.fill_missing),
            "requested_interaction_columns": list(interaction_columns),
        },
    )
    print("mmuad_class_probability_context=ok")
    print(f"output_csv={args.output_csv}")
    if args.provenance_json is not None:
        print(f"provenance_json={args.provenance_json}")
    return 0


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates).copy()
    rows = normalize_candidate_columns(rows)
    if not rows.empty:
        rows["sequence_id"] = rows["sequence_id"].astype(str)
    return rows


def _probability_rows(class_probabilities: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(class_probabilities).copy()
    if rows.empty:
        raise ValueError("class probability table is empty")
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("class probability table missing sequence_id/Sequence column")
    out = pd.DataFrame({"sequence_id": rows[sequence_column].astype(str)})
    found_probability = False
    for label in OFFICIAL_CLASS_LABELS:
        source = _probability_column(rows, label)
        if source is not None:
            out[f"image_class_prob_{label}"] = pd.to_numeric(rows[source], errors="coerce")
            found_probability = True
    predicted_column = _first_present(rows, PREDICTED_CLASS_ALIASES)
    if not found_probability:
        if predicted_column is None:
            raise ValueError("class probability table has neither probabilities nor predicted_class")
        predicted = _predicted_class_labels(rows[predicted_column])
        _validate_predicted_class_labels(predicted)
        for label in OFFICIAL_CLASS_LABELS:
            out[f"image_class_prob_{label}"] = (predicted == label).astype(float)
    elif predicted_column is not None:
        out["image_input_predicted_class_id"] = pd.to_numeric(
            rows[predicted_column], errors="coerce"
        )
    out = out.groupby("sequence_id", as_index=False).mean(numeric_only=True)
    return _normalize_probability_weights(out)


def _predicted_class_labels(values: pd.Series) -> pd.Series:
    """Return canonical Track 5 class id strings from classifier labels."""

    raw = pd.Series(values)
    text = raw.where(raw.notna(), "").astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    boolean_values = raw.map(lambda value: isinstance(value, bool | np.bool_)).to_numpy(bool)
    integer_like = (
        np.isfinite(numeric_array)
        & np.isclose(numeric_array, np.rint(numeric_array))
        & ~boolean_values
    )
    if integer_like.any():
        positions = np.flatnonzero(integer_like)
        text.iloc[positions] = np.rint(numeric_array[positions]).astype(int).astype(str)
    return text


def _validate_predicted_class_labels(labels: pd.Series) -> None:
    """Reject non-empty fallback class labels outside the official Track 5 ids."""

    text = pd.Series(labels).fillna("").astype(str).str.strip()
    present = text.ne("")
    invalid = present & ~text.isin(OFFICIAL_CLASS_LABELS)
    if not invalid.any():
        return
    examples = sorted(text.loc[invalid].unique())
    allowed = ", ".join(OFFICIAL_CLASS_LABELS)
    raise ValueError(
        "missing class probabilities because predicted_class values must be official "
        f"Track 5 class IDs {{{allowed}}}; got {examples}"
    )


def _probability_column(rows: pd.DataFrame, label: str) -> str | None:
    candidates = (
        f"image_class_prob_{label}",
        f"class_prob_{label}",
        f"class_probability_{label}",
        f"predicted_probability_{label}",
        f"probability_{label}",
        f"p_class_{label}",
    )
    return _first_present(rows, candidates)


def _normalize_probability_weights(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    prob_columns = [f"image_class_prob_{label}" for label in OFFICIAL_CLASS_LABELS]
    for column in prob_columns:
        if column not in out.columns:
            out[column] = 0.0
        numeric = pd.to_numeric(out[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        out[column] = numeric.where(finite, 0.0).clip(lower=0.0)
    totals = out[prob_columns].sum(axis=1)
    has_probability = totals > 0.0
    safe_totals = totals.where(has_probability, 1.0)
    for column in prob_columns:
        out[column] = np.where(has_probability, out[column] / safe_totals, np.nan)
    return out


def _fill_missing_probabilities(rows: pd.DataFrame, *, fill_missing: str) -> pd.DataFrame:
    out = rows.copy()
    prob_columns = [f"image_class_prob_{label}" for label in OFFICIAL_CLASS_LABELS]
    missing = out[prob_columns].isna().all(axis=1)
    if missing.any() and fill_missing == "error":
        missing_sequences = sorted(out.loc[missing, "sequence_id"].astype(str).unique())
        raise ValueError(f"missing class probabilities for sequences: {missing_sequences}")
    if fill_missing == "uniform":
        out.loc[missing, prob_columns] = 1.0 / len(prob_columns)
    elif fill_missing == "zero":
        out.loc[missing, prob_columns] = 0.0
    out["image_class_probability_available"] = (~missing).astype(float)
    for column in prob_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def _add_probability_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    prob_columns = [f"image_class_prob_{label}" for label in OFFICIAL_CLASS_LABELS]
    probs = out[prob_columns].to_numpy(float)
    clipped = np.clip(probs, 1.0e-12, 1.0)
    out["image_class_entropy"] = -np.sum(clipped * np.log(clipped), axis=1)
    out["image_class_confidence"] = np.max(probs, axis=1)
    predicted = np.argmax(probs, axis=1)
    out["image_predicted_class_id"] = predicted.astype(float)
    for index, label in enumerate(OFFICIAL_CLASS_LABELS):
        out[f"image_predicted_class_{label}"] = (predicted == index).astype(float)
    return out


def _add_candidate_context_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    branch_text = _text_series(out, "candidate_branch", fallback_column="source")
    source_text = _text_series(out, "source")
    out["image_candidate_branch_dynamic_flag"] = branch_text.str.contains("dynamic").astype(float)
    for source_name in SOURCE_CONTEXT_NAMES:
        source_key = source_name.replace("_", "-")
        source_match = source_text.eq(source_name) | source_text.eq(source_key)
        source_match |= source_text.str.contains(source_name, regex=False)
        source_match |= source_text.str.contains(source_key, regex=False)
        out[f"image_source_is_{source_name}"] = source_match.astype(float)
    return out


def _text_series(rows: pd.DataFrame, column: str, *, fallback_column: str | None = None) -> pd.Series:
    if column in rows.columns:
        values = rows[column]
    elif fallback_column is not None and fallback_column in rows.columns:
        values = rows[fallback_column]
    else:
        values = pd.Series("", index=rows.index)
    return values.fillna("").astype(str).str.lower().str.strip()


def _add_probability_interactions(
    rows: pd.DataFrame,
    *,
    interaction_columns: tuple[str, ...],
) -> pd.DataFrame:
    out = rows.copy()
    available = [column for column in interaction_columns if column in out.columns]
    for base_column in available:
        values = pd.to_numeric(out[base_column], errors="coerce").fillna(0.0)
        safe_name = _safe_feature_name(base_column)
        for label in OFFICIAL_CLASS_LABELS:
            probability = pd.to_numeric(out[f"image_class_prob_{label}"], errors="coerce").fillna(0.0)
            out[f"image_class_prob_{label}_x_{safe_name}"] = probability * values
    return out


def _first_present(rows: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    column_map = {str(column).lower(): str(column) for column in rows.columns}
    for candidate in candidates:
        text = str(candidate)
        if text in rows.columns:
            return text
        lowered = text.lower()
        if lowered in column_map:
            return column_map[lowered]
    return None


def _safe_feature_name(value: str) -> str:
    return str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
