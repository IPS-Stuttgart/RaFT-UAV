"""Sequence-consistent trajectory-medoid ensembling for MMUAD Track 5.

Row-wise averaging and consensus clustering can blend incompatible branches or
change the contributing source at every timestamp.  This module instead scores
complete resampled candidate trajectories against one another and selects one
weighted medoid trajectory per sequence.  The selected output therefore remains
an actual candidate trajectory while still using cross-pipeline agreement as an
inference-safe selection signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    parse_official_sequence_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

TRAJECTORY_MEDOID_ESTIMATES_CSV = "mmuad_track5_trajectory_medoid_estimates.csv"
TRAJECTORY_MEDOID_DIAGNOSTICS_CSV = "mmuad_track5_trajectory_medoid_diagnostics.csv"
TRAJECTORY_MEDOID_MANIFEST_JSON = "mmuad_track5_trajectory_medoid_manifest.json"
VALIDATION_JSON = "mmuad_track5_trajectory_medoid_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_trajectory_medoid_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9


def build_track5_trajectory_medoid_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    min_coverage_fraction: float = 1.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one weighted trajectory medoid per sequence.

    Each input is resampled onto the official template first.  For a sequence,
    candidate ``i`` receives the weighted objective

    ``sum_j weight_j * mean_t ||x_i(t) - x_j(t)|| / sum_j weight_j``.

    Pair distances use timestamps valid for both candidates.  A candidate with
    no overlap with another active candidate has an unavailable score.  Selection
    is restricted to candidates meeting ``min_coverage_fraction``.  When none do,
    the maximum-coverage set is used and the relaxation is recorded.  If the
    selected trajectory is invalid at an individual row, the highest-weight valid
    candidate supplies that row without coordinate blending.
    """

    min_coverage = _validate_fraction(
        min_coverage_fraction,
        name="min_coverage_fraction",
    )
    template_rows = _normalize_template_rows(template)
    input_list = tuple(estimate_inputs)
    if not input_list:
        raise ValueError("at least one estimate input is required")
    if template_rows.empty:
        empty = pd.DataFrame(
            columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]
        )
        return empty, pd.DataFrame()

    parts: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    seen_labels: dict[str, str] = {}
    for order, (raw_label, estimates, raw_weight) in enumerate(input_list):
        raw_text = str(raw_label)
        label = _safe_label(raw_text)
        _check_label_is_unique(label, raw_label=raw_text, seen_labels=seen_labels)
        weight = _validate_weight(raw_weight, label=label)
        resampled, _ = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        part = resampled[
            ["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]
        ].copy()
        part["input_label"] = label
        part["input_order"] = int(order)
        part["input_weight"] = weight
        part["input_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        parts.append(part)
        metadata.append(
            {
                "label": label,
                "order": int(order),
                "weight": weight,
            }
        )
    if not any(float(item["weight"]) > 0.0 for item in metadata):
        raise ValueError("at least one estimate input must have positive weight")
    stacked = pd.concat(parts, ignore_index=True, sort=False)

    estimate_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for sequence_id, sequence_template in template_rows.groupby("sequence_id", sort=True):
        sequence_template = sequence_template.sort_values("time_s").reset_index(drop=True)
        sequence_id = str(sequence_id)
        row_count = int(len(sequence_template))
        xyz_rows: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        for item in metadata:
            candidate = stacked.loc[
                (stacked["sequence_id"].astype(str) == sequence_id)
                & (stacked["input_label"] == item["label"])
            ].sort_values("time_s")
            _validate_resampled_alignment(
                candidate,
                sequence_template,
                label=str(item["label"]),
                sequence_id=sequence_id,
            )
            xyz = candidate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            valid = candidate["input_valid"].astype(bool).to_numpy()
            xyz_rows.append(xyz)
            valid_rows.append(valid)

        xyz_matrix = np.stack(xyz_rows, axis=0)
        valid_matrix = np.stack(valid_rows, axis=0)
        weights = np.asarray([float(item["weight"]) for item in metadata], dtype=float)
        coverage_rows = valid_matrix.sum(axis=1).astype(int)
        coverage_fraction = coverage_rows.astype(float) / float(row_count)
        scores, comparison_counts, min_overlap_fractions = _trajectory_medoid_scores(
            xyz_matrix,
            valid_matrix,
            weights,
        )
        selected_index, reason, coverage_relaxed = _select_medoid_index(
            scores=scores,
            coverage_fraction=coverage_fraction,
            weights=weights,
            metadata=metadata,
            min_coverage_fraction=min_coverage,
        )
        selected_label = str(metadata[selected_index]["label"])
        selected_score = float(scores[selected_index])
        if not np.isfinite(selected_score):
            selected_score = np.nan

        for index, item in enumerate(metadata):
            diagnostic_records.append(
                {
                    "sequence_id": sequence_id,
                    "candidate_label": str(item["label"]),
                    "candidate_order": int(item["order"]),
                    "candidate_weight": float(item["weight"]),
                    "template_row_count": row_count,
                    "valid_row_count": int(coverage_rows[index]),
                    "coverage_fraction": float(coverage_fraction[index]),
                    "meets_min_coverage": bool(
                        coverage_fraction[index] >= min_coverage
                    ),
                    "trajectory_medoid_score_m": float(scores[index])
                    if np.isfinite(scores[index])
                    else np.nan,
                    "pairwise_comparison_count": int(comparison_counts[index]),
                    "minimum_pair_overlap_fraction": float(
                        min_overlap_fractions[index]
                    )
                    if np.isfinite(min_overlap_fractions[index])
                    else np.nan,
                    "selected": bool(index == selected_index),
                    "selected_label": selected_label,
                    "selection_reason": reason,
                    "coverage_relaxed": bool(coverage_relaxed),
                }
            )

        for row_index, template_row in sequence_template.iterrows():
            chosen_index = selected_index
            fallback = not bool(valid_matrix[selected_index, row_index])
            if fallback:
                available = np.flatnonzero(
                    valid_matrix[:, row_index] & (weights > 0.0)
                )
                if len(available):
                    chosen_index = min(
                        available.tolist(),
                        key=lambda index: (
                            -float(weights[index]),
                            int(metadata[index]["order"]),
                            str(metadata[index]["label"]),
                        ),
                    )
            chosen_valid = bool(valid_matrix[chosen_index, row_index])
            xyz = (
                xyz_matrix[chosen_index, row_index]
                if chosen_valid
                else np.asarray([np.nan, np.nan, np.nan], dtype=float)
            )
            row_label = str(metadata[chosen_index]["label"]) if chosen_valid else ""
            estimate_records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(template_row["time_s"]),
                    "source": "track5-trajectory-medoid",
                    "track_id": "track5-trajectory-medoid",
                    "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                    "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                    "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                    "trajectory_medoid_sequence_label": selected_label,
                    "trajectory_medoid_row_label": row_label,
                    "trajectory_medoid_score_m": selected_score,
                    "trajectory_medoid_fallback": bool(fallback),
                    "trajectory_medoid_selection_reason": reason,
                }
            )

    estimates = pd.DataFrame.from_records(estimate_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    return estimates, diagnostics


def _trajectory_medoid_scores(
    xyz: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted trajectory-medoid scores and overlap diagnostics."""

    points = np.asarray(xyz, dtype=float)
    valid_rows = np.asarray(valid, dtype=bool)
    safe_weights = np.asarray(weights, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("xyz must have shape (candidate_count, row_count, 3)")
    if valid_rows.shape != points.shape[:2]:
        raise ValueError("valid must match the first two xyz dimensions")
    if safe_weights.shape != (points.shape[0],):
        raise ValueError("weights must have one value per candidate")

    scores = np.full(points.shape[0], np.inf, dtype=float)
    comparison_counts = np.zeros(points.shape[0], dtype=int)
    minimum_overlaps = np.full(points.shape[0], np.nan, dtype=float)
    coverage_rows = valid_rows.sum(axis=1)
    active = (safe_weights > 0.0) & (coverage_rows > 0)
    active_weight_sum = float(np.sum(safe_weights[active]))
    if active_weight_sum <= 0.0:
        return scores, comparison_counts, minimum_overlaps

    row_count = int(points.shape[1])
    for candidate_index in np.flatnonzero(active):
        weighted_distance_sum = 0.0
        overlaps: list[float] = []
        score_available = True
        for comparison_index in np.flatnonzero(active):
            if comparison_index == candidate_index:
                continue
            overlap = valid_rows[candidate_index] & valid_rows[comparison_index]
            overlap_count = int(overlap.sum())
            if overlap_count == 0:
                score_available = False
                break
            distances = np.linalg.norm(
                points[candidate_index, overlap] - points[comparison_index, overlap],
                axis=1,
            )
            weighted_distance_sum += float(safe_weights[comparison_index]) * float(
                np.mean(distances)
            )
            comparison_counts[candidate_index] += 1
            overlaps.append(float(overlap_count) / float(row_count))
        if score_available:
            scores[candidate_index] = weighted_distance_sum / active_weight_sum
            minimum_overlaps[candidate_index] = min(overlaps) if overlaps else 1.0
    return scores, comparison_counts, minimum_overlaps


def _select_medoid_index(
    *,
    scores: np.ndarray,
    coverage_fraction: np.ndarray,
    weights: np.ndarray,
    metadata: list[dict[str, Any]],
    min_coverage_fraction: float,
) -> tuple[int, str, bool]:
    positive = weights > 0.0
    coverage_eligible = positive & (coverage_fraction >= min_coverage_fraction)
    coverage_relaxed = False
    reason = "trajectory_medoid"
    if not coverage_eligible.any():
        coverage_relaxed = True
        reason = "coverage_relaxed_trajectory_medoid"
        maximum_coverage = float(np.max(coverage_fraction[positive]))
        coverage_eligible = positive & np.isclose(
            coverage_fraction,
            maximum_coverage,
            rtol=0.0,
            atol=1.0e-12,
        )

    finite_eligible = coverage_eligible & np.isfinite(scores)
    if finite_eligible.any():
        candidates = np.flatnonzero(finite_eligible).tolist()
        selected = min(
            candidates,
            key=lambda index: (
                float(scores[index]),
                -float(coverage_fraction[index]),
                -float(weights[index]),
                int(metadata[index]["order"]),
                str(metadata[index]["label"]),
            ),
        )
        return int(selected), reason, coverage_relaxed

    reason = "coverage_weight_fallback"
    candidates = np.flatnonzero(coverage_eligible).tolist()
    selected = min(
        candidates,
        key=lambda index: (
            -float(coverage_fraction[index]),
            -float(weights[index]),
            int(metadata[index]["order"]),
            str(metadata[index]["label"]),
        ),
    )
    return int(selected), reason, coverage_relaxed


def write_track5_trajectory_medoid_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    min_coverage_fraction: float = 1.0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write medoid estimates, diagnostics, official artifacts, and validation."""

    input_list = list(estimate_inputs)
    loaded = [
        (item.label, read_estimate_csv(item.path), float(item.weight))
        for item in input_list
    ]
    estimates, diagnostics = build_track5_trajectory_medoid_ensemble(
        loaded,
        template,
        min_coverage_fraction=min_coverage_fraction,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / TRAJECTORY_MEDOID_ESTIMATES_CSV,
        "diagnostics_csv": output / TRAJECTORY_MEDOID_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / TRAJECTORY_MEDOID_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    write_official_mmaud_results_csv(
        estimates,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map or {},
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        estimates,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map or {},
        invalid_row_policy="raise",
    )
    validation = validate_official_track5_submission(
        paths["official_zip"],
        template=template,
        require_zip=True,
    )
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    selected_mask = (
        diagnostics["selected"].astype(bool)
        if "selected" in diagnostics.columns
        else pd.Series(False, index=diagnostics.index)
    )
    selected = diagnostics.loc[selected_mask]
    manifest = {
        "schema": "raft-uav-mmuad-track5-trajectory-medoid-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "row_count": int(len(estimates)),
        "sequence_count": int(estimates["sequence_id"].nunique())
        if not estimates.empty
        else 0,
        "min_coverage_fraction": float(min_coverage_fraction),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "selected_sequences": selected[
            ["sequence_id", "candidate_label", "trajectory_medoid_score_m"]
        ].to_dict(orient="records")
        if not selected.empty
        else [],
        "coverage_relaxed_sequence_count": int(
            selected.get("coverage_relaxed", pd.Series(dtype=bool)).astype(bool).sum()
        ),
        "fallback_row_count": int(
            estimates.get("trajectory_medoid_fallback", pd.Series(dtype=bool))
            .astype(bool)
            .sum()
        ),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(
            validation.summary.get("codabench_upload_ready", False)
        ),
        "paths": {
            name: str(path)
            for name, path in paths.items()
            if name != "manifest_json"
        },
    }
    paths["manifest_json"].write_text(
        json.dumps(_jsonable(manifest), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.track5_trajectory_medoid_ensemble",
        description=(
            "select one weighted trajectory-medoid estimate source per MMUAD Track 5 sequence"
        ),
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH[@WEIGHT]",
        help="estimate trajectory to include; may be repeated",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--min-coverage-fraction", type=float, default=1.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_trajectory_medoid_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        min_coverage_fraction=float(args.min_coverage_fraction),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_trajectory_medoid_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not validation.get("leaderboard_ready", False):
        reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"trajectory-medoid ensemble is not leaderboard-ready: {reasons}")
    return 0


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, ("sequence_id", "Sequence", "sequence", "seq"))
    time_column = _first_present(
        rows,
        ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"),
    )
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(_template_sequence_or_none),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _validate_resampled_alignment(
    candidate: pd.DataFrame,
    sequence_template: pd.DataFrame,
    *,
    label: str,
    sequence_id: str,
) -> None:
    if len(candidate) != len(sequence_template):
        raise RuntimeError(
            f"resampled input {label!r} has {len(candidate)} rows for {sequence_id}; "
            f"expected {len(sequence_template)}"
        )
    candidate_times = pd.to_numeric(candidate["time_s"], errors="coerce").to_numpy(float)
    template_times = sequence_template["time_s"].to_numpy(float)
    if not np.allclose(
        candidate_times,
        template_times,
        rtol=0.0,
        atol=TEMPLATE_TIME_MATCH_ATOL_S,
    ):
        raise RuntimeError(
            f"resampled input {label!r} does not align with template times for {sequence_id}"
        )


def _template_sequence_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


def _validate_weight(value: float, *, label: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"estimate weight must be finite and non-negative for {label}: {weight}"
        )
    return weight


def _validate_fraction(value: float, *, name: str) -> float:
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return fraction


def _check_label_is_unique(
    label: str,
    *,
    raw_label: str,
    seen_labels: dict[str, str],
) -> None:
    previous = seen_labels.get(label)
    if previous is not None:
        if previous == raw_label:
            raise ValueError(f"estimate input label {label!r} is duplicated")
        raise ValueError(
            "estimate input labels collide after normalization: "
            f"{previous!r} and {raw_label!r} both normalize to {label!r}"
        )
    seen_labels[label] = raw_label


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).strip().lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.strip().lower())
        if found is not None:
            return found
    return None


def _safe_label(value: Any) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return label or "estimate"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
