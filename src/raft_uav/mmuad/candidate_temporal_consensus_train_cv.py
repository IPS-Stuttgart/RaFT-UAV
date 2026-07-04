"""Train-CV selection and inference apply for MMUAD temporal consensus.

The truth-free temporal-consensus stage exposes several weights controlling how
base ranker confidence, backward/forward continuity, bidirectional support,
interpolation consistency, and local acceleration are combined. This module
selects those weights on training sequences with leave-one-sequence-out
cross-validation, writes a frozen JSON config, and applies that config to
validation/test candidates without truth labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from itertools import product
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables
from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    add_temporal_candidate_consensus,
    temporal_consensus_summary,
)
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import (
    CandidateFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)

_DEFAULT_TOP_K = (1, 3, 5, 10, 20)
_DEFAULT_SELECTION_METRIC = "oracle_top3_3d_m_mse"
_CONFIG_SCHEMA_VERSION = 1


def select_temporal_consensus_config_by_sequence_cv(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    base_score_weights: Sequence[float] = (0.1, 0.25, 0.5),
    support_weights: Sequence[float] = (0.5, 1.0, 1.5),
    bidirectional_bonuses: Sequence[float] = (0.0, 0.75),
    interpolation_weights: Sequence[float] = (0.0, 0.75),
    acceleration_weights: Sequence[float] = (0.0, 0.5),
    max_time_gap_s: float = 2.0,
    max_speed_mps: float = 60.0,
    distance_scale_m: float = 5.0,
    acceleration_scale_mps2: float = 20.0,
    score_column: str = "ranker_score",
    fallback_score_column: str = "confidence",
    source_diversity_bonus: float = 0.25,
    branch_diversity_bonus: float = 0.25,
    top_k_values: Sequence[int] = _DEFAULT_TOP_K,
    max_truth_time_delta_s: float = 0.5,
    selection_metric: str = _DEFAULT_SELECTION_METRIC,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select temporal-consensus weights using sequence-level cross-validation."""

    candidate_rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if candidate_rows.empty:
        raise ValueError("temporal-consensus train-CV selection requires candidate rows")
    if truth_rows.empty:
        raise ValueError("temporal-consensus train-CV selection requires truth rows")
    if "sequence_id" not in candidate_rows.columns or "sequence_id" not in truth_rows.columns:
        raise ValueError("candidate and truth rows must include sequence_id")

    candidate_rows["sequence_id"] = candidate_rows["sequence_id"].astype(str)
    truth_rows["sequence_id"] = truth_rows["sequence_id"].astype(str)
    sequences = sorted(set(candidate_rows["sequence_id"]) & set(truth_rows["sequence_id"]))
    if len(sequences) < 2:
        raise ValueError("at least two sequences are required for leave-one-sequence-out CV")

    top_k_tuple = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
    if not top_k_tuple:
        raise ValueError("top_k_values must contain at least one positive value")

    configs = _config_grid(
        base_score_weights=base_score_weights,
        support_weights=support_weights,
        bidirectional_bonuses=bidirectional_bonuses,
        interpolation_weights=interpolation_weights,
        acceleration_weights=acceleration_weights,
        max_time_gap_s=max_time_gap_s,
        max_speed_mps=max_speed_mps,
        distance_scale_m=distance_scale_m,
        acceleration_scale_mps2=acceleration_scale_mps2,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        source_diversity_bonus=source_diversity_bonus,
        branch_diversity_bonus=branch_diversity_bonus,
    )
    if not configs:
        raise ValueError("temporal-consensus grid did not produce any configurations")

    evaluation_rows: list[dict[str, Any]] = []
    cache: dict[str, tuple[TemporalConsensusConfig, pd.DataFrame, pd.DataFrame]] = {}
    for index, config in enumerate(configs, start=1):
        config_id = f"temporal_{index:04d}"
        augmented = add_temporal_candidate_consensus(
            CandidateFrame(candidate_rows),
            config=config,
        ).rows
        scored = augmented.copy()
        scored["candidate_reservoir_score"] = pd.to_numeric(
            scored["candidate_temporal_consensus_score"],
            errors="coerce",
        )
        frame_rows, pooled, _ = build_oracle_recall_tables(
            scored,
            truth_rows,
            top_k_values=top_k_tuple,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )
        if frame_rows.empty or pooled.empty:
            continue
        record = {
            "config_id": config_id,
            "temporal_consensus_config_json": json.dumps(asdict(config), sort_keys=True),
            **asdict(config),
            **pooled.iloc[0].to_dict(),
        }
        evaluation_rows.append(record)
        cache[config_id] = (config, augmented, frame_rows)

    grid_summary = pd.DataFrame.from_records(evaluation_rows)
    if grid_summary.empty:
        raise ValueError("temporal-consensus grid produced no oracle-recall rows")
    if selection_metric not in grid_summary.columns:
        raise ValueError(f"unknown selection metric: {selection_metric}")
    grid_summary = _sort_grid_summary(grid_summary, selection_metric=selection_metric)

    fold_records: list[dict[str, Any]] = []
    for holdout in sequences:
        best_config_id: str | None = None
        best_train_value = float("inf")
        for config_id, (_, _, frame_rows) in cache.items():
            train_frames = frame_rows.loc[frame_rows["sequence_id"].astype(str) != holdout]
            train_value = _metric_from_frame_rows(train_frames, selection_metric)
            if np.isfinite(train_value) and train_value < best_train_value:
                best_train_value = float(train_value)
                best_config_id = config_id
        if best_config_id is None:
            continue
        config, _, frame_rows = cache[best_config_id]
        holdout_frames = frame_rows.loc[frame_rows["sequence_id"].astype(str) == holdout]
        fold_record: dict[str, Any] = {
            "holdout_sequence_id": holdout,
            "selected_config_id": best_config_id,
            "train_selection_metric": selection_metric,
            "train_selection_metric_value": best_train_value,
            "holdout_selection_metric_value": _metric_from_frame_rows(
                holdout_frames,
                selection_metric,
            ),
            "temporal_consensus_config_json": json.dumps(asdict(config), sort_keys=True),
            **asdict(config),
            **_frame_metric_summary(holdout_frames, top_k_tuple),
        }
        fold_records.append(fold_record)

    fold_summary = pd.DataFrame.from_records(fold_records)
    selected_row = grid_summary.iloc[0]
    selected_config_id = str(selected_row["config_id"])
    selected_config, selected_candidates, _ = cache[selected_config_id]
    holdout_metric_values = (
        pd.to_numeric(fold_summary.get("holdout_selection_metric_value"), errors="coerce")
        if not fold_summary.empty
        else pd.Series(dtype=float)
    )
    holdout_metric_values = holdout_metric_values.loc[np.isfinite(holdout_metric_values)]

    selected_payload: dict[str, Any] = {
        "schema_version": _CONFIG_SCHEMA_VERSION,
        "selection_protocol": (
            "leave-one-sequence-out-cv-diagnostic__final-fit-on-all-train"
        ),
        "truth_free_at_inference": True,
        "sequence_count": len(sequences),
        "loso_fold_count": int(len(fold_summary)),
        "selection_metric": selection_metric,
        "selected_config_id": selected_config_id,
        "selected_metric_value": float(selected_row[selection_metric]),
        "loso_holdout_metric_mean": (
            float(holdout_metric_values.mean()) if len(holdout_metric_values) else None
        ),
        "loso_holdout_metric_p95": (
            float(holdout_metric_values.quantile(0.95))
            if len(holdout_metric_values)
            else None
        ),
        "top_k_values": [int(value) for value in top_k_tuple],
        "max_truth_time_delta_s": float(max_truth_time_delta_s),
        "temporal_consensus_config": asdict(selected_config),
    }
    return selected_payload, fold_summary, grid_summary, selected_candidates


def load_train_selected_temporal_consensus_config(path: Path) -> dict[str, Any]:
    """Load and validate a frozen temporal-consensus config JSON."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("temporal-consensus config JSON must contain an object")
    schema_version = int(payload.get("schema_version", _CONFIG_SCHEMA_VERSION))
    if schema_version != _CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported temporal-consensus config schema: {schema_version}")
    config_values = payload.get("temporal_consensus_config")
    if not isinstance(config_values, dict):
        raise ValueError("temporal-consensus config missing temporal_consensus_config")
    allowed = {field.name for field in fields(TemporalConsensusConfig)}
    unknown = sorted(set(config_values) - allowed)
    if unknown:
        raise ValueError(f"unknown temporal-consensus config keys: {unknown}")
    normalized = dict(payload)
    normalized["schema_version"] = schema_version
    normalized["temporal_consensus_config"] = {
        key: config_values[key] for key in config_values if key in allowed
    }
    return normalized


def apply_train_selected_temporal_consensus(
    candidates: CandidateFrame | pd.DataFrame,
    payload: dict[str, Any],
) -> CandidateFrame:
    """Apply a frozen train-selected temporal-consensus config without truth."""

    config_values = payload.get("temporal_consensus_config")
    if not isinstance(config_values, dict):
        raise ValueError("temporal-consensus payload missing temporal_consensus_config")
    config = TemporalConsensusConfig(**config_values)
    return add_temporal_candidate_consensus(candidates, config=config)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for train-only temporal-consensus grid selection."""

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
    selected, folds, grid, selected_candidates = (
        select_temporal_consensus_config_by_sequence_cv(
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


def apply_main(argv: list[str] | None = None) -> int:
    """CLI entry point for truth-free application of a frozen config."""

    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-apply-temporal-consensus-config",
        description="apply a train-selected MMUAD temporal-consensus config",
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--replace-confidence", action="store_true")
    args = parser.parse_args(argv)

    payload = load_train_selected_temporal_consensus_config(args.config_json)
    augmented = apply_train_selected_temporal_consensus(
        load_candidate_file(args.candidate_csv),
        payload,
    )
    rows = augmented.rows.copy()
    if args.replace_confidence and not rows.empty:
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence"), errors="coerce")
        rows["confidence"] = pd.to_numeric(
            rows["candidate_temporal_consensus_score"],
            errors="coerce",
        )
        augmented = CandidateFrame(normalize_candidate_columns(rows))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented.rows.to_csv(args.output_csv, index=False)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "truth_free": True,
            "config_json": str(args.config_json),
            "candidate_csv": str(args.candidate_csv),
            "output_csv": str(args.output_csv),
            "replace_confidence": bool(args.replace_confidence),
            "selected_config_id": payload.get("selected_config_id"),
            "selection_protocol": payload.get("selection_protocol"),
            "temporal_consensus_config": payload["temporal_consensus_config"],
            "summary": temporal_consensus_summary(augmented),
        }
        args.summary_json.write_text(
            json.dumps(_jsonable(summary), indent=2),
            encoding="utf-8",
        )

    print("mmuad_apply_temporal_consensus_config=ok")
    print(f"output_csv={args.output_csv}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def _config_grid(
    *,
    base_score_weights: Sequence[float],
    support_weights: Sequence[float],
    bidirectional_bonuses: Sequence[float],
    interpolation_weights: Sequence[float],
    acceleration_weights: Sequence[float],
    max_time_gap_s: float,
    max_speed_mps: float,
    distance_scale_m: float,
    acceleration_scale_mps2: float,
    score_column: str,
    fallback_score_column: str,
    source_diversity_bonus: float,
    branch_diversity_bonus: float,
) -> list[TemporalConsensusConfig]:
    configs: list[TemporalConsensusConfig] = []
    for base, support, bidirectional, interpolation, acceleration in product(
        _finite_unique(base_score_weights),
        _finite_unique(support_weights),
        _finite_unique(bidirectional_bonuses),
        _finite_unique(interpolation_weights),
        _finite_unique(acceleration_weights),
    ):
        configs.append(
            TemporalConsensusConfig(
                max_time_gap_s=float(max_time_gap_s),
                max_speed_mps=float(max_speed_mps),
                distance_scale_m=float(distance_scale_m),
                acceleration_scale_mps2=float(acceleration_scale_mps2),
                score_column=str(score_column),
                fallback_score_column=str(fallback_score_column),
                base_score_weight=float(base),
                backward_support_weight=float(support),
                forward_support_weight=float(support),
                bidirectional_bonus=float(bidirectional),
                interpolation_weight=float(interpolation),
                acceleration_weight=float(acceleration),
                source_diversity_bonus=float(source_diversity_bonus),
                branch_diversity_bonus=float(branch_diversity_bonus),
            )
        )
    return configs


def _sort_grid_summary(summary: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    values = pd.to_numeric(summary[selection_metric], errors="coerce")
    return (
        summary.assign(_selection_value=values)
        .sort_values(["_selection_value", "config_id"], na_position="last")
        .drop(columns=["_selection_value"])
        .reset_index(drop=True)
    )


def _metric_from_frame_rows(frame_rows: pd.DataFrame, metric: str) -> float:
    if frame_rows.empty:
        return float("nan")
    suffixes = ("_mse", "_rmse", "_p95", "_max")
    suffix = next((item for item in suffixes if metric.endswith(item)), None)
    if suffix is None:
        raise ValueError(f"unsupported temporal-consensus selection metric: {metric}")
    column = metric[: -len(suffix)]
    if column not in frame_rows.columns:
        raise ValueError(f"selection metric references missing frame column: {column}")
    values = pd.to_numeric(frame_rows[column], errors="coerce")
    values = values.loc[np.isfinite(values)]
    if values.empty:
        return float("nan")
    array = values.to_numpy(float)
    if suffix == "_mse":
        return float(np.mean(array**2))
    if suffix == "_rmse":
        return float(np.sqrt(np.mean(array**2)))
    if suffix == "_p95":
        return float(np.quantile(array, 0.95))
    return float(np.max(array))


def _frame_metric_summary(
    frame_rows: pd.DataFrame,
    top_k_values: Sequence[int],
) -> dict[str, Any]:
    record: dict[str, Any] = {"holdout_frame_count": int(len(frame_rows))}
    columns = ["oracle_all_3d_m"] + [f"oracle_top{int(k)}_3d_m" for k in top_k_values]
    for column in columns:
        for suffix in ("mse", "rmse", "p95", "max"):
            metric = f"{column}_{suffix}"
            record[metric] = _metric_from_frame_rows(frame_rows, metric)
    return record


def _parse_float_grid(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise ValueError("grid must contain at least one numeric value")
    return parsed


def _finite_unique(values: Sequence[float]) -> tuple[float, ...]:
    unique = sorted({float(value) for value in values if np.isfinite(float(value))})
    if not unique:
        raise ValueError("temporal-consensus grid axis must contain a finite value")
    return tuple(unique)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
