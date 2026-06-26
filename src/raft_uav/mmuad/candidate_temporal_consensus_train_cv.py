"""Train-only model selection for MMUAD temporal candidate consensus.

Temporal consensus is truth-free at inference, but its speed gate, distance
scale, and score weights still need to be chosen without using validation or
test labels. This module evaluates candidate configurations on training
sequences, reports leave-one-sequence-out selection stability, and writes one
frozen configuration for inference.
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

from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    add_temporal_candidate_consensus,
    temporal_consensus_summary,
    write_temporal_consensus_outputs,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.cluster_ranker import label_cluster_features_against_truth
from raft_uav.mmuad.schema import (
    CandidateFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)

SELECTION_SCHEMA = "raft-uav-mmuad-temporal-consensus-train-cv-v1"
SELECTION_METRICS = (
    "top1_3d_m_mse",
    "top1_3d_m_mean",
    "candidate_regret_3d_m_mean",
)
_DEFAULT_MAX_SPEED_MPS = (40.0, 70.0)
_DEFAULT_DISTANCE_SCALE_M = (3.0, 8.0)
_DEFAULT_BASE_SCORE_WEIGHT = (0.0, 0.25)
_DEFAULT_BIDIRECTIONAL_BONUS = (0.5, 1.0)


def build_temporal_consensus_config_grid(
    *,
    max_speed_mps_values: Sequence[float] = _DEFAULT_MAX_SPEED_MPS,
    distance_scale_m_values: Sequence[float] = _DEFAULT_DISTANCE_SCALE_M,
    base_score_weight_values: Sequence[float] = _DEFAULT_BASE_SCORE_WEIGHT,
    bidirectional_bonus_values: Sequence[float] = _DEFAULT_BIDIRECTIONAL_BONUS,
    max_time_gap_s: float = 2.0,
    acceleration_scale_mps2: float = 20.0,
    score_column: str = "ranker_score",
    fallback_score_column: str = "confidence",
    backward_support_weight: float = 1.0,
    forward_support_weight: float = 1.0,
    interpolation_weight: float = 0.75,
    acceleration_weight: float = 0.5,
    source_diversity_bonus: float = 0.25,
    branch_diversity_bonus: float = 0.25,
) -> list[TemporalConsensusConfig]:
    """Build a deterministic, de-duplicated temporal-consensus grid."""

    speeds = _positive_values(max_speed_mps_values, name="max_speed_mps")
    distances = _positive_values(distance_scale_m_values, name="distance_scale_m")
    base_weights = _finite_values(base_score_weight_values, name="base_score_weight")
    bidirectional = _finite_values(
        bidirectional_bonus_values,
        name="bidirectional_bonus",
    )
    configs = [
        TemporalConsensusConfig(
            max_time_gap_s=float(max_time_gap_s),
            max_speed_mps=max_speed,
            distance_scale_m=distance_scale,
            acceleration_scale_mps2=float(acceleration_scale_mps2),
            score_column=str(score_column),
            fallback_score_column=str(fallback_score_column),
            base_score_weight=base_weight,
            backward_support_weight=float(backward_support_weight),
            forward_support_weight=float(forward_support_weight),
            bidirectional_bonus=bidirectional_bonus,
            interpolation_weight=float(interpolation_weight),
            acceleration_weight=float(acceleration_weight),
            source_diversity_bonus=float(source_diversity_bonus),
            branch_diversity_bonus=float(branch_diversity_bonus),
        )
        for max_speed, distance_scale, base_weight, bidirectional_bonus in product(
            speeds,
            distances,
            base_weights,
            bidirectional,
        )
    ]
    return _deduplicate_configs(configs)


def select_temporal_consensus_config_by_sequence_cv(
    candidates: CandidateFrame | pd.DataFrame,
    truth: pd.DataFrame,
    *,
    configs: Sequence[TemporalConsensusConfig] | None = None,
    selection_metric: str = "top1_3d_m_mse",
    max_truth_time_delta_s: float = 0.5,
) -> tuple[
    TemporalConsensusConfig,
    pd.DataFrame,
    pd.DataFrame,
    CandidateFrame,
    dict[str, Any],
]:
    """Select one temporal-consensus config using training sequences only.

    Every configuration is scored per frame. For each LOSO fold, the best
    configuration on the remaining sequences is evaluated on the held-out
    sequence. The final frozen configuration is selected using all supplied
    training sequences.
    """

    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"unsupported selection_metric={selection_metric!r}; "
            f"expected one of {SELECTION_METRICS}"
        )
    candidate_frame = _candidate_frame(candidates)
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    sequences = sorted(
        set(candidate_frame.rows["sequence_id"].astype(str))
        & set(truth_rows["sequence_id"].astype(str))
    )
    if len(sequences) < 2:
        raise ValueError("temporal consensus train-CV requires at least two sequences")

    config_grid = _deduplicate_configs(
        list(configs) if configs is not None else build_temporal_consensus_config_grid()
    )
    if not config_grid:
        raise ValueError("temporal consensus train-CV requires at least one config")

    frame_parts: list[pd.DataFrame] = []
    config_records: list[dict[str, Any]] = []
    for config_index, config in enumerate(config_grid, start=1):
        augmented = add_temporal_candidate_consensus(candidate_frame, config=config)
        labeled = label_cluster_features_against_truth(
            augmented.rows,
            truth_rows,
            max_truth_time_delta_s=float(max_truth_time_delta_s),
        )
        selected_frames = _selected_frame_rows(labeled)
        selected_frames = selected_frames.loc[
            selected_frames["sequence_id"].astype(str).isin(sequences)
        ].copy()
        if selected_frames.empty:
            continue
        selected_frames["config_index"] = int(config_index)
        frame_parts.append(selected_frames)
        config_records.append(
            {
                "config_index": int(config_index),
                **_config_summary_fields(config),
                "config_json": json.dumps(asdict(config), sort_keys=True),
            }
        )

    if not frame_parts:
        raise ValueError("no temporal-consensus configurations produced labeled frames")
    frame_rows = pd.concat(frame_parts, ignore_index=True)
    config_table = pd.DataFrame.from_records(config_records).drop_duplicates("config_index")
    available_sequences = sorted(frame_rows["sequence_id"].astype(str).unique())
    if len(available_sequences) < 2:
        raise ValueError("fewer than two sequences produced temporal-consensus metrics")

    grid_summary = _aggregate_config_frames(frame_rows).merge(
        config_table,
        on="config_index",
        how="left",
        validate="one_to_one",
    )
    grid_summary = _rank_summaries(grid_summary, selection_metric=selection_metric)

    fold_records: list[dict[str, Any]] = []
    for holdout_sequence in available_sequences:
        train_frames = frame_rows.loc[
            frame_rows["sequence_id"].astype(str) != holdout_sequence
        ]
        holdout_frames = frame_rows.loc[
            frame_rows["sequence_id"].astype(str) == holdout_sequence
        ]
        train_summary = _aggregate_config_frames(train_frames).merge(
            config_table,
            on="config_index",
            how="left",
            validate="one_to_one",
        )
        if train_summary.empty:
            continue
        train_summary = _rank_summaries(
            train_summary,
            selection_metric=selection_metric,
        )
        selected_index = int(train_summary.iloc[0]["config_index"])
        holdout_selected = holdout_frames.loc[
            holdout_frames["config_index"] == selected_index
        ]
        if holdout_selected.empty:
            continue
        holdout_metrics = _aggregate_frame_metrics(holdout_selected)
        selected_config_row = config_table.loc[
            config_table["config_index"] == selected_index
        ].iloc[0]
        fold_records.append(
            {
                "holdout_sequence_id": str(holdout_sequence),
                "selected_config_index": selected_index,
                "selection_metric": selection_metric,
                "train_selection_metric_value": float(
                    train_summary.iloc[0][selection_metric]
                ),
                "train_frame_count": int(train_summary.iloc[0]["frame_count"]),
                **{
                    key: selected_config_row[key]
                    for key in _config_summary_field_names()
                },
                **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
            }
        )

    folds = pd.DataFrame.from_records(fold_records)
    if folds.empty:
        raise ValueError("no temporal-consensus LOSO folds could be evaluated")

    selected_row = grid_summary.iloc[0]
    selected_config_index = int(selected_row["config_index"])
    selected_config = config_grid[selected_config_index - 1]
    selected_candidates = add_temporal_candidate_consensus(
        candidate_frame,
        config=selected_config,
    )
    selected_metrics = {
        key: _json_number(selected_row[key])
        for key in _metric_columns()
        if key in selected_row
    }
    provenance = {
        "schema": SELECTION_SCHEMA,
        "selection_protocol": (
            "leave-one-sequence-out-diagnostic__final-selection-on-all-training-sequences"
        ),
        "selection_metric": selection_metric,
        "sequence_ids": available_sequences,
        "sequence_count": int(len(available_sequences)),
        "config_count": int(len(config_table)),
        "selected_config_index": selected_config_index,
        "selected_config": asdict(selected_config),
        "selected_training_metrics": selected_metrics,
        "loso_fold_count": int(len(folds)),
        "loso_selected_config_counts": {
            str(key): int(value)
            for key, value in folds["selected_config_index"].value_counts().sort_index().items()
        },
        "max_truth_time_delta_s": float(max_truth_time_delta_s),
    }
    return selected_config, folds, grid_summary, selected_candidates, provenance


def save_temporal_consensus_selection(
    provenance: dict[str, Any],
    path: Path,
) -> Path:
    """Write a train-selected temporal-consensus configuration JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_temporal_consensus_selection(path: Path) -> TemporalConsensusConfig:
    """Load the selected configuration from a selection or direct config JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema is not None and schema != SELECTION_SCHEMA:
        raise ValueError(f"unsupported temporal consensus selection schema: {schema!r}")
    config_payload = payload.get("selected_config", payload)
    if not isinstance(config_payload, dict):
        raise ValueError("temporal consensus config JSON must contain an object")
    allowed = {field.name for field in fields(TemporalConsensusConfig)}
    filtered = {key: value for key, value in config_payload.items() if key in allowed}
    if not filtered:
        raise ValueError("temporal consensus config JSON contains no recognized fields")
    return TemporalConsensusConfig(**filtered)


def select_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-temporal-consensus-train-cv",
        description="select temporal-consensus hyperparameters on training sequences",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default=SELECTION_METRICS[0],
    )
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--max-time-gap-s", type=float, default=2.0)
    parser.add_argument("--max-speed-mps", type=float, action="append", default=[])
    parser.add_argument("--distance-scale-m", type=float, action="append", default=[])
    parser.add_argument("--base-score-weight", type=float, action="append", default=[])
    parser.add_argument("--bidirectional-bonus", type=float, action="append", default=[])
    parser.add_argument("--acceleration-scale-mps2", type=float, default=20.0)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--backward-support-weight", type=float, default=1.0)
    parser.add_argument("--forward-support-weight", type=float, default=1.0)
    parser.add_argument("--interpolation-weight", type=float, default=0.75)
    parser.add_argument("--acceleration-weight", type=float, default=0.5)
    parser.add_argument("--source-diversity-bonus", type=float, default=0.25)
    parser.add_argument("--branch-diversity-bonus", type=float, default=0.25)
    parser.add_argument("--write-selected-candidates", action="store_true")
    args = parser.parse_args(argv)

    configs = build_temporal_consensus_config_grid(
        max_speed_mps_values=tuple(args.max_speed_mps) or _DEFAULT_MAX_SPEED_MPS,
        distance_scale_m_values=(
            tuple(args.distance_scale_m) or _DEFAULT_DISTANCE_SCALE_M
        ),
        base_score_weight_values=(
            tuple(args.base_score_weight) or _DEFAULT_BASE_SCORE_WEIGHT
        ),
        bidirectional_bonus_values=(
            tuple(args.bidirectional_bonus) or _DEFAULT_BIDIRECTIONAL_BONUS
        ),
        max_time_gap_s=args.max_time_gap_s,
        acceleration_scale_mps2=args.acceleration_scale_mps2,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        backward_support_weight=args.backward_support_weight,
        forward_support_weight=args.forward_support_weight,
        interpolation_weight=args.interpolation_weight,
        acceleration_weight=args.acceleration_weight,
        source_diversity_bonus=args.source_diversity_bonus,
        branch_diversity_bonus=args.branch_diversity_bonus,
    )
    _, folds, grid_summary, selected_candidates, provenance = (
        select_temporal_consensus_config_by_sequence_cv(
            load_candidate_file(args.candidate_csv),
            load_evaluation_truth_file(args.truth_csv).rows,
            configs=configs,
            selection_metric=args.selection_metric,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_json = output_dir / "mmuad_temporal_consensus_train_selected_config.json"
    folds_csv = output_dir / "mmuad_temporal_consensus_train_cv_folds.csv"
    grid_csv = output_dir / "mmuad_temporal_consensus_train_grid_summary.csv"
    save_temporal_consensus_selection(provenance, config_json)
    folds.to_csv(folds_csv, index=False)
    grid_summary.to_csv(grid_csv, index=False)
    if args.write_selected_candidates:
        selected_candidates.rows.to_csv(
            output_dir / "mmuad_temporal_consensus_train_selected_candidates.csv",
            index=False,
        )

    print("mmuad_temporal_consensus_train_cv=ok")
    print(f"selected_config_json={config_json}")
    print(f"folds_csv={folds_csv}")
    print(f"grid_summary_csv={grid_csv}")
    print(f"selected_config_index={provenance['selected_config_index']}")
    return 0


def apply_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-apply-temporal-consensus-config",
        description="apply a train-selected temporal-consensus configuration",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--replace-confidence", action="store_true")
    args = parser.parse_args(argv)

    config = load_temporal_consensus_selection(args.config_json)
    augmented = add_temporal_candidate_consensus(
        load_candidate_file(args.candidate_csv),
        config=config,
    )
    if args.replace_confidence and not augmented.rows.empty:
        rows = augmented.rows.copy()
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence"), errors="coerce")
        rows["confidence"] = pd.to_numeric(
            rows["candidate_temporal_consensus_score"],
            errors="coerce",
        )
        augmented = CandidateFrame(normalize_candidate_columns(rows))
    write_temporal_consensus_outputs(
        augmented,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        provenance={
            "config_json": str(args.config_json),
            "config": asdict(config),
            "replace_confidence": bool(args.replace_confidence),
            "summary": temporal_consensus_summary(augmented),
        },
    )
    print("mmuad_apply_temporal_consensus_config=ok")
    print(f"output_csv={args.output_csv}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the train-CV selector when invoked as a module."""

    return select_main(argv)


def _candidate_frame(candidates: CandidateFrame | pd.DataFrame) -> CandidateFrame:
    rows = (
        candidates.rows.copy()
        if isinstance(candidates, CandidateFrame)
        else pd.DataFrame(candidates).copy()
    )
    frame = CandidateFrame(normalize_candidate_columns(rows))
    frame.validate()
    return frame


def _selected_frame_rows(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(labeled).copy()
    score_column = "candidate_temporal_consensus_score"
    required = {"sequence_id", "time_s", "truth_distance_3d_m", score_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"temporal consensus evaluation missing columns: {sorted(missing)}")
    rows["truth_distance_3d_m"] = pd.to_numeric(
        rows["truth_distance_3d_m"],
        errors="coerce",
    )
    rows[score_column] = pd.to_numeric(rows[score_column], errors="coerce")
    finite = np.isfinite(rows["truth_distance_3d_m"].to_numpy(float))
    finite &= np.isfinite(rows[score_column].to_numpy(float))
    rows = rows.loc[finite].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "time_s",
                "selected_truth_distance_3d_m",
                "oracle_truth_distance_3d_m",
                "candidate_regret_3d_m",
                "selected_temporal_consensus_score",
            ]
        )
    rows["_stable_row_order"] = np.arange(len(rows), dtype=int)
    selected = (
        rows.sort_values(
            [
                "sequence_id",
                "time_s",
                score_column,
                "_stable_row_order",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby(["sequence_id", "time_s"], sort=False)
        .head(1)
        .copy()
    )
    oracle = (
        rows.groupby(["sequence_id", "time_s"], sort=False)["truth_distance_3d_m"]
        .min()
        .rename("oracle_truth_distance_3d_m")
        .reset_index()
    )
    selected = selected[
        ["sequence_id", "time_s", "truth_distance_3d_m", score_column]
    ].rename(
        columns={
            "truth_distance_3d_m": "selected_truth_distance_3d_m",
            score_column: "selected_temporal_consensus_score",
        }
    )
    result = selected.merge(
        oracle,
        on=["sequence_id", "time_s"],
        how="inner",
        validate="one_to_one",
    )
    result["candidate_regret_3d_m"] = (
        result["selected_truth_distance_3d_m"]
        - result["oracle_truth_distance_3d_m"]
    )
    result["sequence_id"] = result["sequence_id"].astype(str)
    return result


def _aggregate_config_frames(frame_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_index, group in frame_rows.groupby("config_index", sort=True):
        records.append(
            {
                "config_index": int(config_index),
                **_aggregate_frame_metrics(group),
            }
        )
    return pd.DataFrame.from_records(records)


def _aggregate_frame_metrics(frame_rows: pd.DataFrame) -> dict[str, float | int]:
    selected = pd.to_numeric(
        frame_rows["selected_truth_distance_3d_m"],
        errors="coerce",
    )
    oracle = pd.to_numeric(
        frame_rows["oracle_truth_distance_3d_m"],
        errors="coerce",
    )
    regret = pd.to_numeric(frame_rows["candidate_regret_3d_m"], errors="coerce")
    score = pd.to_numeric(
        frame_rows["selected_temporal_consensus_score"],
        errors="coerce",
    )
    finite = selected.notna() & oracle.notna() & regret.notna() & score.notna()
    if not finite.any():
        return {
            "frame_count": 0,
            "top1_3d_m_mean": float("nan"),
            "top1_3d_m_mse": float("nan"),
            "top1_3d_m_p95": float("nan"),
            "top1_within_5m": float("nan"),
            "oracle_3d_m_mean": float("nan"),
            "candidate_regret_3d_m_mean": float("nan"),
            "selected_score_mean": float("nan"),
        }
    selected_values = selected.loc[finite].to_numpy(float)
    oracle_values = oracle.loc[finite].to_numpy(float)
    regret_values = regret.loc[finite].to_numpy(float)
    score_values = score.loc[finite].to_numpy(float)
    return {
        "frame_count": int(finite.sum()),
        "top1_3d_m_mean": float(np.mean(selected_values)),
        "top1_3d_m_mse": float(np.mean(selected_values**2)),
        "top1_3d_m_p95": float(np.quantile(selected_values, 0.95)),
        "top1_within_5m": float(np.mean(selected_values <= 5.0)),
        "oracle_3d_m_mean": float(np.mean(oracle_values)),
        "candidate_regret_3d_m_mean": float(np.mean(regret_values)),
        "selected_score_mean": float(np.mean(score_values)),
    }


def _rank_summaries(
    summary: pd.DataFrame,
    *,
    selection_metric: str,
) -> pd.DataFrame:
    if summary.empty:
        return summary
    sort_columns = []
    for column in (
        selection_metric,
        "candidate_regret_3d_m_mean",
        "top1_3d_m_mean",
        "config_index",
    ):
        if column not in sort_columns:
            sort_columns.append(column)
    ranked = summary.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    ranked.insert(1, "selection_metric", selection_metric)
    return ranked


def _config_summary_fields(config: TemporalConsensusConfig) -> dict[str, Any]:
    return {
        "max_time_gap_s": float(config.max_time_gap_s),
        "max_speed_mps": float(config.max_speed_mps),
        "distance_scale_m": float(config.distance_scale_m),
        "base_score_weight": float(config.base_score_weight),
        "bidirectional_bonus": float(config.bidirectional_bonus),
    }


def _config_summary_field_names() -> tuple[str, ...]:
    return tuple(_config_summary_fields(TemporalConsensusConfig()).keys())


def _metric_columns() -> tuple[str, ...]:
    return (
        "frame_count",
        "top1_3d_m_mean",
        "top1_3d_m_mse",
        "top1_3d_m_p95",
        "top1_within_5m",
        "oracle_3d_m_mean",
        "candidate_regret_3d_m_mean",
        "selected_score_mean",
    )


def _deduplicate_configs(
    configs: Sequence[TemporalConsensusConfig],
) -> list[TemporalConsensusConfig]:
    unique: dict[str, TemporalConsensusConfig] = {}
    for config in configs:
        key = json.dumps(asdict(config), sort_keys=True)
        unique.setdefault(key, config)
    return list(unique.values())


def _positive_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    finite = _finite_values(values, name=name)
    if any(value <= 0.0 for value in finite):
        raise ValueError(f"{name} values must be positive")
    return finite


def _finite_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    unique = sorted({float(value) for value in values})
    if not unique:
        raise ValueError(f"{name} grid must not be empty")
    if not np.isfinite(np.asarray(unique, dtype=float)).all():
        raise ValueError(f"{name} values must be finite")
    return tuple(unique)


def _json_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
