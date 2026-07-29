"""Scenario-stratified cross-validation for fixed-population LTS parameters."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from ._config_ids import claim_config_id
from .fixed_population import postprocess_fixed_population
from .metrics import evaluate_lts_predictions


@dataclass(frozen=True)
class FoldScore:
    fold: int
    sequences: tuple[str, ...]
    codabench_hota: float
    codabench_mota: float
    codabench_idf1: float
    hota: float
    deta: float
    assa: float
    loca: float
    mota: float
    idf1: float


@dataclass(frozen=True)
class CrossValidationRow:
    rank: int
    config_id: str
    min_seed_iou: float
    relink_max_gap: int
    relink_max_cost: float
    interpolate_single_frame: bool
    mean_codabench_hota: float
    std_codabench_hota: float
    mean_codabench_mota: float
    mean_codabench_idf1: float
    mean_hota: float
    std_hota: float
    mean_deta: float
    mean_assa: float
    mean_loca: float
    mean_mota: float
    mean_idf1: float
    pooled_codabench_hota: float
    pooled_codabench_mota: float
    pooled_codabench_idf1: float
    pooled_hota: float
    pooled_mota: float
    pooled_idf1: float
    output_rows: int
    dropped_input_tracks: int
    relinked_tracklets: int
    interpolated_rows: int
    prediction_dir: str
    folds: tuple[FoldScore, ...]


def scenario_prefix(sequence: str) -> str:
    """Return the benchmark scenario prefix used for stratification."""

    prefix = sequence.split("_", 1)[0].strip()
    return prefix or sequence


def build_stratified_folds(
    sequences: tuple[str, ...], *, fold_count: int = 5, seed: int = 0
) -> tuple[tuple[str, ...], ...]:
    """Distribute every scenario group round-robin across deterministic folds."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    unique_sequences = set(sequences)
    if len(unique_sequences) != len(sequences):
        duplicates = sorted(
            sequence for sequence in unique_sequences if sequences.count(sequence) > 1
        )
        raise ValueError(f"duplicate sequence names: {', '.join(duplicates)}")
    if fold_count > len(unique_sequences):
        raise ValueError("fold_count cannot exceed the sequence count")
    by_scenario: dict[str, list[str]] = {}
    for sequence in sorted(unique_sequences):
        by_scenario.setdefault(scenario_prefix(sequence), []).append(sequence)
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    generator = random.Random(seed)
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        generator.shuffle(group)
        start = min(range(fold_count), key=lambda index: (len(folds[index]), index))
        for offset, sequence in enumerate(group):
            folds[(start + offset) % fold_count].append(sequence)
    return tuple(tuple(sorted(fold)) for fold in folds)


def run_fixed_population_cv(
    prediction_path: Path,
    truth_dir: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    fold_count: int = 5,
    seed: int = 0,
    min_seed_ious: tuple[float, ...] = (0.3, 0.5, 0.7),
    relink_max_gaps: tuple[int, ...] = (0, 2, 5, 15),
    relink_max_costs: tuple[float, ...] = (1.0, 1.5, 2.0),
    interpolation_options: tuple[bool, ...] = (False, True),
    sequences: tuple[str, ...] = (),
) -> tuple[CrossValidationRow, ...]:
    """Rank configurations by mean held-out HOTA over stratified folds."""

    available = tuple(path.stem for path in sorted(truth_dir.glob("*.txt")))
    selected = tuple(sorted(sequences or available))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"unknown truth sequences: {', '.join(missing)}")
    folds = build_stratified_folds(selected, fold_count=fold_count, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    used_config_ids: set[str] = set()
    for grid_index, config in enumerate(
        _configurations(
            min_seed_ious,
            relink_max_gaps,
            relink_max_costs,
            interpolation_options,
        )
    ):
        min_iou, max_gap, max_cost, interpolate = config
        base_config_id = _config_id(min_iou, max_gap, max_cost, interpolate)
        config_id = claim_config_id(base_config_id, used_config_ids)
        prediction_dir = output_dir / "configs" / config_id / "predictions"
        summary = postprocess_fixed_population(
            prediction_path,
            first_frame_label_dir,
            prediction_dir,
            min_seed_iou=min_iou,
            relink_max_gap=max_gap,
            relink_max_cost=max_cost,
            interpolate_single_frame=interpolate,
            sequences=selected,
        )
        fold_scores: list[FoldScore] = []
        for fold_index, fold_sequences in enumerate(folds):
            metrics = evaluate_lts_predictions(
                prediction_dir,
                truth_dir,
                sequences=fold_sequences,
            )
            fold_scores.append(
                FoldScore(
                    fold=fold_index,
                    sequences=fold_sequences,
                    codabench_hota=metrics.codabench_hota,
                    codabench_mota=metrics.codabench_mota,
                    codabench_idf1=metrics.codabench_idf1,
                    hota=metrics.hota,
                    deta=metrics.deta,
                    assa=metrics.assa,
                    loca=metrics.loca,
                    mota=metrics.mota,
                    idf1=metrics.idf1,
                )
            )
        pooled = evaluate_lts_predictions(
            prediction_dir,
            truth_dir,
            sequences=selected,
        )
        raw_rows.append(
            {
                "config_id": config_id,
                "min_seed_iou": min_iou,
                "relink_max_gap": max_gap,
                "relink_max_cost": max_cost,
                "interpolate_single_frame": interpolate,
                "mean_codabench_hota": _mean(
                    score.codabench_hota for score in fold_scores
                ),
                "std_codabench_hota": _std(
                    score.codabench_hota for score in fold_scores
                ),
                "mean_codabench_mota": _mean(
                    score.codabench_mota for score in fold_scores
                ),
                "mean_codabench_idf1": _mean(
                    score.codabench_idf1 for score in fold_scores
                ),
                "mean_hota": _mean(score.hota for score in fold_scores),
                "std_hota": _std(score.hota for score in fold_scores),
                "mean_deta": _mean(score.deta for score in fold_scores),
                "mean_assa": _mean(score.assa for score in fold_scores),
                "mean_loca": _mean(score.loca for score in fold_scores),
                "mean_mota": _mean(score.mota for score in fold_scores),
                "mean_idf1": _mean(score.idf1 for score in fold_scores),
                "pooled_codabench_hota": pooled.codabench_hota,
                "pooled_codabench_mota": pooled.codabench_mota,
                "pooled_codabench_idf1": pooled.codabench_idf1,
                "pooled_hota": pooled.hota,
                "pooled_mota": pooled.mota,
                "pooled_idf1": pooled.idf1,
                "output_rows": summary.output_rows,
                "dropped_input_tracks": summary.dropped_input_tracks,
                "relinked_tracklets": summary.relinked_tracklets,
                "interpolated_rows": summary.interpolated_rows,
                "prediction_dir": str(prediction_dir),
                "folds": tuple(fold_scores),
                "grid_index": grid_index,
            }
        )
    raw_rows.sort(
        key=lambda row: (
            -float(row["mean_codabench_hota"]),
            float(row["std_codabench_hota"]),
            -float(row["mean_codabench_idf1"]),
            -float(row["mean_codabench_mota"]),
            int(row["grid_index"]),
        )
    )
    rows = tuple(
        CrossValidationRow(
            rank=rank,
            config_id=str(row["config_id"]),
            min_seed_iou=float(row["min_seed_iou"]),
            relink_max_gap=int(row["relink_max_gap"]),
            relink_max_cost=float(row["relink_max_cost"]),
            interpolate_single_frame=bool(row["interpolate_single_frame"]),
            mean_codabench_hota=float(row["mean_codabench_hota"]),
            std_codabench_hota=float(row["std_codabench_hota"]),
            mean_codabench_mota=float(row["mean_codabench_mota"]),
            mean_codabench_idf1=float(row["mean_codabench_idf1"]),
            mean_hota=float(row["mean_hota"]),
            std_hota=float(row["std_hota"]),
            mean_deta=float(row["mean_deta"]),
            mean_assa=float(row["mean_assa"]),
            mean_loca=float(row["mean_loca"]),
            mean_mota=float(row["mean_mota"]),
            mean_idf1=float(row["mean_idf1"]),
            pooled_codabench_hota=float(row["pooled_codabench_hota"]),
            pooled_codabench_mota=float(row["pooled_codabench_mota"]),
            pooled_codabench_idf1=float(row["pooled_codabench_idf1"]),
            pooled_hota=float(row["pooled_hota"]),
            pooled_mota=float(row["pooled_mota"]),
            pooled_idf1=float(row["pooled_idf1"]),
            output_rows=int(row["output_rows"]),
            dropped_input_tracks=int(row["dropped_input_tracks"]),
            relinked_tracklets=int(row["relinked_tracklets"]),
            interpolated_rows=int(row["interpolated_rows"]),
            prediction_dir=str(row["prediction_dir"]),
            folds=tuple(row["folds"]),
        )
        for rank, row in enumerate(raw_rows, start=1)
    )
    if rows:
        target = output_dir / "best_predictions"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(Path(rows[0].prediction_dir), target)
    _write_outputs(rows, folds, output_dir, seed=seed)
    return rows


def _configurations(
    min_seed_ious: tuple[float, ...],
    relink_max_gaps: tuple[int, ...],
    relink_max_costs: tuple[float, ...],
    interpolation_options: tuple[bool, ...],
):
    for min_iou, max_gap, interpolate in itertools.product(
        min_seed_ious, relink_max_gaps, interpolation_options
    ):
        costs = relink_max_costs if max_gap > 0 else relink_max_costs[:1]
        for max_cost in costs:
            yield min_iou, max_gap, max_cost, interpolate


def _config_id(
    min_iou: float, max_gap: int, max_cost: float, interpolate: bool
) -> str:
    return (
        f"seed{min_iou:.3f}_gap{max_gap}_cost{max_cost:.3f}_"
        f"interp{int(interpolate)}"
    ).replace(".", "p")


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _std(values) -> float:
    materialized = list(values)
    return float(statistics.pstdev(materialized)) if len(materialized) > 1 else 0.0


def _write_outputs(
    rows: tuple[CrossValidationRow, ...],
    folds: tuple[tuple[str, ...], ...],
    output_dir: Path,
    *,
    seed: int,
) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-fixed-population-cv-v1",
        "seed": seed,
        "folds": [list(fold) for fold in folds],
        "best": asdict(rows[0]) if rows else None,
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "cv_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    ranking_fields = [
        field for field in CrossValidationRow.__dataclass_fields__ if field != "folds"
    ]
    with (output_dir / "cv_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_fields)
        writer.writeheader()
        for row in rows:
            payload_row = asdict(row)
            writer.writerow({field: payload_row[field] for field in ranking_fields})
    with (output_dir / "fold_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "scenario", "sequence"])
        for fold_index, fold in enumerate(folds):
            for sequence in fold:
                writer.writerow([fold_index, scenario_prefix(sequence), sequence])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-seed-ious", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--relink-max-gaps", type=int, nargs="+", default=[0, 2, 5, 15])
    parser.add_argument("--relink-max-costs", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument(
        "--interpolation-options",
        choices=("off", "on", "both"),
        default="both",
    )
    parser.add_argument("--sequences", nargs="*", default=[])
    args = parser.parse_args(argv)
    interpolation_options = {
        "off": (False,),
        "on": (True,),
        "both": (False, True),
    }[args.interpolation_options]
    rows = run_fixed_population_cv(
        args.prediction_path,
        args.truth_dir,
        args.first_frame_label_dir,
        args.output_dir,
        fold_count=args.fold_count,
        seed=args.seed,
        min_seed_ious=tuple(args.min_seed_ious),
        relink_max_gaps=tuple(args.relink_max_gaps),
        relink_max_costs=tuple(args.relink_max_costs),
        interpolation_options=interpolation_options,
        sequences=tuple(args.sequences),
    )
    if rows:
        print(f"best_config={rows[0].config_id}")
        print(f"best_cv_CODABENCH_HOTA={rows[0].mean_codabench_hota:.6f}")
        print(
            f"best_cv_CODABENCH_HOTA_std={rows[0].std_codabench_hota:.6f}"
        )
        print(f"best_cv_CODABENCH_IDF1={rows[0].mean_codabench_idf1:.6f}")
        print(f"best_cv_CODABENCH_MOTA={rows[0].mean_codabench_mota:.6f}")
        print(f"best_cv_standard_HOTA={rows[0].mean_hota:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
