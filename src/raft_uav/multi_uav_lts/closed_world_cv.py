"""Guarded scenario-stratified tuning for closed-world Multi-UAV LTS tracking."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from ._config_ids import claim_config_id
from ._records import (
    prediction_texts,
    validate_nonnegative_finite,
    validate_nonnegative_int,
)
from .closed_world import postprocess_closed_world
from .fixed_population_cv import build_stratified_folds, scenario_prefix
from .metrics import BenchmarkMetrics, evaluate_lts_predictions


@dataclass(frozen=True)
class ClosedWorldFoldScore:
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
class ClosedWorldCVRow:
    rank: int
    selected: bool
    eligible: bool
    config_id: str
    is_raw: bool
    max_gap: int | None
    max_cost: float | None
    source_continuity_bonus: float | None
    emit_coasts: bool
    coast_max_gap: int
    mean_codabench_hota: float
    std_codabench_hota: float
    mean_codabench_mota: float
    mean_codabench_idf1: float
    hota_gain_vs_raw: float
    mota_delta_vs_raw: float
    idf1_delta_vs_raw: float
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
    dropped_candidate_rows: int
    absorbed_source_switches: int
    coasted_rows: int
    prediction_dir: str
    folds: tuple[ClosedWorldFoldScore, ...]


def run_closed_world_cv(
    prediction_path: Path,
    truth_dir: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    fold_count: int = 5,
    seed: int = 0,
    max_gaps: tuple[int, ...] = (5, 15, 30),
    max_costs: tuple[float, ...] = (1.5, 2.0, 2.5),
    source_continuity_bonuses: tuple[float, ...] = (0.0, 0.2, 0.4),
    coast_options: tuple[bool, ...] = (False, True),
    coast_max_gap: int = 2,
    max_mota_drop: float = 0.005,
    max_idf1_drop: float = 0.005,
    min_hota_gain: float = 0.0,
    sequences: tuple[str, ...] = (),
) -> tuple[ClosedWorldCVRow, ...]:
    """Rank closed-world configurations while retaining a raw fallback.

    The raw prediction set is an explicit candidate. A transformed candidate is
    eligible only when its held-out MOTA and IDF1 stay within the configured
    floors and its HOTA gain reaches ``min_hota_gain``. Consequently, the
    selected output cannot silently be the least-bad post-processing variant.
    """

    max_mota_drop = validate_nonnegative_finite(
        max_mota_drop, name="max_mota_drop"
    )
    max_idf1_drop = validate_nonnegative_finite(
        max_idf1_drop, name="max_idf1_drop"
    )
    min_hota_gain = validate_nonnegative_finite(
        min_hota_gain, name="min_hota_gain"
    )
    coast_max_gap = validate_nonnegative_int(
        coast_max_gap, name="coast_max_gap"
    )
    validated_gaps = tuple(
        validate_nonnegative_int(value, name="max_gap") for value in max_gaps
    )
    validated_costs = tuple(
        validate_nonnegative_finite(value, name="max_cost") for value in max_costs
    )
    validated_bonuses = tuple(
        validate_nonnegative_finite(value, name="source_continuity_bonus")
        for value in source_continuity_bonuses
    )
    _validate_paths(prediction_path, truth_dir, first_frame_label_dir, output_dir)
    available = _truth_sequences(truth_dir)
    selected_sequences = tuple(sorted(sequences or available))
    missing = sorted(set(selected_sequences) - set(available))
    if missing:
        raise ValueError(f"unknown truth sequences: {', '.join(missing)}")
    folds = build_stratified_folds(
        selected_sequences, fold_count=fold_count, seed=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_folds = _fold_scores(prediction_path, truth_dir, folds)
    raw_pooled = evaluate_lts_predictions(
        prediction_path, truth_dir, sequences=selected_sequences
    )
    raw_stats = _aggregate_fold_scores(raw_folds)
    raw_row: dict[str, object] = {
        "config_id": "raw",
        "is_raw": True,
        "max_gap": None,
        "max_cost": None,
        "source_continuity_bonus": None,
        "emit_coasts": False,
        "coast_max_gap": 0,
        **raw_stats,
        "pooled": raw_pooled,
        "output_rows": raw_pooled.predicted_detections,
        "dropped_candidate_rows": 0,
        "absorbed_source_switches": 0,
        "coasted_rows": 0,
        "prediction_dir": str(prediction_path),
        "folds": raw_folds,
        "grid_index": -1,
    }
    candidates: list[dict[str, object]] = [raw_row]
    used_config_ids = {"raw"}
    for grid_index, (max_gap, max_cost, continuity_bonus, emit_coasts) in enumerate(
        itertools.product(
            validated_gaps,
            validated_costs,
            validated_bonuses,
            tuple(bool(value) for value in coast_options),
        )
    ):
        effective_coast_gap = min(coast_max_gap, max_gap) if emit_coasts else 0
        base_id = _config_id(
            max_gap,
            max_cost,
            continuity_bonus,
            emit_coasts,
            effective_coast_gap,
        )
        config_id = claim_config_id(base_id, used_config_ids)
        prediction_dir = output_dir / "configs" / config_id / "predictions"
        summary = postprocess_closed_world(
            prediction_path,
            first_frame_label_dir,
            prediction_dir,
            max_gap=max_gap,
            max_cost=max_cost,
            source_continuity_bonus=continuity_bonus,
            emit_coasts=emit_coasts,
            coast_max_gap=effective_coast_gap,
            sequences=selected_sequences,
        )
        fold_scores = _fold_scores(prediction_dir, truth_dir, folds)
        pooled = evaluate_lts_predictions(
            prediction_dir, truth_dir, sequences=selected_sequences
        )
        candidates.append(
            {
                "config_id": config_id,
                "is_raw": False,
                "max_gap": max_gap,
                "max_cost": max_cost,
                "source_continuity_bonus": continuity_bonus,
                "emit_coasts": emit_coasts,
                "coast_max_gap": effective_coast_gap,
                **_aggregate_fold_scores(fold_scores),
                "pooled": pooled,
                "output_rows": summary.output_rows,
                "dropped_candidate_rows": summary.dropped_candidate_rows,
                "absorbed_source_switches": summary.absorbed_source_switches,
                "coasted_rows": summary.coasted_rows,
                "prediction_dir": str(prediction_dir),
                "folds": fold_scores,
                "grid_index": grid_index,
            }
        )

    candidates = _rank_candidates(
        candidates,
        max_mota_drop=max_mota_drop,
        max_idf1_drop=max_idf1_drop,
        min_hota_gain=min_hota_gain,
    )
    rows = tuple(
        _materialize_row(row, rank=rank, selected=rank == 1)
        for rank, row in enumerate(candidates, start=1)
    )
    if rows:
        _materialize_predictions(
            Path(rows[0].prediction_dir),
            output_dir / "best_predictions",
            selected_sequences,
        )
    _write_outputs(
        rows,
        folds,
        output_dir,
        seed=seed,
        max_mota_drop=max_mota_drop,
        max_idf1_drop=max_idf1_drop,
        min_hota_gain=min_hota_gain,
    )
    return rows


def _rank_candidates(
    candidates: list[dict[str, object]],
    *,
    max_mota_drop: float,
    max_idf1_drop: float,
    min_hota_gain: float,
) -> list[dict[str, object]]:
    """Apply raw-relative metric guards and return a deterministic ranking."""

    raw_candidates = [row for row in candidates if bool(row["is_raw"])]
    if len(raw_candidates) != 1:
        raise ValueError(
            "closed-world ranking requires exactly one raw candidate; "
            f"received {len(raw_candidates)}"
        )
    raw = raw_candidates[0]
    raw_hota = float(raw["mean_codabench_hota"])
    raw_mota = float(raw["mean_codabench_mota"])
    raw_idf1 = float(raw["mean_codabench_idf1"])
    ranked: list[dict[str, object]] = []
    for source_row in candidates:
        row = dict(source_row)
        hota_gain = float(row["mean_codabench_hota"]) - raw_hota
        mota_delta = float(row["mean_codabench_mota"]) - raw_mota
        idf1_delta = float(row["mean_codabench_idf1"]) - raw_idf1
        row["hota_gain_vs_raw"] = hota_gain
        row["mota_delta_vs_raw"] = mota_delta
        row["idf1_delta_vs_raw"] = idf1_delta
        row["eligible"] = bool(row["is_raw"]) or (
            hota_gain >= min_hota_gain
            and mota_delta >= -max_mota_drop
            and idf1_delta >= -max_idf1_drop
        )
        ranked.append(row)

    ranked.sort(
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["mean_codabench_hota"]),
            float(row["std_codabench_hota"]),
            -float(row["mean_codabench_idf1"]),
            -float(row["mean_codabench_mota"]),
            0 if bool(row["is_raw"]) else 1,
            int(row["grid_index"]),
        )
    )
    return ranked


def _truth_sequences(truth_dir: Path) -> tuple[str, ...]:
    if not truth_dir.exists():
        raise FileNotFoundError(f"truth directory does not exist: {truth_dir}")
    if not truth_dir.is_dir():
        raise NotADirectoryError(f"truth path is not a directory: {truth_dir}")
    sequences = tuple(path.stem for path in sorted(truth_dir.glob("*.txt")))
    if not sequences:
        raise ValueError(f"truth directory contains no .txt files: {truth_dir}")
    return sequences


def _validate_paths(
    prediction_path: Path,
    truth_dir: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
) -> None:
    output_resolved = output_dir.resolve()
    aliases = {
        "truth directory": truth_dir,
        "first-frame label directory": first_frame_label_dir,
    }
    if prediction_path.is_dir():
        aliases["prediction directory"] = prediction_path
    for label, path in aliases.items():
        if output_resolved == path.resolve():
            raise ValueError(f"output directory must differ from {label}: {output_dir}")


def _fold_scores(
    prediction_path: Path,
    truth_dir: Path,
    folds: tuple[tuple[str, ...], ...],
) -> tuple[ClosedWorldFoldScore, ...]:
    scores: list[ClosedWorldFoldScore] = []
    for fold_index, fold_sequences in enumerate(folds):
        metrics = evaluate_lts_predictions(
            prediction_path, truth_dir, sequences=fold_sequences
        )
        scores.append(_fold_score(fold_index, fold_sequences, metrics))
    return tuple(scores)


def _fold_score(
    fold: int,
    sequences: tuple[str, ...],
    metrics: BenchmarkMetrics,
) -> ClosedWorldFoldScore:
    return ClosedWorldFoldScore(
        fold=fold,
        sequences=sequences,
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


def _aggregate_fold_scores(
    scores: tuple[ClosedWorldFoldScore, ...],
) -> dict[str, float]:
    return {
        "mean_codabench_hota": _mean(score.codabench_hota for score in scores),
        "std_codabench_hota": _std(score.codabench_hota for score in scores),
        "mean_codabench_mota": _mean(score.codabench_mota for score in scores),
        "mean_codabench_idf1": _mean(score.codabench_idf1 for score in scores),
        "mean_hota": _mean(score.hota for score in scores),
        "std_hota": _std(score.hota for score in scores),
        "mean_deta": _mean(score.deta for score in scores),
        "mean_assa": _mean(score.assa for score in scores),
        "mean_loca": _mean(score.loca for score in scores),
        "mean_mota": _mean(score.mota for score in scores),
        "mean_idf1": _mean(score.idf1 for score in scores),
    }


def _materialize_row(
    row: dict[str, object], *, rank: int, selected: bool
) -> ClosedWorldCVRow:
    pooled = row["pooled"]
    return ClosedWorldCVRow(
        rank=rank,
        selected=selected,
        eligible=bool(row["eligible"]),
        config_id=str(row["config_id"]),
        is_raw=bool(row["is_raw"]),
        max_gap=None if row["max_gap"] is None else int(row["max_gap"]),
        max_cost=None if row["max_cost"] is None else float(row["max_cost"]),
        source_continuity_bonus=(
            None
            if row["source_continuity_bonus"] is None
            else float(row["source_continuity_bonus"])
        ),
        emit_coasts=bool(row["emit_coasts"]),
        coast_max_gap=int(row["coast_max_gap"]),
        mean_codabench_hota=float(row["mean_codabench_hota"]),
        std_codabench_hota=float(row["std_codabench_hota"]),
        mean_codabench_mota=float(row["mean_codabench_mota"]),
        mean_codabench_idf1=float(row["mean_codabench_idf1"]),
        hota_gain_vs_raw=float(row["hota_gain_vs_raw"]),
        mota_delta_vs_raw=float(row["mota_delta_vs_raw"]),
        idf1_delta_vs_raw=float(row["idf1_delta_vs_raw"]),
        mean_hota=float(row["mean_hota"]),
        std_hota=float(row["std_hota"]),
        mean_deta=float(row["mean_deta"]),
        mean_assa=float(row["mean_assa"]),
        mean_loca=float(row["mean_loca"]),
        mean_mota=float(row["mean_mota"]),
        mean_idf1=float(row["mean_idf1"]),
        pooled_codabench_hota=pooled.codabench_hota,
        pooled_codabench_mota=pooled.codabench_mota,
        pooled_codabench_idf1=pooled.codabench_idf1,
        pooled_hota=pooled.hota,
        pooled_mota=pooled.mota,
        pooled_idf1=pooled.idf1,
        output_rows=int(row["output_rows"]),
        dropped_candidate_rows=int(row["dropped_candidate_rows"]),
        absorbed_source_switches=int(row["absorbed_source_switches"]),
        coasted_rows=int(row["coasted_rows"]),
        prediction_dir=str(row["prediction_dir"]),
        folds=tuple(row["folds"]),
    )


def _config_id(
    max_gap: int,
    max_cost: float,
    continuity_bonus: float,
    emit_coasts: bool,
    coast_max_gap: int,
) -> str:
    return (
        f"cw_gap{max_gap}_cost{max_cost:.3f}_src{continuity_bonus:.3f}_"
        f"coast{int(emit_coasts)}_cgap{coast_max_gap}"
    ).replace(".", "p")


def _materialize_predictions(
    prediction_path: Path,
    output_dir: Path,
    sequences: tuple[str, ...],
) -> None:
    # Read before replacing the destination so a rerun can safely use the
    # previous best_predictions directory as its raw input.
    texts = prediction_texts(prediction_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for sequence in sequences:
        (output_dir / f"{sequence}.txt").write_text(
            texts.get(f"{sequence}.txt", ""), encoding="utf-8"
        )


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _std(values) -> float:
    materialized = list(values)
    return float(statistics.pstdev(materialized)) if len(materialized) > 1 else 0.0


def _write_outputs(
    rows: tuple[ClosedWorldCVRow, ...],
    folds: tuple[tuple[str, ...], ...],
    output_dir: Path,
    *,
    seed: int,
    max_mota_drop: float,
    max_idf1_drop: float,
    min_hota_gain: float,
) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-closed-world-cv-v1",
        "seed": seed,
        "guards": {
            "max_mota_drop": max_mota_drop,
            "max_idf1_drop": max_idf1_drop,
            "min_hota_gain": min_hota_gain,
        },
        "folds": [list(fold) for fold in folds],
        "best": asdict(rows[0]) if rows else None,
        "raw": asdict(next(row for row in rows if row.is_raw)) if rows else None,
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "cv_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = [
        field for field in ClosedWorldCVRow.__dataclass_fields__ if field != "folds"
    ]
    with (output_dir / "cv_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row_payload = asdict(row)
            writer.writerow({field: row_payload[field] for field in fields})
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
    parser.add_argument("--max-gaps", type=int, nargs="+", default=[5, 15, 30])
    parser.add_argument("--max-costs", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    parser.add_argument(
        "--source-continuity-bonuses",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4],
    )
    parser.add_argument(
        "--coast-options", choices=("off", "on", "both"), default="both"
    )
    parser.add_argument("--coast-max-gap", type=int, default=2)
    parser.add_argument("--max-mota-drop", type=float, default=0.005)
    parser.add_argument("--max-idf1-drop", type=float, default=0.005)
    parser.add_argument("--min-hota-gain", type=float, default=0.0)
    parser.add_argument("--sequences", nargs="*", default=[])
    args = parser.parse_args(argv)
    coast_options = {
        "off": (False,),
        "on": (True,),
        "both": (False, True),
    }[args.coast_options]
    rows = run_closed_world_cv(
        args.prediction_path,
        args.truth_dir,
        args.first_frame_label_dir,
        args.output_dir,
        fold_count=args.fold_count,
        seed=args.seed,
        max_gaps=tuple(args.max_gaps),
        max_costs=tuple(args.max_costs),
        source_continuity_bonuses=tuple(args.source_continuity_bonuses),
        coast_options=coast_options,
        coast_max_gap=args.coast_max_gap,
        max_mota_drop=args.max_mota_drop,
        max_idf1_drop=args.max_idf1_drop,
        min_hota_gain=args.min_hota_gain,
        sequences=tuple(args.sequences),
    )
    if rows:
        print(f"best_config={rows[0].config_id}")
        print(f"best_is_raw={rows[0].is_raw}")
        print(f"best_cv_CODABENCH_HOTA={rows[0].mean_codabench_hota:.6f}")
        print(f"best_cv_CODABENCH_MOTA={rows[0].mean_codabench_mota:.6f}")
        print(f"best_cv_CODABENCH_IDF1={rows[0].mean_codabench_idf1:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
