"""Tune fixed-population LTS post-processing against HOTA on training data."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .fixed_population import postprocess_fixed_population
from .metrics import evaluate_lts_predictions


@dataclass(frozen=True)
class GridRow:
    rank: int
    config_id: str
    min_seed_iou: float
    relink_max_gap: int
    relink_max_cost: float
    interpolate_single_frame: bool
    codabench_hota: float
    codabench_mota: float
    codabench_idf1: float
    hota: float
    deta: float
    assa: float
    loca: float
    mota: float
    idf1: float
    output_rows: int
    dropped_input_tracks: int
    relinked_tracklets: int
    interpolated_rows: int
    prediction_dir: str


def run_fixed_population_grid(
    prediction_path: Path,
    truth_dir: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    min_seed_ious: tuple[float, ...] = (0.3, 0.5, 0.7),
    relink_max_gaps: tuple[int, ...] = (0, 2, 5, 15),
    relink_max_costs: tuple[float, ...] = (1.0, 1.5, 2.0),
    interpolation_options: tuple[bool, ...] = (False, True),
    sequences: tuple[str, ...] = (),
) -> tuple[GridRow, ...]:
    """Evaluate a deterministic Cartesian grid and materialize the best predictions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    combinations = itertools.product(
        min_seed_ious,
        relink_max_gaps,
        relink_max_costs,
        interpolation_options,
    )
    for index, (min_iou, max_gap, max_cost, interpolate) in enumerate(combinations):
        config_id = (
            f"seed{min_iou:.3f}_gap{max_gap}_cost{max_cost:.3f}_"
            f"interp{int(interpolate)}"
        ).replace(".", "p")
        prediction_dir = output_dir / "configs" / config_id / "predictions"
        summary = postprocess_fixed_population(
            prediction_path,
            first_frame_label_dir,
            prediction_dir,
            min_seed_iou=min_iou,
            relink_max_gap=max_gap,
            relink_max_cost=max_cost,
            interpolate_single_frame=interpolate,
            sequences=sequences,
        )
        metrics = evaluate_lts_predictions(
            prediction_dir,
            truth_dir,
            sequences=sequences,
        )
        raw_rows.append(
            {
                "config_id": config_id,
                "min_seed_iou": min_iou,
                "relink_max_gap": max_gap,
                "relink_max_cost": max_cost,
                "interpolate_single_frame": interpolate,
                "codabench_hota": metrics.codabench_hota,
                "codabench_mota": metrics.codabench_mota,
                "codabench_idf1": metrics.codabench_idf1,
                "hota": metrics.hota,
                "deta": metrics.deta,
                "assa": metrics.assa,
                "loca": metrics.loca,
                "mota": metrics.mota,
                "idf1": metrics.idf1,
                "output_rows": summary.output_rows,
                "dropped_input_tracks": summary.dropped_input_tracks,
                "relinked_tracklets": summary.relinked_tracklets,
                "interpolated_rows": summary.interpolated_rows,
                "prediction_dir": str(prediction_dir),
                "grid_index": index,
            }
        )

    raw_rows.sort(
        key=lambda row: (
            -float(row["codabench_hota"]),
            -float(row["codabench_idf1"]),
            -float(row["codabench_mota"]),
            int(row["grid_index"]),
        )
    )
    rows = tuple(
        GridRow(
            rank=rank,
            config_id=str(row["config_id"]),
            min_seed_iou=float(row["min_seed_iou"]),
            relink_max_gap=int(row["relink_max_gap"]),
            relink_max_cost=float(row["relink_max_cost"]),
            interpolate_single_frame=bool(row["interpolate_single_frame"]),
            codabench_hota=float(row["codabench_hota"]),
            codabench_mota=float(row["codabench_mota"]),
            codabench_idf1=float(row["codabench_idf1"]),
            hota=float(row["hota"]),
            deta=float(row["deta"]),
            assa=float(row["assa"]),
            loca=float(row["loca"]),
            mota=float(row["mota"]),
            idf1=float(row["idf1"]),
            output_rows=int(row["output_rows"]),
            dropped_input_tracks=int(row["dropped_input_tracks"]),
            relinked_tracklets=int(row["relinked_tracklets"]),
            interpolated_rows=int(row["interpolated_rows"]),
            prediction_dir=str(row["prediction_dir"]),
        )
        for rank, row in enumerate(raw_rows, start=1)
    )
    if rows:
        best_source = Path(rows[0].prediction_dir)
        best_target = output_dir / "best_predictions"
        if best_target.exists():
            shutil.rmtree(best_target)
        shutil.copytree(best_source, best_target)
    _write_grid_outputs(rows, output_dir)
    return rows


def _write_grid_outputs(rows: tuple[GridRow, ...], output_dir: Path) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-fixed-population-grid-v1",
        "rows": [asdict(row) for row in rows],
        "best": asdict(rows[0]) if rows else None,
    }
    (output_dir / "grid_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = list(GridRow.__dataclass_fields__)
    with (output_dir / "grid_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    rows = run_fixed_population_grid(
        args.prediction_path,
        args.truth_dir,
        args.first_frame_label_dir,
        args.output_dir,
        min_seed_ious=tuple(args.min_seed_ious),
        relink_max_gaps=tuple(args.relink_max_gaps),
        relink_max_costs=tuple(args.relink_max_costs),
        interpolation_options=interpolation_options,
        sequences=tuple(args.sequences),
    )
    if rows:
        print(f"best_config={rows[0].config_id}")
        print(f"best_CODABENCH_HOTA={rows[0].codabench_hota:.6f}")
        print(f"best_CODABENCH_IDF1={rows[0].codabench_idf1:.6f}")
        print(f"best_CODABENCH_MOTA={rows[0].codabench_mota:.6f}")
        print(f"best_standard_HOTA={rows[0].hota:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
