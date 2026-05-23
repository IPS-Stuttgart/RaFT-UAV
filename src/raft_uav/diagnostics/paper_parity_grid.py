"""Grid search paper-parity reproduction configurations.

This diagnostic is deliberately narrower than SOTA evaluation.  It sweeps the
configuration knobs most likely to explain a large gap to the paper baseline:
file variant, Fortem target-track/range-gate ordering, optional catProb cut,
bootstrap source, and small residual RF/radar clock offsets.  Each candidate is
scored against the published Table-II count/error fingerprint so tracker tuning
starts only after the input/metric protocol is reproducible.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_parity import parity_score_from_count_audit
from raft_uav.diagnostics.paper_strict import (
    PAPER_STRICT_NIS_GATE_PROBABILITY,
    PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS,
    PAPER_STRICT_RANGE_GATE_M,
    PaperStrictConfig,
    build_count_audit,
    build_paper_parity_report,
    build_paper_strict_table,
    load_paper_strict_inputs,
    run_paper_strict_fusion,
)
from raft_uav.io.aerpaw import (
    DEFAULT_RADAR_CLOCK_OFFSET_S,
    DEFAULT_RF_CLOCK_OFFSET_S,
    FLIGHT_FILE_VARIANTS,
)

PAPER_PARITY_GRID_BOOTSTRAP_SOURCES = ("radar", "first-event")
_DEFAULT_VARIANTS = ("original", "rerun")
_DEFAULT_CATPROB_THRESHOLDS = (None, 0.4, 0.5)
_FAILURE_SCORE = 1.0e18


@dataclass(frozen=True)
class PaperParityGridCandidate:
    """One strict-reproduction candidate configuration."""

    flight: str
    variant: str
    radar_track_selection_order: str
    bootstrap_source: str
    radar_catprob_threshold: float | None
    rf_residual_offset_s: float
    radar_residual_offset_s: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-paper-parity-grid",
        description=(
            "sweep paper-strict reproduction settings and rank them by the "
            "published Table-II count/error fingerprint"
        ),
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--flight",
        action="append",
        required=True,
        help="flight name or substring; repeat to evaluate multiple flights",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper-parity-grid"))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=FLIGHT_FILE_VARIANTS,
        default=list(_DEFAULT_VARIANTS),
        help="file variants to evaluate; use 'auto' only for exploratory diagnostics",
    )
    parser.add_argument(
        "--radar-track-selection-orders",
        nargs="+",
        choices=PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS,
        default=list(PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS),
    )
    parser.add_argument(
        "--bootstrap-sources",
        nargs="+",
        choices=PAPER_PARITY_GRID_BOOTSTRAP_SOURCES,
        default=list(PAPER_PARITY_GRID_BOOTSTRAP_SOURCES),
    )
    parser.add_argument(
        "--radar-catprob-thresholds",
        nargs="+",
        default=["none", "0.4", "0.5"],
        help="one or more thresholds; accepts 'none' and comma-separated values",
    )
    parser.add_argument(
        "--rf-residual-grid-s",
        default="0,0,1",
        help="START,STOP,STEP residual RF offsets added to --rf-clock-offset-s",
    )
    parser.add_argument(
        "--radar-residual-grid-s",
        default="0,0,1",
        help="START,STOP,STEP residual radar offsets added to --radar-clock-offset-s",
    )
    parser.add_argument("--rf-clock-offset-s", type=float, default=DEFAULT_RF_CLOCK_OFFSET_S)
    parser.add_argument("--radar-clock-offset-s", type=float, default=DEFAULT_RADAR_CLOCK_OFFSET_S)
    parser.add_argument("--range-gate-m", type=float, default=PAPER_STRICT_RANGE_GATE_M)
    parser.add_argument("--nis-gate-prob", type=float, default=PAPER_STRICT_NIS_GATE_PROBABILITY)
    parser.add_argument(
        "--rf-nis-gate-prob",
        type=float,
        default=None,
        help="optional RF-specific NIS gate probability; defaults to --nis-gate-prob",
    )
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument(
        "--no-empirical-covariance",
        action="store_true",
        help="use diagonal fallback RF/radar covariance instead of truth residual covariance",
    )
    parser.add_argument(
        "--allow-missing-radar-range",
        action="store_true",
        help="allow ENU-norm range fallback; not recommended for paper parity",
    )
    parser.add_argument("--enu-origin", choices=["truth-first", "lla", "lw1"], default="lw1")
    parser.add_argument("--enu-origin-lla", default=None)
    parser.add_argument("--lw1-origin-lla", default=None)
    parser.add_argument("--origin-config", type=Path, default=None)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="raise the first candidate error instead of recording it in the summary",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_paper_parity_grid(
        dataset_root=args.dataset_root,
        flights=args.flight,
        output_dir=args.output_dir,
        variants=args.variants,
        radar_track_selection_orders=args.radar_track_selection_orders,
        bootstrap_sources=args.bootstrap_sources,
        radar_catprob_thresholds=_parse_nullable_float_list(args.radar_catprob_thresholds),
        rf_residual_grid_s=_parse_grid(args.rf_residual_grid_s),
        radar_residual_grid_s=_parse_grid(args.radar_residual_grid_s),
        rf_clock_offset_s=args.rf_clock_offset_s,
        radar_clock_offset_s=args.radar_clock_offset_s,
        range_gate_m=args.range_gate_m,
        nis_gate_probability=args.nis_gate_prob,
        rf_nis_gate_probability=args.rf_nis_gate_prob,
        truth_time_gate_s=args.truth_time_gate_s,
        acceleration_std_mps2=args.acceleration_std,
        empirical_covariance=not args.no_empirical_covariance,
        require_radar_range_m=not args.allow_missing_radar_range,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
        fail_fast=args.fail_fast,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"summary_csv={result['summary_csv']}")
    print(f"best_json={result['best_json']}")
    return 0


def run_paper_parity_grid(
    *,
    dataset_root: Path,
    flights: Sequence[str],
    output_dir: Path,
    variants: Sequence[str] = _DEFAULT_VARIANTS,
    radar_track_selection_orders: Sequence[str] = PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS,
    bootstrap_sources: Sequence[str] = PAPER_PARITY_GRID_BOOTSTRAP_SOURCES,
    radar_catprob_thresholds: Sequence[float | None] = _DEFAULT_CATPROB_THRESHOLDS,
    rf_residual_grid_s: np.ndarray | Sequence[float] = (0.0,),
    radar_residual_grid_s: np.ndarray | Sequence[float] = (0.0,),
    rf_clock_offset_s: float = DEFAULT_RF_CLOCK_OFFSET_S,
    radar_clock_offset_s: float = DEFAULT_RADAR_CLOCK_OFFSET_S,
    range_gate_m: float = PAPER_STRICT_RANGE_GATE_M,
    nis_gate_probability: float = PAPER_STRICT_NIS_GATE_PROBABILITY,
    rf_nis_gate_probability: float | None = None,
    truth_time_gate_s: float = 2.0,
    acceleration_std_mps2: float = 4.0,
    empirical_covariance: bool = True,
    require_radar_range_m: bool = True,
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Evaluate and rank paper-strict reproduction candidates."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = build_candidate_grid(
        flights=flights,
        variants=variants,
        radar_track_selection_orders=radar_track_selection_orders,
        bootstrap_sources=bootstrap_sources,
        radar_catprob_thresholds=radar_catprob_thresholds,
        rf_residual_grid_s=rf_residual_grid_s,
        radar_residual_grid_s=radar_residual_grid_s,
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            rows.append(
                evaluate_candidate(
                    dataset_root=dataset_root,
                    candidate=candidate,
                    rf_clock_offset_s=rf_clock_offset_s,
                    radar_clock_offset_s=radar_clock_offset_s,
                    range_gate_m=range_gate_m,
                    nis_gate_probability=nis_gate_probability,
                    rf_nis_gate_probability=rf_nis_gate_probability,
                    truth_time_gate_s=truth_time_gate_s,
                    acceleration_std_mps2=acceleration_std_mps2,
                    empirical_covariance=empirical_covariance,
                    require_radar_range_m=require_radar_range_m,
                    enu_origin=enu_origin,
                    enu_origin_lla=enu_origin_lla,
                    lw1_origin_lla=lw1_origin_lla,
                    origin_config=origin_config,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            rows.append(_failed_candidate_row(candidate, exc))

    summary = rank_grid_summary(pd.DataFrame.from_records(rows))
    summary_csv = output / "paper_parity_grid_summary.csv"
    best_json = output / "paper_parity_grid_best.json"
    summary.to_csv(summary_csv, index=False)
    best_row = _best_row(summary)
    payload = {
        "output_dir": str(output),
        "summary_csv": str(summary_csv),
        "candidate_count": int(len(candidates)),
        "successful_candidate_count": int((~summary.get("failed", pd.Series(dtype=bool)).astype(bool)).sum())
        if not summary.empty
        else 0,
        "best": _jsonable_mapping(best_row),
        "recommended_strict_command": _recommended_strict_command(
            dataset_root=Path(dataset_root),
            output_dir=output / "best-strict-run",
            row=best_row,
        ),
    }
    best_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "best_json": str(best_json)}


def build_candidate_grid(
    *,
    flights: Sequence[str],
    variants: Sequence[str],
    radar_track_selection_orders: Sequence[str],
    bootstrap_sources: Sequence[str],
    radar_catprob_thresholds: Sequence[float | None],
    rf_residual_grid_s: np.ndarray | Sequence[float],
    radar_residual_grid_s: np.ndarray | Sequence[float],
) -> list[PaperParityGridCandidate]:
    """Return the Cartesian product of paper-parity grid candidates."""

    _validate_nonempty("flights", flights)
    _validate_nonempty("variants", variants)
    _validate_nonempty("radar_track_selection_orders", radar_track_selection_orders)
    _validate_nonempty("bootstrap_sources", bootstrap_sources)
    _validate_nonempty("radar_catprob_thresholds", radar_catprob_thresholds)
    rf_offsets = [float(value) for value in np.asarray(rf_residual_grid_s, dtype=float).reshape(-1)]
    radar_offsets = [float(value) for value in np.asarray(radar_residual_grid_s, dtype=float).reshape(-1)]
    _validate_nonempty("rf_residual_grid_s", rf_offsets)
    _validate_nonempty("radar_residual_grid_s", radar_offsets)

    candidates: list[PaperParityGridCandidate] = []
    for flight, variant, order, bootstrap, catprob, rf_residual, radar_residual in itertools.product(
        flights,
        variants,
        radar_track_selection_orders,
        bootstrap_sources,
        radar_catprob_thresholds,
        rf_offsets,
        radar_offsets,
    ):
        if variant not in FLIGHT_FILE_VARIANTS:
            raise ValueError(f"variant must be one of {FLIGHT_FILE_VARIANTS}: {variant!r}")
        if order not in PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS:
            raise ValueError(
                "radar_track_selection_order must be one of "
                f"{PAPER_STRICT_RADAR_TRACK_SELECTION_ORDERS}: {order!r}"
            )
        if bootstrap not in PAPER_PARITY_GRID_BOOTSTRAP_SOURCES:
            raise ValueError(
                f"bootstrap_source must be one of {PAPER_PARITY_GRID_BOOTSTRAP_SOURCES}: "
                f"{bootstrap!r}"
            )
        candidates.append(
            PaperParityGridCandidate(
                flight=str(flight),
                variant=str(variant),
                radar_track_selection_order=str(order),
                bootstrap_source=str(bootstrap),
                radar_catprob_threshold=None if catprob is None else float(catprob),
                rf_residual_offset_s=float(rf_residual),
                radar_residual_offset_s=float(radar_residual),
            )
        )
    return candidates


def evaluate_candidate(
    *,
    dataset_root: Path,
    candidate: PaperParityGridCandidate,
    rf_clock_offset_s: float,
    radar_clock_offset_s: float,
    range_gate_m: float,
    nis_gate_probability: float,
    rf_nis_gate_probability: float | None,
    truth_time_gate_s: float,
    acceleration_std_mps2: float,
    empirical_covariance: bool,
    require_radar_range_m: bool,
    enu_origin: str,
    enu_origin_lla: str | None,
    lw1_origin_lla: str | None,
    origin_config: Path | None,
) -> dict[str, Any]:
    """Run one strict candidate and return its flattened parity row."""

    config = PaperStrictConfig(
        range_gate_m=range_gate_m,
        nis_gate_probability=nis_gate_probability,
        rf_nis_gate_probability=rf_nis_gate_probability,
        truth_time_gate_s=truth_time_gate_s,
        acceleration_std_mps2=acceleration_std_mps2,
        radar_catprob_threshold=candidate.radar_catprob_threshold,
        radar_track_selection_order=candidate.radar_track_selection_order,
        empirical_covariance=empirical_covariance,
        require_radar_range_m=require_radar_range_m,
        bootstrap_source=candidate.bootstrap_source,
    )
    applied_rf_offset_s = float(rf_clock_offset_s + candidate.rf_residual_offset_s)
    applied_radar_offset_s = float(radar_clock_offset_s + candidate.radar_residual_offset_s)
    inputs = load_paper_strict_inputs(
        dataset_root=Path(dataset_root),
        flight_name=candidate.flight,
        enu_origin=enu_origin,
        enu_origin_lla=enu_origin_lla,
        lw1_origin_lla=lw1_origin_lla,
        origin_config=origin_config,
        rf_default_std_m=config.rf_default_std_m,
        variant=candidate.variant,
        rf_clock_offset_s=applied_rf_offset_s,
        radar_clock_offset_s=applied_radar_offset_s,
    )
    fusion = run_paper_strict_fusion(inputs=inputs, config=config)
    table = build_paper_strict_table(inputs=inputs, fusion=fusion, config=config)
    count_audit = build_count_audit(table)
    parity = build_paper_parity_report(table, count_audit)
    count_abs_delta_total = _count_abs_delta_total(count_audit)
    kf_mean_delta = _method_metric_delta(parity, "KF all steps", "mean_delta_m")
    score = parity_score_from_count_audit(
        count_audit.to_dict(orient="records"),
        error_delta_m=kf_mean_delta,
    )

    row: dict[str, Any] = {
        **_candidate_columns(candidate),
        "resolved_flight": inputs.flight_name,
        "failed": False,
        "error": "",
        "rf_clock_offset_s": applied_rf_offset_s,
        "radar_clock_offset_s": applied_radar_offset_s,
        "paper_parity_score": float(score),
        "count_abs_delta_total": int(count_abs_delta_total),
        "kf_all_steps_mean_delta_m": kf_mean_delta,
        "kf_all_steps_mean_abs_delta_m": None if kf_mean_delta is None else abs(float(kf_mean_delta)),
        "kf_all_steps_observed_mean_m": _method_metric(parity, "KF all steps", "observed_mean_m"),
        "kf_all_steps_reference_mean_m": _method_metric(parity, "KF all steps", "reference_mean_m"),
        "rf_raw_rows": int(inputs.raw_rf_rows),
        "radar_file_rows": int(inputs.raw_radar_rows),
        "range_gate_m": float(config.range_gate_m),
        "nis_gate_probability": float(config.nis_gate_probability),
        "rf_nis_gate_probability": (
            float(config.nis_gate_probability)
            if config.rf_nis_gate_probability is None
            else float(config.rf_nis_gate_probability)
        ),
        "empirical_covariance": bool(config.empirical_covariance),
        "require_radar_range_m": bool(config.require_radar_range_m),
    }
    row.update(_count_columns(count_audit))
    row.update(_parity_metric_columns(parity))
    return row


def rank_grid_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return grid rows sorted by paper-parity fitness."""

    if summary.empty:
        return summary
    ranked = summary.copy()
    ranked["failed"] = ranked["failed"].fillna(False).astype(bool)
    for column in ("paper_parity_score", "count_abs_delta_total", "kf_all_steps_mean_abs_delta_m"):
        if column not in ranked.columns:
            ranked[column] = _FAILURE_SCORE
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce").fillna(_FAILURE_SCORE)
    ranked = ranked.sort_values(
        [
            "failed",
            "paper_parity_score",
            "count_abs_delta_total",
            "kf_all_steps_mean_abs_delta_m",
            "variant",
            "radar_track_selection_order",
            "bootstrap_source",
        ],
        ascending=[True, True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def _parse_grid(spec: str) -> np.ndarray:
    parts = [float(part.strip()) for part in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError("grid must have the form START,STOP,STEP")
    start, stop, step = parts
    if step <= 0.0:
        raise ValueError("grid STEP must be positive")
    if stop < start:
        raise ValueError("grid STOP must be >= START")
    count = int(np.floor((stop - start) / step + 0.5)) + 1
    return start + step * np.arange(count, dtype=float)


def _parse_nullable_float_list(values: Sequence[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for raw_value in values:
        for token in str(raw_value).split(","):
            value = token.strip().lower()
            if value in {"", "none", "null", "nan"}:
                parsed.append(None)
            else:
                parsed.append(float(value))
    if not parsed:
        raise ValueError("at least one radar catProb threshold is required")
    return tuple(parsed)


def _candidate_columns(candidate: PaperParityGridCandidate) -> dict[str, Any]:
    return {
        "flight": candidate.flight,
        "variant": candidate.variant,
        "radar_track_selection_order": candidate.radar_track_selection_order,
        "bootstrap_source": candidate.bootstrap_source,
        "radar_catprob_threshold": candidate.radar_catprob_threshold,
        "rf_residual_offset_s": float(candidate.rf_residual_offset_s),
        "radar_residual_offset_s": float(candidate.radar_residual_offset_s),
    }


def _failed_candidate_row(candidate: PaperParityGridCandidate, exc: Exception) -> dict[str, Any]:
    return {
        **_candidate_columns(candidate),
        "resolved_flight": candidate.flight,
        "failed": True,
        "error": f"{type(exc).__name__}: {exc}",
        "paper_parity_score": _FAILURE_SCORE,
        "count_abs_delta_total": _FAILURE_SCORE,
        "kf_all_steps_mean_abs_delta_m": _FAILURE_SCORE,
    }


def _count_columns(count_audit: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for _, audit_row in count_audit.iterrows():
        key = _slug(str(audit_row["method"]))
        row[f"{key}_observed_count"] = _optional_int(audit_row.get("observed_count"))
        row[f"{key}_reference_count"] = _optional_int(audit_row.get("reference_count"))
        row[f"{key}_count_delta"] = _optional_int(audit_row.get("delta"))
        row[f"{key}_count_matches_reference"] = _optional_bool(
            audit_row.get("matches_reference")
        )
    return row


def _parity_metric_columns(parity: pd.DataFrame) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for _, parity_row in parity.iterrows():
        key = _slug(str(parity_row["method"]))
        for metric in ("mean", "std", "max"):
            observed = _optional_float(parity_row.get(f"observed_{metric}_m"))
            reference = _optional_float(parity_row.get(f"reference_{metric}_m"))
            delta = _optional_float(parity_row.get(f"{metric}_delta_m"))
            if observed is not None:
                columns[f"{key}_observed_{metric}_m"] = observed
            if reference is not None:
                columns[f"{key}_reference_{metric}_m"] = reference
            if delta is not None:
                columns[f"{key}_{metric}_delta_m"] = delta
    return columns


def _count_abs_delta_total(count_audit: pd.DataFrame) -> int:
    if count_audit.empty or "delta" not in count_audit.columns:
        return 0
    deltas = pd.to_numeric(count_audit["delta"], errors="coerce").dropna().to_numpy(dtype=float)
    return int(np.abs(deltas).sum())


def _method_metric(parity: pd.DataFrame, method: str, column: str) -> float | None:
    rows = parity.loc[parity["method"].astype(str) == method]
    if rows.empty or column not in rows.columns:
        return None
    return _optional_float(rows[column].iloc[0])


def _method_metric_delta(parity: pd.DataFrame, method: str, column: str) -> float | None:
    return _method_metric(parity, method, column)


def _best_row(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {}
    return dict(summary.iloc[0].to_dict())


def _recommended_strict_command(
    *,
    dataset_root: Path,
    output_dir: Path,
    row: dict[str, Any],
) -> list[str]:
    if not row or bool(row.get("failed", False)):
        return []
    command = [
        "raft-uav-paper-strict",
        str(dataset_root),
        "--flight",
        str(row.get("resolved_flight") or row.get("flight")),
        "--output-dir",
        str(output_dir),
        "--variant",
        str(row.get("variant")),
        "--radar-track-selection-order",
        str(row.get("radar_track_selection_order")),
        "--bootstrap-source",
        str(row.get("bootstrap_source")),
        "--rf-clock-offset-s",
        str(row.get("rf_clock_offset_s")),
        "--radar-clock-offset-s",
        str(row.get("radar_clock_offset_s")),
        "--count-mismatch-action",
        "fail",
    ]
    catprob = _optional_float(row.get("radar_catprob_threshold"))
    if catprob is not None:
        command.extend(["--radar-catprob-threshold", str(catprob)])
    return command


def _jsonable_mapping(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in row.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _validate_nonempty(name: str, values: Sequence[object]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")


def _slug(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", "_").split())


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _optional_int(value: Any) -> int | None:
    scalar = _optional_float(value)
    return None if scalar is None else int(round(scalar))


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return bool(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
