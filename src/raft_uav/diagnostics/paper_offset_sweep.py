"""Fingerprint-scored RF/radar clock-offset sweep for paper parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_parity import parity_score_from_count_audit
from raft_uav.diagnostics.paper_strict import (
    PAPER_STRICT_NIS_GATE_PROBABILITY,
    PAPER_STRICT_RANGE_GATE_M,
    PaperStrictConfig,
    build_count_audit,
    build_paper_parity_report,
    build_paper_strict_table,
    load_paper_strict_inputs,
    run_paper_strict_fusion,
)
from raft_uav.io.aerpaw import DEFAULT_RADAR_CLOCK_OFFSET_S, DEFAULT_RF_CLOCK_OFFSET_S


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-paper-offset-sweep",
        description="sweep RF/radar residual clock offsets using paper-count fingerprints",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper-offset-sweep"))
    parser.add_argument("--variant", choices=["auto", "original", "rerun"], default="auto")
    parser.add_argument("--rf-clock-offset-s", type=float, default=DEFAULT_RF_CLOCK_OFFSET_S)
    parser.add_argument("--radar-clock-offset-s", type=float, default=DEFAULT_RADAR_CLOCK_OFFSET_S)
    parser.add_argument(
        "--rf-residual-grid-s",
        default="-2.0,2.0,0.05",
        help="START,STOP,STEP residual RF offsets added to --rf-clock-offset-s",
    )
    parser.add_argument(
        "--radar-residual-grid-s",
        default="-1.0,1.0,0.02",
        help="START,STOP,STEP residual radar offsets added to --radar-clock-offset-s",
    )
    parser.add_argument("--range-gate-m", type=float, default=PAPER_STRICT_RANGE_GATE_M)
    parser.add_argument("--nis-gate-prob", type=float, default=PAPER_STRICT_NIS_GATE_PROBABILITY)
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--enu-origin", choices=["truth-first", "lla", "lw1"], default="lw1")
    parser.add_argument("--enu-origin-lla", default=None)
    parser.add_argument("--lw1-origin-lla", default=None)
    parser.add_argument("--origin-config", type=Path, default=None)
    parser.add_argument("--no-empirical-covariance", action="store_true")
    args = parser.parse_args(argv)

    result = run_offset_sweep(
        dataset_root=args.dataset_root,
        flight=args.flight,
        output_dir=args.output_dir,
        variant=args.variant,
        rf_clock_offset_s=args.rf_clock_offset_s,
        radar_clock_offset_s=args.radar_clock_offset_s,
        rf_residual_grid_s=_parse_grid(args.rf_residual_grid_s),
        radar_residual_grid_s=_parse_grid(args.radar_residual_grid_s),
        range_gate_m=args.range_gate_m,
        nis_gate_probability=args.nis_gate_prob,
        truth_time_gate_s=args.truth_time_gate_s,
        acceleration_std_mps2=args.acceleration_std,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
        empirical_covariance=not args.no_empirical_covariance,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"summary_csv={result['summary_csv']}")
    print(f"best_json={result['best_json']}")
    return 0


def run_offset_sweep(
    *,
    dataset_root: Path,
    flight: str,
    output_dir: Path,
    variant: str,
    rf_clock_offset_s: float,
    radar_clock_offset_s: float,
    rf_residual_grid_s: np.ndarray,
    radar_residual_grid_s: np.ndarray,
    range_gate_m: float,
    nis_gate_probability: float,
    truth_time_gate_s: float,
    acceleration_std_mps2: float,
    enu_origin: str,
    enu_origin_lla: str | None,
    lw1_origin_lla: str | None,
    origin_config: Path | None,
    empirical_covariance: bool,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = PaperStrictConfig(
        range_gate_m=range_gate_m,
        nis_gate_probability=nis_gate_probability,
        truth_time_gate_s=truth_time_gate_s,
        acceleration_std_mps2=acceleration_std_mps2,
        empirical_covariance=empirical_covariance,
        bootstrap_source="radar",
    )

    rows: list[dict[str, Any]] = []
    for rf_residual_s in rf_residual_grid_s:
        for radar_residual_s in radar_residual_grid_s:
            row = _evaluate_offset_pair(
                dataset_root=dataset_root,
                flight=flight,
                variant=variant,
                rf_clock_offset_s=float(rf_clock_offset_s + rf_residual_s),
                radar_clock_offset_s=float(radar_clock_offset_s + radar_residual_s),
                rf_residual_s=float(rf_residual_s),
                radar_residual_s=float(radar_residual_s),
                config=config,
                enu_origin=enu_origin,
                enu_origin_lla=enu_origin_lla,
                lw1_origin_lla=lw1_origin_lla,
                origin_config=origin_config,
            )
            rows.append(row)

    summary = pd.DataFrame.from_records(rows).sort_values(
        ["paper_parity_score", "count_abs_delta_total", "kf_all_steps_mean_abs_delta_m"],
        ascending=[True, True, True],
    )
    summary_csv = output / "paper_offset_sweep_summary.csv"
    best_json = output / "paper_offset_sweep_best.json"
    summary.to_csv(summary_csv, index=False)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    payload = {
        "output_dir": str(output),
        "summary_csv": str(summary_csv),
        "flight": flight,
        "variant": variant,
        "base_rf_clock_offset_s": float(rf_clock_offset_s),
        "base_radar_clock_offset_s": float(radar_clock_offset_s),
        "best": best,
    }
    best_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "best_json": str(best_json)}


def _evaluate_offset_pair(
    *,
    dataset_root: Path,
    flight: str,
    variant: str,
    rf_clock_offset_s: float,
    radar_clock_offset_s: float,
    rf_residual_s: float,
    radar_residual_s: float,
    config: PaperStrictConfig,
    enu_origin: str,
    enu_origin_lla: str | None,
    lw1_origin_lla: str | None,
    origin_config: Path | None,
) -> dict[str, Any]:
    inputs = load_paper_strict_inputs(
        dataset_root=Path(dataset_root),
        flight_name=flight,
        enu_origin=enu_origin,
        enu_origin_lla=enu_origin_lla,
        lw1_origin_lla=lw1_origin_lla,
        origin_config=origin_config,
        rf_default_std_m=config.rf_default_std_m,
        variant=variant,
        rf_clock_offset_s=rf_clock_offset_s,
        radar_clock_offset_s=radar_clock_offset_s,
    )
    fusion = run_paper_strict_fusion(inputs=inputs, config=config)
    table = build_paper_strict_table(inputs=inputs, fusion=fusion, config=config)
    count_audit = build_count_audit(table)
    parity = build_paper_parity_report(table, count_audit)
    count_delta_total = int(count_audit["delta"].dropna().abs().sum()) if not count_audit.empty else 0
    kf_row = parity.loc[parity["method"] == "KF all steps"]
    kf_mean_delta = _optional_float(kf_row["mean_delta_m"].iloc[0]) if not kf_row.empty else None
    score = parity_score_from_count_audit(
        count_audit.to_dict(orient="records"),
        error_delta_m=kf_mean_delta,
    )
    row: dict[str, Any] = {
        "flight": inputs.flight_name,
        "rf_residual_offset_s": float(rf_residual_s),
        "radar_residual_offset_s": float(radar_residual_s),
        "rf_clock_offset_s": float(rf_clock_offset_s),
        "radar_clock_offset_s": float(radar_clock_offset_s),
        "paper_parity_score": float(score),
        "count_abs_delta_total": count_delta_total,
        "kf_all_steps_mean_delta_m": kf_mean_delta,
        "kf_all_steps_mean_abs_delta_m": None if kf_mean_delta is None else abs(float(kf_mean_delta)),
    }
    for _, audit_row in count_audit.iterrows():
        key = str(audit_row["method"]).lower().replace(" ", "_")
        row[f"{key}_observed_count"] = _optional_int(audit_row.get("observed_count"))
        row[f"{key}_count_delta"] = _optional_int(audit_row.get("delta"))
    return row


def _parse_grid(spec: str) -> np.ndarray:
    parts = [float(part.strip()) for part in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError("grid must have the form START,STOP,STEP")
    start, stop, step = parts
    if step <= 0.0:
        raise ValueError("grid STEP must be positive")
    count = int(np.floor((stop - start) / step + 0.5)) + 1
    if count < 1:
        raise ValueError("grid STOP must be >= START")
    return start + step * np.arange(count, dtype=float)


def _optional_float(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _optional_int(value: Any) -> int | None:
    scalar = _optional_float(value)
    return None if scalar is None else int(round(scalar))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
