"""Count-fingerprint diagnostics for AERPAW paper-parity reproduction.

This command is intentionally stricter than a performance leaderboard.  It is
meant to answer one question before model tuning starts: are we feeding the
Kalman filter the same RF/radar/truth streams, gates, and evaluation timestamps
as the reference paper table?
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_strict import (
    COUNT_MISMATCH_ACTIONS,
    PAPER_REFERENCE_COUNTS,
    PAPER_STRICT_NIS_GATE_PROBABILITY,
    PAPER_STRICT_RANGE_GATE_M,
    PaperStrictConfig,
    _handle_count_mismatch,
    build_count_audit,
    build_paper_strict_table,
    load_paper_strict_inputs,
    paper_strict_stage_counts,
    radar_range_audit,
    run_paper_strict_fusion,
)
from raft_uav.io.aerpaw import discover_flights, select_flight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-paper-fingerprint",
        description="audit paper-reference count fingerprints before tuning trackers",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--flight",
        action="append",
        default=None,
        help=(
            "flight name or substring; repeat to process multiple flights; "
            "defaults to all discovered flights"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper-fingerprint"))
    parser.add_argument(
        "--variant",
        choices=["auto", "original", "rerun"],
        default="auto",
        help="RF/radar/truth file variant; auto preserves historical rerun preference",
    )
    parser.add_argument("--range-gate-m", type=float, default=PAPER_STRICT_RANGE_GATE_M)
    parser.add_argument("--nis-gate-prob", type=float, default=PAPER_STRICT_NIS_GATE_PROBABILITY)
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument(
        "--radar-catprob-threshold",
        type=float,
        default=None,
        help="optional hard UAV catProb cut; omitted by default for paper-count matching",
    )
    parser.add_argument(
        "--no-empirical-covariance",
        action="store_true",
        help="use fallback covariance instead of same-flight residual covariance",
    )
    parser.add_argument(
        "--allow-missing-radar-range",
        action="store_true",
        help=(
            "allow ENU-norm fallback for exploratory diagnostics; "
            "paper parity should not use this"
        ),
    )
    parser.add_argument(
        "--bootstrap-source",
        choices=["radar", "first-event"],
        default="radar",
    )
    parser.add_argument(
        "--enu-origin",
        choices=["truth-first", "lla", "lw1"],
        default="lw1",
    )
    parser.add_argument(
        "--enu-origin-lla",
        default=None,
        help="LAT,LON,ALT origin for --enu-origin lla",
    )
    parser.add_argument(
        "--origin-config",
        type=Path,
        default=None,
        help="optional JSON/TOML origin registry; also read from RAFT_UAV_ORIGINS_FILE",
    )
    parser.add_argument(
        "--count-mismatch-action",
        choices=COUNT_MISMATCH_ACTIONS,
        default="warn",
        help="how to handle strict Table-II reference-count mismatches",
    )
    parser.add_argument(
        "--lw1-origin-lla",
        default=None,
        help="LAT,LON,ALT origin for --enu-origin lw1",
    )
    parser.add_argument("--rf-default-std-m", type=float, default=75.0)
    parser.add_argument("--radar-default-xy-std-m", type=float, default=25.0)
    parser.add_argument("--radar-default-z-std-m", type=float, default=35.0)
    args = parser.parse_args(argv)

    config = PaperStrictConfig(
        range_gate_m=args.range_gate_m,
        nis_gate_probability=args.nis_gate_prob,
        truth_time_gate_s=args.truth_time_gate_s,
        acceleration_std_mps2=args.acceleration_std,
        radar_catprob_threshold=args.radar_catprob_threshold,
        empirical_covariance=not args.no_empirical_covariance,
        require_radar_range_m=not args.allow_missing_radar_range,
        bootstrap_source=args.bootstrap_source,
        rf_default_std_m=args.rf_default_std_m,
        radar_default_xy_std_m=args.radar_default_xy_std_m,
        radar_default_z_std_m=args.radar_default_z_std_m,
    )
    result = run_paper_fingerprint(
        dataset_root=args.dataset_root,
        flights=args.flight,
        output_dir=args.output_dir,
        config=config,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
        variant=args.variant,
        count_mismatch_action=args.count_mismatch_action,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"summary_csv={result['summary_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def run_paper_fingerprint(
    *,
    dataset_root: Path,
    flights: Iterable[str] | None,
    output_dir: Path = Path("outputs/paper-fingerprint"),
    config: PaperStrictConfig = PaperStrictConfig(),
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
    variant: str = "auto",
    count_mismatch_action: str = "warn",
) -> dict[str, Any]:
    """Run strict count-fingerprint diagnostics and write CSV/JSON artifacts."""

    dataset_root = Path(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_flights = _resolve_flights(dataset_root, flights, variant=variant)

    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for flight_name in selected_flights:
        flight = select_flight(dataset_root, flight_name, variant=variant)
        inputs = load_paper_strict_inputs(
            dataset_root=dataset_root,
            flight_name=flight.name,
            enu_origin=enu_origin,
            enu_origin_lla=enu_origin_lla,
            lw1_origin_lla=lw1_origin_lla,
            origin_config=origin_config,
            rf_default_std_m=config.rf_default_std_m,
            variant=variant,
        )
        fusion = run_paper_strict_fusion(inputs=inputs, config=config)
        table = build_paper_strict_table(inputs=inputs, fusion=fusion, config=config)
        count_audit = build_count_audit(table)
        _handle_count_mismatch(
            count_audit,
            flight_name=inputs.flight_name,
            action=count_mismatch_action,
        )
        row = _fingerprint_row(
            dataset_root=dataset_root,
            flight_name=flight.name,
            rf_csv=flight.rf_csv,
            radar_json=flight.radar_json,
            truth_txt=flight.truth_txt,
            inputs=inputs,
            fusion=fusion,
            table=table,
            count_audit=count_audit,
        )
        rows.append(row)

        flight_dir = output / flight.name
        flight_dir.mkdir(parents=True, exist_ok=True)
        table_csv = flight_dir / "paper_strict_table.csv"
        count_csv = flight_dir / "paper_count_audit.csv"
        table.to_csv(table_csv, index=False)
        count_audit.to_csv(count_csv, index=False)
        manifest = {
            **row,
            "table_csv": str(table_csv),
            "count_audit_csv": str(count_csv),
            "range_audit": radar_range_audit(inputs.radar),
            "file_manifest": inputs.file_manifest,
        }
        manifest_json = flight_dir / "fingerprint_manifest.json"
        manifest_json.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        manifests.append({**manifest, "manifest_json": str(manifest_json)})

    summary = pd.DataFrame.from_records(rows)
    summary_csv = output / "paper_fingerprint_summary.csv"
    summary_json = output / "paper_fingerprint_summary.json"
    summary.to_csv(summary_csv, index=False)
    payload = {
        "output_dir": str(output),
        "summary_csv": str(summary_csv),
        "reference_counts": PAPER_REFERENCE_COUNTS,
        "config": asdict(config),
        "variant": variant,
        "count_mismatch_action": count_mismatch_action,
        "flights": manifests,
    }
    summary_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return {**payload, "summary_json": str(summary_json)}


def _fingerprint_row(
    *,
    dataset_root: Path,
    flight_name: str,
    rf_csv: Path | None,
    radar_json: Path | None,
    truth_txt: Path | None,
    inputs,
    fusion,
    table: pd.DataFrame,
    count_audit: pd.DataFrame,
) -> dict[str, Any]:
    stage_counts = paper_strict_stage_counts(inputs=inputs, fusion=fusion)
    count_deltas = {
        f"count_delta_{_slug(str(row['method']))}": row.get("delta")
        for _, row in count_audit.iterrows()
    }
    abs_deltas = [
        abs(float(value))
        for value in count_deltas.values()
        if value is not None and not pd.isna(value)
    ]
    table_rows = {str(row["method"]): row for _, row in table.iterrows()}
    return {
        "flight": flight_name,
        "rf_csv": _relative_path(dataset_root, rf_csv),
        "radar_json": _relative_path(dataset_root, radar_json),
        "truth_txt": _relative_path(dataset_root, truth_txt),
        "truth_rows": int(len(inputs.truth)),
        "truth_time_s_min": float(inputs.truth["time_s"].min()) if len(inputs.truth) else np.nan,
        "truth_time_s_max": float(inputs.truth["time_s"].max()) if len(inputs.truth) else np.nan,
        "enu_origin_mode": inputs.enu_origin_mode,
        "origin_latitude_deg": None if inputs.projector is None else float(inputs.projector.origin_latitude_deg),
        "origin_longitude_deg": None if inputs.projector is None else float(inputs.projector.origin_longitude_deg),
        "origin_altitude_m": None if inputs.projector is None else float(inputs.projector.origin_altitude_m),
        "rf_file_variant": _manifest_variant(inputs.file_manifest, "rf"),
        "radar_file_variant": _manifest_variant(inputs.file_manifest, "radar"),
        "truth_file_variant": _manifest_variant(inputs.file_manifest, "truth"),
        **stage_counts,
        **count_deltas,
        "reference_count_abs_delta_sum": float(sum(abs_deltas)) if abs_deltas else np.nan,
        "reference_count_matches_all": bool(all(value == 0 for value in count_deltas.values())),
        "kf_all_steps_mean_m": _table_value(table_rows, "KF all steps", "paper_error_mean_m"),
        "kf_all_steps_std_m": _table_value(table_rows, "KF all steps", "paper_error_std_m"),
        "kf_all_steps_max_m": _table_value(table_rows, "KF all steps", "paper_error_max_m"),
    }


def _resolve_flights(dataset_root: Path, flights: Iterable[str] | None, *, variant: str = "auto") -> list[str]:
    requested = list(flights or [])
    if requested:
        return [select_flight(dataset_root, name, variant=variant).name for name in requested]
    return [flight.name for flight in discover_flights(dataset_root, variant=variant)]


def _manifest_variant(manifest: dict[str, Any], key: str) -> str | None:
    entry = manifest.get(key) if isinstance(manifest, dict) else None
    if not isinstance(entry, dict):
        return None
    value = entry.get("variant")
    return None if value is None else str(value)


def _relative_path(root: Path, value: Path | None) -> str | None:
    if value is None:
        return None
    try:
        return str(Path(value).relative_to(root))
    except ValueError:
        return str(value)


def _table_value(rows: dict[str, pd.Series], method: str, column: str) -> float | None:
    row = rows.get(method)
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def _slug(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("/", " ").split())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
