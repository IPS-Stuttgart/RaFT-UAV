"""Stage-wise first-mismatch diagnostics for strict paper reproduction.

The paper-strict command reports count deltas after a full run. This helper is
more surgical: it orders the reproduction pipeline into human-readable stages
and marks the first stage where the observed Table-II fingerprint diverges.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_strict import (
    PAPER_REFERENCE_COUNTS,
    PAPER_STRICT_NIS_GATE_PROBABILITY,
    PAPER_STRICT_RANGE_GATE_M,
    PaperStrictConfig,
    build_paper_strict_table,
    load_paper_strict_inputs,
    paper_strict_range_gated_radar_candidates,
    paper_strict_stage_counts,
    run_paper_strict_fusion,
)
from raft_uav.io.aerpaw import discover_flights, select_flight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-paper-first-mismatch",
        description="pinpoint the first count-fingerprint mismatch in paper-strict reproduction",
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper-first-mismatch"))
    parser.add_argument("--variant", choices=["auto", "original", "rerun"], default="auto")
    parser.add_argument("--range-gate-m", type=float, default=PAPER_STRICT_RANGE_GATE_M)
    parser.add_argument("--nis-gate-prob", type=float, default=PAPER_STRICT_NIS_GATE_PROBABILITY)
    parser.add_argument("--rf-nis-gate-prob", type=float, default=None)
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=None)
    parser.add_argument("--no-empirical-covariance", action="store_true")
    parser.add_argument("--allow-missing-radar-range", action="store_true")
    parser.add_argument("--bootstrap-source", choices=["radar", "first-event"], default="radar")
    parser.add_argument("--enu-origin", choices=["truth-first", "lla", "lw1"], default="lw1")
    parser.add_argument("--enu-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lla")
    parser.add_argument("--lw1-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lw1")
    parser.add_argument("--origin-config", type=Path, default=None)
    parser.add_argument("--rf-default-std-m", type=float, default=75.0)
    parser.add_argument("--radar-default-xy-std-m", type=float, default=25.0)
    parser.add_argument("--radar-default-z-std-m", type=float, default=35.0)
    args = parser.parse_args(argv)

    config = PaperStrictConfig(
        range_gate_m=args.range_gate_m,
        nis_gate_probability=args.nis_gate_prob,
        rf_nis_gate_probability=args.rf_nis_gate_prob,
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
    result = run_paper_first_mismatch(
        dataset_root=args.dataset_root,
        flights=args.flight,
        output_dir=args.output_dir,
        config=config,
        variant=args.variant,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
    )
    print(f"summary_csv={result['summary_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def run_paper_first_mismatch(
    *,
    dataset_root: Path,
    flights: Iterable[str] | None,
    output_dir: Path = Path("outputs/paper-first-mismatch"),
    config: PaperStrictConfig = PaperStrictConfig(),
    variant: str = "auto",
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
) -> dict[str, Any]:
    """Run strict fusion and write stage-wise mismatch artifacts."""

    dataset_root = Path(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_flights = _resolve_flights(dataset_root, flights, variant=variant)

    summary_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for flight_name in selected_flights:
        flight = select_flight(dataset_root, flight_name, variant=variant)
        inputs = load_paper_strict_inputs(
            dataset_root=dataset_root,
            flight_name=flight.name,
            enu_origin=enu_origin,
            enu_origin_lla=enu_origin_lla,
            lw1_origin_lla=lw1_origin_lla,
            rf_default_std_m=config.rf_default_std_m,
            origin_config=origin_config,
            variant=variant,
        )
        fusion = run_paper_strict_fusion(inputs=inputs, config=config)
        table = build_paper_strict_table(inputs=inputs, fusion=fusion, config=config)
        report = build_first_mismatch_report(inputs=inputs, fusion=fusion, config=config)
        first = first_mismatch_row(report)

        flight_dir = output / inputs.flight_name
        flight_dir.mkdir(parents=True, exist_ok=True)
        report_csv = flight_dir / "paper_first_mismatch.csv"
        table_csv = flight_dir / "paper_strict_table.csv"
        manifest_json = flight_dir / "paper_first_mismatch_manifest.json"
        report.to_csv(report_csv, index=False)
        table.to_csv(table_csv, index=False)
        row = {
            "flight": inputs.flight_name,
            "first_mismatch_stage": None if first is None else first["stage"],
            "first_mismatch_observed": None if first is None else first["observed"],
            "first_mismatch_expected": None if first is None else first["expected"],
            "first_mismatch_delta": None if first is None else first["delta"],
            "all_reference_counts_match": first is None,
            "report_csv": str(report_csv),
            "table_csv": str(table_csv),
        }
        manifest = {
            **row,
            "config": asdict(config),
            "variant": variant,
            "file_manifest": inputs.file_manifest,
            "stage_rows": report.to_dict(orient="records"),
        }
        manifest_json.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        row["manifest_json"] = str(manifest_json)
        summary_rows.append(row)
        manifests.append(manifest)

    summary = pd.DataFrame.from_records(summary_rows)
    summary_csv = output / "paper_first_mismatch_summary.csv"
    summary_json = output / "paper_first_mismatch_summary.json"
    summary.to_csv(summary_csv, index=False)
    payload = {
        "summary_csv": str(summary_csv),
        "variant": variant,
        "reference_counts": PAPER_REFERENCE_COUNTS,
        "flights": manifests,
    }
    summary_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return {**payload, "summary_json": str(summary_json)}


def build_first_mismatch_report(
    *,
    inputs: Any,
    fusion: Any,
    config: PaperStrictConfig,
) -> pd.DataFrame:
    """Return ordered reproduction stages with observed/expected counts."""

    counts = paper_strict_stage_counts(inputs=inputs, fusion=fusion)
    range_pool = paper_strict_range_gated_radar_candidates(
        inputs.radar,
        range_gate_m=config.range_gate_m,
        catprob_threshold=config.radar_catprob_threshold,
        require_range_m=config.require_radar_range_m,
    )
    stages: list[dict[str, Any]] = [
        _stage(
            "stage_00_file_selection",
            observed=_file_selection_detail(inputs),
            expected=None,
            suspected_cause="Wrong Opt1 original/rerun, RF, radar, or truth file selection.",
        ),
        _stage(
            "stage_01_raw_rf_count",
            observed=counts["rf_raw_rows"],
            expected=PAPER_REFERENCE_COUNTS["RF raw"],
            suspected_cause="RF clock offset, truth-window clipping, or wrong RF CSV variant.",
        ),
        _stage(
            "stage_02_raw_radar_count",
            observed=counts["radar_all_track_rows"],
            expected=PAPER_REFERENCE_COUNTS["Radar raw"],
            suspected_cause=(
                "Radar JSON extraction, truth-window clipping, or wrong radar JSON variant."
            ),
        ),
        _stage(
            "stage_03_time_normalization",
            observed=_time_span_detail(inputs),
            expected=None,
            suspected_cause=(
                "Sensor clock offset sign, timezone, or truth origin timestamp mismatch."
            ),
        ),
        _stage(
            "stage_04_enu_origin",
            observed=_origin_detail(inputs),
            expected=None,
            suspected_cause=(
                "LW1 origin, truth-first fallback, altitude datum, "
                "or latitude/longitude ordering."
            ),
        ),
        _stage(
            "stage_05_radar_range_gate",
            observed=len(range_pool),
            expected=None,
            suspected_cause=(
                "800 m Fortem range gate applied to wrong range field or ENU norm fallback."
            ),
        ),
        _stage(
            "stage_06_largest_track_selection",
            observed=counts["radar_largest_continuous_track_rows"],
            expected=None,
            suspected_cause=(
                "Continuous-segment splitting, track_id parsing, or catProb prefilter mismatch."
            ),
            detail=_selected_track_detail(fusion.preselected_radar),
        ),
        _stage(
            "stage_07_rf_after_nis",
            observed=counts["rf_after_nis_rows"],
            expected=PAPER_REFERENCE_COUNTS["RF after NIS"],
            suspected_cause=(
                "RF covariance, RF NIS gate probability, RF timestamp alignment, "
                "or RF residual bias."
            ),
        ),
        _stage(
            "stage_08_radar_after_nis",
            observed=counts["radar_after_nis_rows"],
            expected=PAPER_REFERENCE_COUNTS["Radar after NIS"],
            suspected_cause=(
                "Radar covariance, radar NIS gate probability, or selected radar segment mismatch."
            ),
        ),
        _stage(
            "stage_09_kf_all_steps",
            observed=counts["kf_all_steps_rows"],
            expected=PAPER_REFERENCE_COUNTS["KF all steps"],
            suspected_cause=(
                "Event schedule, radar coasting policy, bootstrap source, "
                "or duplicate timestamp handling."
            ),
        ),
        _stage(
            "stage_10_kf_updated",
            observed=counts["kf_updated_rows"],
            expected=PAPER_REFERENCE_COUNTS["KF updated"],
            suspected_cause="RF/radar NIS validation or missed-detection accounting mismatch.",
        ),
        _stage(
            "stage_11_kf_coasted",
            observed=counts["kf_coasted_rows"],
            expected=PAPER_REFERENCE_COUNTS["KF coasted"],
            suspected_cause=(
                "Missing selected radar frames, rejected updates counted as coasts, "
                "or range-gate support."
            ),
        ),
    ]
    report = pd.DataFrame.from_records(stages)
    mismatch_seen = False
    first_flags: list[bool] = []
    for _, row in report.iterrows():
        is_mismatch = bool(row.get("matches_reference") is False)
        first_flags.append(is_mismatch and not mismatch_seen)
        mismatch_seen = mismatch_seen or is_mismatch
    report["first_mismatch"] = first_flags
    return report


def first_mismatch_row(report: pd.DataFrame) -> dict[str, Any] | None:
    """Return the first mismatching stage row, or None if no reference-count mismatch."""

    rows = report.loc[report["first_mismatch"].fillna(False).astype(bool)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _stage(
    stage: str,
    *,
    observed: object,
    expected: object,
    suspected_cause: str,
    detail: object | None = None,
) -> dict[str, Any]:
    observed_number = _as_number(observed)
    expected_number = _as_number(expected)
    if expected_number is None or observed_number is None:
        matches = None
        delta = None
    else:
        delta = observed_number - expected_number
        matches = bool(delta == 0)
    return {
        "stage": stage,
        "observed": observed,
        "expected": expected,
        "delta": delta,
        "matches_reference": matches,
        "suspected_cause": suspected_cause,
        "detail": detail,
    }


def _file_selection_detail(inputs: Any) -> str:
    manifest = getattr(inputs, "file_manifest", {}) or {}
    entries = []
    for key in ("rf", "radar", "truth"):
        entry = manifest.get(key, {}) if isinstance(manifest, dict) else {}
        if not isinstance(entry, dict):
            continue
        entries.append(f"{key}:{entry.get('name')}:{entry.get('variant')}")
    return "; ".join(entries)


def _time_span_detail(inputs: Any) -> str:
    parts = []
    for name in ("truth", "rf", "radar"):
        frame = getattr(inputs, name)
        if frame is None or frame.empty or "time_s" not in frame.columns:
            parts.append(f"{name}:empty")
            continue
        times = pd.to_numeric(frame["time_s"], errors="coerce").dropna()
        parts.append(f"{name}:{float(times.min()):.3f}..{float(times.max()):.3f}s n={len(times)}")
    return "; ".join(parts)


def _origin_detail(inputs: Any) -> str:
    projector = getattr(inputs, "projector", None)
    mode = getattr(inputs, "enu_origin_mode", None)
    if projector is None:
        return f"mode={mode}; projector=None"
    return (
        f"mode={mode}; lat={projector.origin_latitude_deg:.9f}; "
        f"lon={projector.origin_longitude_deg:.9f}; alt={projector.origin_altitude_m:.3f}"
    )


def _selected_track_detail(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "selected_track=empty"
    track_ids = pd.to_numeric(
        frame.get("track_id", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    track_id = int(track_ids.iloc[0]) if len(track_ids) else None
    times = pd.to_numeric(frame["time_s"], errors="coerce").dropna()
    return (
        f"track_id={track_id}; rows={len(frame)}; "
        f"time={float(times.min()):.3f}..{float(times.max()):.3f}s"
    )


def _as_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _resolve_flights(
    dataset_root: Path,
    flights: Iterable[str] | None,
    *,
    variant: str,
) -> list[str]:
    requested = list(flights or [])
    if requested:
        return [select_flight(dataset_root, name, variant=variant).name for name in requested]
    return [flight.name for flight in discover_flights(dataset_root, variant=variant)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
