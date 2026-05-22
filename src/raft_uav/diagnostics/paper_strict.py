"""Strict paper-parity reproduction for the AERPAW RF + Fortem radar baseline.

This module is intentionally narrower than the general result-improvement
pipeline.  It mirrors the reference-table conventions used by the paper:
truth is interpolated to measurement/output timestamps, Fortem radar is
preselected by the 800 m range gate plus the largest continuous track, RF/radar
updates are validated with a 95% chi-square NIS gate, measurement covariances
can be estimated empirically from truth residuals, and the Kalman filter is
bootstrapped from the radar track instead of an arbitrary first RF row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tomllib
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
    TrackingUpdateDiagnostics,
    gate_threshold_from_probability,
)
from raft_uav.coordinates import LocalENUProjector
from raft_uav.evaluation.metrics import (
    empirical_position_covariance_at_times,
    position_errors_at_times_m,
    summarize_errors,
)
from raft_uav.io.aerpaw import (
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    projector_from_lla,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
)

PAPER_STRICT_NIS_GATE_PROBABILITY = 0.95
PAPER_STRICT_RANGE_GATE_M = 800.0
PAPER_STRICT_LW1_ORIGIN_LLA_ENV = "RAFT_UAV_LW1_ORIGIN_LLA"
PAPER_STRICT_ORIGINS_FILE_ENV = "RAFT_UAV_ORIGINS_FILE"
COUNT_MISMATCH_ACTIONS = ("ignore", "warn", "fail")
PAPER_REFERENCE_COUNTS = {
    "RF raw": 206,
    "RF after NIS": 125,
    "Radar raw": 3106,
    "Radar after NIS": 2403,
    "KF all steps": 2655,
    "KF updated": 2528,
    "KF coasted": 127,
}
PAPER_REFERENCE_ERROR_M = {
    "RF raw": {"mean": 471.8, "std": 885.2, "max": 4831.6},
    "RF after NIS": {"mean": 25.8, "std": 16.2, "max": 113.9},
    "Radar raw": {"mean": 26.2, "std": 25.5, "max": 195.6},
    "Radar after NIS": {"mean": 21.0, "std": 17.1, "max": 97.2},
    "KF all steps": {"mean": 21.9, "std": 17.9, "max": 109.1},
}


@dataclass(frozen=True)
class PaperStrictConfig:
    """Configuration for the strict reference-table reproduction path."""

    range_gate_m: float = PAPER_STRICT_RANGE_GATE_M
    nis_gate_probability: float = PAPER_STRICT_NIS_GATE_PROBABILITY
    rf_nis_gate_probability: float | None = None
    truth_time_gate_s: float = 2.0
    acceleration_std_mps2: float = 4.0
    radar_catprob_threshold: float | None = None
    empirical_covariance: bool = True
    require_radar_range_m: bool = True
    bootstrap_source: str = "radar"
    rf_default_std_m: float = 75.0
    radar_default_xy_std_m: float = 25.0
    radar_default_z_std_m: float = 35.0


@dataclass(frozen=True)
class PaperStrictInputs:
    """Normalized data frames used by the paper-strict reproduction."""

    flight_name: str
    truth: pd.DataFrame
    rf: pd.DataFrame
    radar: pd.DataFrame
    projector: LocalENUProjector | None
    truth_origin_time: pd.Timestamp
    enu_origin_mode: str


@dataclass(frozen=True)
class PaperStrictFusionResult:
    """Fusion outputs and selected measurement streams."""

    records: list[dict[str, object]]
    preselected_radar: pd.DataFrame
    range_gated_radar: pd.DataFrame
    accepted_radar: pd.DataFrame
    accepted_rf: pd.DataFrame
    rf_covariance: np.ndarray
    radar_covariance: np.ndarray
    radar_gate_threshold: float
    rf_gate_threshold: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-paper-strict",
        description="run strict paper-parity RF/radar/KF reproduction diagnostics",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--flight",
        action="append",
        default=None,
        help="flight name or substring; repeat to process multiple flights; defaults to all discovered flights",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper-strict"))
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
        "--radar-catprob-threshold",
        type=float,
        default=None,
        help="optional hard UAV catProb cut; omitted by default because the reference text does not require it",
    )
    parser.add_argument(
        "--no-empirical-covariance",
        action="store_true",
        help="use diagonal fallback RF/radar covariance instead of truth residual covariance",
    )
    parser.add_argument(
        "--allow-missing-radar-range",
        action="store_true",
        help="allow fallback range from ENU norm; not recommended for paper parity",
    )
    parser.add_argument(
        "--bootstrap-source",
        choices=["radar", "first-event"],
        default="radar",
        help="radar avoids initializing from unvalidated RF outliers",
    )
    parser.add_argument(
        "--count-mismatch-action",
        choices=COUNT_MISMATCH_ACTIONS,
        default="warn",
        help="how to handle strict Table-II reference-count mismatches",
    )
    parser.add_argument(
        "--enu-origin",
        choices=["truth-first", "lla", "lw1"],
        default="lw1",
        help="LW1 is the paper-parity default; pass truth-first only for diagnostics",
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
        "--lw1-origin-lla",
        default=None,
        help="LAT,LON,ALT origin for --enu-origin lw1; also accepted via RAFT_UAV_LW1_ORIGIN_LLA",
    )
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
    result = run_paper_strict_reproduction(
        dataset_root=args.dataset_root,
        flights=args.flight,
        output_dir=args.output_dir,
        config=config,
        count_mismatch_action=args.count_mismatch_action,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"summary_csv={result['summary_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def run_paper_strict_reproduction(
    *,
    dataset_root: Path,
    flights: Iterable[str] | None,
    output_dir: Path = Path("outputs/paper-strict"),
    config: PaperStrictConfig = PaperStrictConfig(),
    count_mismatch_action: str = "warn",
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
) -> dict[str, Any]:
    """Run strict paper-parity diagnostics and write per-flight artifacts."""

    _validate_config(config)
    _validate_count_mismatch_action(count_mismatch_action)
    selected_flights = _resolve_flights(Path(dataset_root), flights)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for flight_name in selected_flights:
        inputs = load_paper_strict_inputs(
            dataset_root=Path(dataset_root),
            flight_name=flight_name,
            enu_origin=enu_origin,
            enu_origin_lla=enu_origin_lla,
            lw1_origin_lla=lw1_origin_lla,
            origin_config=origin_config,
            rf_default_std_m=config.rf_default_std_m,
        )
        flight_dir = output / inputs.flight_name
        flight_dir.mkdir(parents=True, exist_ok=True)
        fusion = run_paper_strict_fusion(inputs=inputs, config=config)
        table = build_paper_strict_table(inputs=inputs, fusion=fusion, config=config)
        count_audit = build_count_audit(table)
        _handle_count_mismatch(
            count_audit,
            flight_name=inputs.flight_name,
            action=count_mismatch_action,
        )
        range_audit = radar_range_audit(inputs.radar)

        table_csv = flight_dir / "paper_strict_table.csv"
        count_csv = flight_dir / "paper_count_audit.csv"
        estimates_csv = flight_dir / "paper_strict_estimates.csv"
        selected_radar_csv = flight_dir / "selected_radar.csv"
        accepted_rf_csv = flight_dir / "accepted_rf.csv"
        covariance_json = flight_dir / "empirical_covariances.json"
        manifest_json = flight_dir / "manifest.json"

        table.to_csv(table_csv, index=False)
        count_audit.to_csv(count_csv, index=False)
        _records_to_estimate_frame(fusion.records).to_csv(estimates_csv, index=False)
        fusion.accepted_radar.to_csv(selected_radar_csv, index=False)
        fusion.accepted_rf.to_csv(accepted_rf_csv, index=False)
        covariance_payload = {
            "empirical_covariance": bool(config.empirical_covariance),
            "rf_covariance": fusion.rf_covariance.tolist(),
            "radar_covariance": fusion.radar_covariance.tolist(),
            "rf_gate_threshold": float(fusion.rf_gate_threshold),
            "radar_gate_threshold": float(fusion.radar_gate_threshold),
        }
        covariance_json.write_text(json.dumps(covariance_payload, indent=2), encoding="utf-8")
        manifest = {
            "flight": inputs.flight_name,
            "table_csv": str(table_csv),
            "count_audit_csv": str(count_csv),
            "estimates_csv": str(estimates_csv),
            "selected_radar_csv": str(selected_radar_csv),
            "accepted_rf_csv": str(accepted_rf_csv),
            "covariance_json": str(covariance_json),
            "enu_origin_mode": inputs.enu_origin_mode,
            "truth_origin_time": str(inputs.truth_origin_time),
            "range_audit": range_audit,
            "stage_counts": paper_strict_stage_counts(inputs=inputs, fusion=fusion),
            "count_mismatch_action": count_mismatch_action,
            "config": _jsonable_config(config),
        }
        manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append({**manifest, "manifest_json": str(manifest_json)})
        all_rows.extend(table.to_dict(orient="records"))

    summary = pd.DataFrame.from_records(all_rows)
    summary_csv = output / "paper_strict_summary.csv"
    summary_json = output / "paper_strict_summary.json"
    summary.to_csv(summary_csv, index=False)
    payload = {
        "output_dir": str(output),
        "summary_csv": str(summary_csv),
        "flights": manifests,
        "reference_counts": PAPER_REFERENCE_COUNTS,
        "reference_errors_m": PAPER_REFERENCE_ERROR_M,
        "count_mismatch_action": count_mismatch_action,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "summary_json": str(summary_json)}


def load_paper_strict_inputs(
    *,
    dataset_root: Path,
    flight_name: str,
    enu_origin: str,
    enu_origin_lla: str | None,
    lw1_origin_lla: str | None,
    rf_default_std_m: float,
    origin_config: Path | None = None,
) -> PaperStrictInputs:
    """Load one flight and normalize all streams into a shared ENU frame."""

    flight = select_flight(Path(dataset_root), flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    if flight.rf_csv is None:
        raise FileNotFoundError(f"{flight.name} has no RF CSV file")
    if flight.radar_json is None:
        raise FileNotFoundError(f"{flight.name} has no radar JSON file")

    projector = _projector_for_origin(
        enu_origin=enu_origin,
        enu_origin_lla=enu_origin_lla,
        lw1_origin_lla=lw1_origin_lla,
        origin_config=origin_config,
    )
    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw, projector=projector)
    truth = truth.sort_values("time_s").reset_index(drop=True)
    rf = _inside_truth_window(
        normalize_rf(
            read_rf_csv(flight.rf_csv),
            projector,
            truth_origin_time,
            default_std_m=rf_default_std_m,
        ),
        truth,
    )
    radar = _inside_truth_window(
        normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
        truth,
    )
    return PaperStrictInputs(
        flight_name=flight.name,
        truth=truth,
        rf=rf.reset_index(drop=True),
        radar=radar.reset_index(drop=True),
        projector=projector,
        truth_origin_time=truth_origin_time,
        enu_origin_mode=enu_origin,
    )


def run_paper_strict_fusion(
    *,
    inputs: PaperStrictInputs,
    config: PaperStrictConfig = PaperStrictConfig(),
) -> PaperStrictFusionResult:
    """Run the strict radar-bootstrap CV Kalman fusion path."""

    _validate_config(config)
    radar = inputs.radar
    if config.require_radar_range_m:
        require_fortem_range_m(radar)
    range_gated_radar = paper_strict_range_gated_radar_candidates(
        radar,
        range_gate_m=config.range_gate_m,
        catprob_threshold=config.radar_catprob_threshold,
        require_range_m=config.require_radar_range_m,
    )
    preselected_radar = select_paper_strict_radar_track(
        radar,
        range_gate_m=config.range_gate_m,
        catprob_threshold=config.radar_catprob_threshold,
        require_range_m=config.require_radar_range_m,
    )
    rf_covariance, radar_covariance = measurement_covariances(
        inputs=inputs,
        preselected_radar=preselected_radar,
        config=config,
    )
    rf_measurements = rf_measurements_with_covariance(inputs.rf, rf_covariance)
    events = _events(rf_measurements=rf_measurements, radar=radar)
    if not events:
        return PaperStrictFusionResult(
            records=[],
            preselected_radar=preselected_radar,
            range_gated_radar=range_gated_radar,
            accepted_radar=preselected_radar.iloc[0:0].copy(),
            accepted_rf=inputs.rf.iloc[0:0].copy(),
            rf_covariance=rf_covariance,
            radar_covariance=radar_covariance,
            radar_gate_threshold=float("nan"),
            rf_gate_threshold=float("nan"),
        )

    radar_by_key = {_radar_row_key(row): row for _, row in preselected_radar.iterrows()}
    bootstrap = _bootstrap_event(
        events,
        radar_by_key=radar_by_key,
        radar_covariance=radar_covariance,
        bootstrap_source=config.bootstrap_source,
    )
    if bootstrap is None:
        return PaperStrictFusionResult(
            records=[],
            preselected_radar=preselected_radar,
            range_gated_radar=range_gated_radar,
            accepted_radar=preselected_radar.iloc[0:0].copy(),
            accepted_rf=inputs.rf.iloc[0:0].copy(),
            rf_covariance=rf_covariance,
            radar_covariance=radar_covariance,
            radar_gate_threshold=float("nan"),
            rf_gate_threshold=float("nan"),
        )
    start_index, initial_measurement, initial_row = bootstrap
    radar_gate_threshold = gate_threshold_from_probability(config.nis_gate_probability, 3)
    rf_gate_probability = (
        config.nis_gate_probability
        if config.rf_nis_gate_probability is None
        else config.rf_nis_gate_probability
    )
    rf_gate_threshold = gate_threshold_from_probability(rf_gate_probability, 2)
    assert radar_gate_threshold is not None
    assert rf_gate_threshold is not None

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=config.acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    accepted_radar_rows: list[pd.Series] = []
    accepted_rf_rows: list[pd.Series] = []

    initial_gate = radar_gate_threshold if initial_measurement.vector.size == 3 else rf_gate_threshold
    initial_diagnostics = tracker.update(initial_measurement, gate_threshold=initial_gate)
    if initial_diagnostics.accepted:
        if initial_measurement.source == "radar" and initial_row is not None:
            accepted_radar_rows.append(initial_row.copy())
        elif initial_measurement.source == "rf":
            row = _rf_row_at_time(inputs.rf, initial_measurement.time_s)
            if row is not None:
                accepted_rf_rows.append(row)
    records.append(
        _tracking_record(
            initial_measurement,
            tracker,
            initial_diagnostics,
            association_mode="paper-strict",
            selected_row=initial_row,
        )
    )

    for event in events[start_index + 1 :]:
        time_s = float(event["time_s"])
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = tracker.update(measurement, gate_threshold=rf_gate_threshold)
            if diagnostics.accepted:
                row = _rf_row_at_time(inputs.rf, measurement.time_s)
                if row is not None:
                    accepted_rf_rows.append(row)
            records.append(
                _tracking_record(
                    measurement,
                    tracker,
                    diagnostics,
                    association_mode="paper-strict",
                )
            )
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        tracker.predict_to(time_s)
        selected = radar_by_key.get(_radar_event_key(event))
        if selected is None:
            records.append(
                _coast_record(
                    time_s=time_s,
                    tracker=tracker,
                    association_mode="paper-strict",
                    gate_threshold=radar_gate_threshold,
                    reason="no_preselected_radar",
                )
            )
            continue
        measurement = radar_measurement_from_row(selected, radar_covariance)
        diagnostics = tracker.update(measurement, gate_threshold=radar_gate_threshold)
        if diagnostics.accepted:
            accepted_radar_rows.append(selected.copy())
        records.append(
            _tracking_record(
                measurement,
                tracker,
                diagnostics,
                association_mode="paper-strict",
                selected_row=selected,
            )
        )

    accepted_radar = _selected_rows_frame(preselected_radar, accepted_radar_rows)
    accepted_rf = pd.DataFrame(accepted_rf_rows) if accepted_rf_rows else inputs.rf.iloc[0:0].copy()
    if not accepted_rf.empty:
        accepted_rf = accepted_rf.sort_values("time_s").reset_index(drop=True)
    return PaperStrictFusionResult(
        records=records,
        preselected_radar=preselected_radar,
        range_gated_radar=range_gated_radar,
        accepted_radar=accepted_radar,
        accepted_rf=accepted_rf,
        rf_covariance=rf_covariance,
        radar_covariance=radar_covariance,
        radar_gate_threshold=float(radar_gate_threshold),
        rf_gate_threshold=float(rf_gate_threshold),
    )


def build_paper_strict_table(
    *,
    inputs: PaperStrictInputs,
    fusion: PaperStrictFusionResult,
    config: PaperStrictConfig,
) -> pd.DataFrame:
    """Return the paper-style metric rows for one strict reproduction run."""

    records = fusion.records
    estimates = _records_to_estimate_frame(records)
    accepted_updates = estimates.loc[estimates["accepted"].fillna(False).astype(bool)].copy()
    coasts = estimates.loc[~estimates["accepted"].fillna(False).astype(bool)].copy()

    rows = [
        _metric_row(
            method="RF raw",
            source="rf",
            frame=inputs.rf,
            truth=inputs.truth,
            dimensions=2,
            candidate_count=len(inputs.rf),
            selected_count=len(inputs.rf),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="RF after NIS",
            source="rf",
            frame=fusion.accepted_rf,
            truth=inputs.truth,
            dimensions=2,
            candidate_count=len(inputs.rf),
            selected_count=len(fusion.accepted_rf),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="Radar all Fortem track rows",
            source="radar",
            frame=inputs.radar,
            truth=inputs.truth,
            dimensions=3,
            candidate_count=len(inputs.radar),
            selected_count=len(inputs.radar),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="Radar after 800 m range gate",
            source="radar",
            frame=fusion.range_gated_radar,
            truth=inputs.truth,
            dimensions=3,
            candidate_count=len(inputs.radar),
            selected_count=len(fusion.range_gated_radar),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="Radar raw",
            source="radar",
            frame=fusion.preselected_radar,
            truth=inputs.truth,
            dimensions=3,
            candidate_count=len(inputs.radar),
            selected_count=len(fusion.preselected_radar),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="Radar after NIS",
            source="radar",
            frame=fusion.accepted_radar,
            truth=inputs.truth,
            dimensions=3,
            candidate_count=len(inputs.radar),
            selected_count=len(fusion.accepted_radar),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="KF all steps",
            source="fusion",
            frame=estimates,
            truth=inputs.truth,
            dimensions=3,
            candidate_count=len(estimates),
            selected_count=len(estimates),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="KF updated",
            source="fusion",
            frame=accepted_updates,
            truth=inputs.truth,
            dimensions=3,
            candidate_count=len(estimates),
            selected_count=len(accepted_updates),
            max_time_delta_s=config.truth_time_gate_s,
        ),
        _metric_row(
            method="KF coasted",
            source="fusion",
            frame=coasts,
            truth=inputs.truth,
            dimensions=3,
            candidate_count=len(estimates),
            selected_count=len(coasts),
            max_time_delta_s=config.truth_time_gate_s,
        ),
    ]
    table = pd.DataFrame.from_records(rows)
    table.insert(0, "flight", inputs.flight_name)
    table["paper_strict_nis_gate_probability"] = float(config.nis_gate_probability)
    table["paper_strict_range_gate_m"] = float(config.range_gate_m)
    table["paper_strict_catprob_threshold"] = (
        np.nan if config.radar_catprob_threshold is None else float(config.radar_catprob_threshold)
    )
    table["paper_strict_empirical_covariance"] = bool(config.empirical_covariance)
    table["paper_strict_bootstrap_source"] = config.bootstrap_source
    table["paper_strict_rf_nis_gate_probability"] = (
        float(config.nis_gate_probability)
        if config.rf_nis_gate_probability is None
        else float(config.rf_nis_gate_probability)
    )
    table["enu_origin_mode"] = inputs.enu_origin_mode
    return table


def build_count_audit(table: pd.DataFrame) -> pd.DataFrame:
    """Compare produced strict-count rows with the published count targets."""

    rows: list[dict[str, Any]] = []
    by_method = {str(row["method"]): row for _, row in table.iterrows()}
    for method, target in PAPER_REFERENCE_COUNTS.items():
        observed = by_method.get(method, {}).get("selected_count")
        if observed is None or pd.isna(observed):
            observed_int = None
            delta = None
        else:
            observed_int = int(observed)
            delta = observed_int - int(target)
        rows.append(
            {
                "method": method,
                "reference_count": int(target),
                "observed_count": observed_int,
                "delta": delta,
                "matches_reference": bool(delta == 0) if delta is not None else False,
            }
        )
    return pd.DataFrame.from_records(rows)


def _handle_count_mismatch(
    count_audit: pd.DataFrame,
    *,
    flight_name: str,
    action: str,
) -> None:
    """Warn or fail when strict paper row counts do not match the reference table."""

    _validate_count_mismatch_action(action)
    if action == "ignore" or count_audit.empty:
        return
    mismatches = count_audit.loc[~count_audit["matches_reference"].fillna(False).astype(bool)]
    if mismatches.empty:
        return
    message = (
        f"paper-strict reference-count mismatch for {flight_name}: "
        f"{_format_count_mismatches(mismatches)}"
    )
    if action == "fail":
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _format_count_mismatches(mismatches: pd.DataFrame) -> str:
    parts: list[str] = []
    for _, row in mismatches.iterrows():
        parts.append(
            f"{row['method']} observed={row['observed_count']} "
            f"reference={row['reference_count']} delta={row['delta']}"
        )
    return "; ".join(parts)


def _validate_count_mismatch_action(action: str) -> None:
    if action not in COUNT_MISMATCH_ACTIONS:
        raise ValueError(f"count_mismatch_action must be one of {COUNT_MISMATCH_ACTIONS}")


def measurement_covariances(
    *,
    inputs: PaperStrictInputs,
    preselected_radar: pd.DataFrame,
    config: PaperStrictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RF 2D and radar 3D covariance matrices for strict fusion."""

    rf_fallback = np.diag([float(config.rf_default_std_m) ** 2] * 2)
    radar_fallback = np.diag(
        [
            float(config.radar_default_xy_std_m) ** 2,
            float(config.radar_default_xy_std_m) ** 2,
            float(config.radar_default_z_std_m) ** 2,
        ]
    )
    if not config.empirical_covariance:
        return rf_fallback, radar_fallback

    rf_covariance = empirical_position_covariance_at_times(
        estimate_times_s=inputs.rf["time_s"].to_numpy(dtype=float),
        estimate_positions_m=inputs.rf[["east_m", "north_m"]].to_numpy(dtype=float),
        truth_times_s=inputs.truth["time_s"].to_numpy(dtype=float),
        truth_positions_m=inputs.truth[["east_m", "north_m"]].to_numpy(dtype=float),
        max_time_delta_s=config.truth_time_gate_s,
        dimensions=2,
    )
    radar_covariance = empirical_position_covariance_at_times(
        estimate_times_s=preselected_radar["time_s"].to_numpy(dtype=float),
        estimate_positions_m=preselected_radar[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        truth_times_s=inputs.truth["time_s"].to_numpy(dtype=float),
        truth_positions_m=inputs.truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        max_time_delta_s=config.truth_time_gate_s,
        dimensions=3,
    )
    return _safe_covariance(rf_covariance, rf_fallback), _safe_covariance(
        radar_covariance,
        radar_fallback,
    )


def paper_strict_range_gated_radar_candidates(
    radar: pd.DataFrame,
    *,
    range_gate_m: float = PAPER_STRICT_RANGE_GATE_M,
    catprob_threshold: float | None = None,
    require_range_m: bool = True,
) -> pd.DataFrame:
    """Return strict radar rows after the Fortem range gate and optional catProb gate."""

    if require_range_m:
        require_fortem_range_m(radar)
    pool = _range_candidate_pool(radar, range_gate_m=range_gate_m, require_range_m=require_range_m)
    pool = _catprob_candidate_pool(pool, catprob_threshold)
    return _sort_radar_rows(pool).reset_index(drop=True)


def select_paper_strict_radar_track(
    radar: pd.DataFrame,
    *,
    range_gate_m: float = PAPER_STRICT_RANGE_GATE_M,
    catprob_threshold: float | None = None,
    require_range_m: bool = True,
) -> pd.DataFrame:
    """Select the largest continuous Fortem track after a radar-range gate."""

    if require_range_m:
        require_fortem_range_m(radar)
    pool = paper_strict_range_gated_radar_candidates(
        radar,
        range_gate_m=range_gate_m,
        catprob_threshold=catprob_threshold,
        require_range_m=require_range_m,
    )
    if pool.empty or "track_id" not in pool.columns:
        return radar.iloc[0:0].copy()
    segments = _continuous_track_segments(pool)
    if not segments:
        return radar.iloc[0:0].copy()
    selected_segment = max(
        segments,
        key=lambda segment: (
            int(len(segment)),
            float(segment["time_s"].iloc[-1] - segment["time_s"].iloc[0]),
            _mean_catprob(segment),
            -float(segment["time_s"].iloc[0]),
            -int(pd.to_numeric(segment["track_id"], errors="coerce").iloc[0]),
        ),
    )
    selected = selected_segment.copy()
    selected["association_mode"] = "paper-strict-largest-continuous-track"
    selected["association_action"] = "range_gated_largest_continuous_track_anchor"
    selected["association_range_gate_m"] = float(range_gate_m)
    selected["association_preselector_raw_rows"] = int(len(radar))
    selected["association_preselector_range_gated_rows"] = int(len(pool))
    selected["association_segment_frames"] = int(len(selected))
    if catprob_threshold is not None:
        selected["association_catprob_threshold"] = float(catprob_threshold)
    return _sort_radar_rows(selected).reset_index(drop=True)


def paper_strict_stage_counts(
    *,
    inputs: PaperStrictInputs,
    fusion: PaperStrictFusionResult,
) -> dict[str, int]:
    """Return count fingerprints for paper-parity debugging."""

    estimates = _records_to_estimate_frame(fusion.records)
    if estimates.empty:
        updated = 0
        coasted = 0
    else:
        accepted = estimates["accepted"].fillna(False).astype(bool)
        updated = int(accepted.sum())
        coasted = int((~accepted).sum())
    return {
        "rf_raw_rows": int(len(inputs.rf)),
        "rf_after_nis_rows": int(len(fusion.accepted_rf)),
        "radar_all_track_rows": int(len(inputs.radar)),
        "radar_after_range_gate_rows": int(len(fusion.range_gated_radar)),
        "radar_largest_continuous_track_rows": int(len(fusion.preselected_radar)),
        "radar_after_nis_rows": int(len(fusion.accepted_radar)),
        "kf_all_steps_rows": int(len(estimates)),
        "kf_updated_rows": updated,
        "kf_coasted_rows": coasted,
    }


def require_fortem_range_m(radar: pd.DataFrame) -> None:
    """Fail if strict range gating would have to fall back to ENU distance."""

    if radar.empty:
        return
    if "range_m" not in radar.columns:
        raise ValueError(
            "paper-strict range gating requires Fortem range_m; pass "
            "--allow-missing-radar-range only for non-parity diagnostics"
        )
    ranges = pd.to_numeric(radar["range_m"], errors="coerce").to_numpy(dtype=float)
    finite_fraction = float(np.mean(np.isfinite(ranges))) if ranges.size else 0.0
    if finite_fraction < 0.99:
        raise ValueError(
            f"paper-strict range gating requires finite range_m for >=99% of rows; "
            f"observed {finite_fraction:.3f}"
        )


def radar_range_audit(radar: pd.DataFrame) -> dict[str, Any]:
    """Return a compact audit of Fortem range availability and ENU-norm fallback risk."""

    if radar.empty:
        return {"rows": 0, "range_m_present": "range_m" in radar.columns}
    audit: dict[str, Any] = {"rows": int(len(radar)), "range_m_present": "range_m" in radar.columns}
    if "range_m" not in radar.columns:
        return audit
    range_m = pd.to_numeric(radar["range_m"], errors="coerce").to_numpy(dtype=float)
    positions = radar[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    enu_norm = np.linalg.norm(positions, axis=1)
    finite = np.isfinite(range_m) & np.isfinite(enu_norm)
    audit["finite_range_fraction"] = float(np.mean(np.isfinite(range_m))) if range_m.size else 0.0
    if finite.any():
        delta = range_m[finite] - enu_norm[finite]
        audit["range_minus_enu_norm_mean_m"] = float(np.mean(delta))
        audit["range_minus_enu_norm_p95_abs_m"] = float(np.percentile(np.abs(delta), 95.0))
        audit["range_minus_enu_norm_max_abs_m"] = float(np.max(np.abs(delta)))
    return audit


def rf_measurements_with_covariance(rf: pd.DataFrame, covariance: np.ndarray) -> list[TrackingMeasurement]:
    """Convert normalized RF rows to 2D measurements with a shared covariance."""

    cov = np.asarray(covariance, dtype=float).reshape(2, 2)
    measurements: list[TrackingMeasurement] = []
    for _, row in rf.sort_values("time_s").iterrows():
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=np.array([float(row["east_m"]), float(row["north_m"])]),
                covariance=cov,
                source="rf",
            )
        )
    return measurements


def radar_measurement_from_row(row: pd.Series, covariance: np.ndarray) -> TrackingMeasurement:
    """Convert one selected radar row into a 3D measurement."""

    return TrackingMeasurement(
        time_s=float(row["time_s"]),
        vector=np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])]),
        covariance=np.asarray(covariance, dtype=float).reshape(3, 3),
        source="radar",
    )


def _metric_row(
    *,
    method: str,
    source: str,
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    dimensions: int,
    candidate_count: int,
    selected_count: int,
    max_time_delta_s: float,
) -> dict[str, Any]:
    if frame.empty:
        errors = np.empty(0, dtype=float)
        matched_count = 0
    else:
        positions = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
        errors = position_errors_at_times_m(
            estimate_times_s=frame["time_s"].to_numpy(dtype=float),
            estimate_positions_m=positions,
            truth_times_s=truth["time_s"].to_numpy(dtype=float),
            truth_positions_m=truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
            max_time_delta_s=max_time_delta_s,
            dimensions=dimensions,
        )
        matched_count = int(errors.size)
    summary = summarize_errors(errors)
    reference = PAPER_REFERENCE_ERROR_M.get(method, {})
    row: dict[str, Any] = {
        "method": method,
        "source": source,
        "paper_dimensions": int(dimensions),
        "candidate_count": int(candidate_count),
        "selected_count": int(selected_count),
        "matched_count": matched_count,
        "coverage": _safe_ratio(matched_count, candidate_count),
        "reference_count": PAPER_REFERENCE_COUNTS.get(method),
        "count_delta": None
        if method not in PAPER_REFERENCE_COUNTS
        else int(selected_count) - int(PAPER_REFERENCE_COUNTS[method]),
        "paper_error_mean_m": summary["mean_m"],
        "paper_error_std_m": summary["std_m"],
        "paper_error_rmse_m": summary["rmse_m"],
        "paper_error_p50_m": summary["p50_m"],
        "paper_error_p95_m": summary["p95_m"],
        "paper_error_max_m": summary["max_m"],
        "reference_mean_m": reference.get("mean"),
        "reference_std_m": reference.get("std"),
        "reference_max_m": reference.get("max"),
    }
    if reference:
        row["mean_delta_m"] = None if summary["mean_m"] is None else float(summary["mean_m"]) - float(reference["mean"])
        row["std_delta_m"] = None if summary["std_m"] is None else float(summary["std_m"]) - float(reference["std"])
        row["max_delta_m"] = None if summary["max_m"] is None else float(summary["max_m"]) - float(reference["max"])
    return row


def _events(
    *,
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "time_s": float(measurement.time_s),
            "priority": 0,
            "kind": "rf",
            "measurement": measurement,
        }
        for measurement in rf_measurements
    ]
    for group in _radar_frame_groups(radar):
        events.append(
            {
                "time_s": float(group["time_s"].median()),
                "priority": 1,
                "kind": "radar",
                "candidates": group,
            }
        )
    return sorted(events, key=lambda item: (float(item["time_s"]), int(item["priority"])))


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar.empty:
        return []
    sort_columns = [column for column in ("time_s", "frame_index", "track_id", "track_index") if column in radar.columns]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def _bootstrap_event(
    events: list[dict[str, object]],
    *,
    radar_by_key: dict[object, pd.Series],
    radar_covariance: np.ndarray,
    bootstrap_source: str,
) -> tuple[int, TrackingMeasurement, pd.Series | None] | None:
    if bootstrap_source not in {"radar", "first-event"}:
        raise ValueError("bootstrap_source must be 'radar' or 'first-event'")
    for index, event in enumerate(events):
        if event["kind"] == "radar":
            row = radar_by_key.get(_radar_event_key(event))
            if row is None:
                continue
            return int(index), radar_measurement_from_row(row, radar_covariance), row.copy()
        if bootstrap_source == "first-event" and event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            return int(index), measurement, None
    if bootstrap_source == "radar":
        return None
    return None


def _tracking_record(
    measurement: TrackingMeasurement,
    tracker: AsyncConstantVelocityKalmanTracker,
    diagnostics: TrackingUpdateDiagnostics,
    *,
    association_mode: str,
    selected_row: pd.Series | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "time_s": float(measurement.time_s),
        "source": measurement.source,
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        "measurement_vector": np.asarray(measurement.vector, dtype=float).copy(),
        "measurement_covariance": np.asarray(measurement.covariance, dtype=float).copy(),
        "association_mode": association_mode,
        **diagnostics.to_record(),
    }
    if selected_row is not None:
        if "track_id" in selected_row.index and pd.notna(selected_row["track_id"]):
            record["track_id"] = int(float(selected_row["track_id"]))
        if "frame_index" in selected_row.index and pd.notna(selected_row["frame_index"]):
            record["frame_index"] = int(float(selected_row["frame_index"]))
    return record


def _coast_record(
    *,
    time_s: float,
    tracker: AsyncConstantVelocityKalmanTracker,
    association_mode: str,
    gate_threshold: float,
    reason: str,
) -> dict[str, object]:
    diagnostics = TrackingUpdateDiagnostics(
        time_s=float(time_s),
        source="radar",
        measurement_dim=3,
        accepted=False,
        update_action="missed_detection",
        nis=float("nan"),
        gate_threshold=float(gate_threshold),
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=float("nan"),
    )
    return {
        "time_s": float(time_s),
        "source": "radar",
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        "association_mode": association_mode,
        "association_missed_detection_reason": reason,
        **diagnostics.to_record(),
    }


def _records_to_estimate_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(-1)
        row: dict[str, Any] = {
            "time_s": float(record["time_s"]),
            "source": record.get("source"),
            "east_m": float(state[0]),
            "north_m": float(state[1]),
            "up_m": float(state[2]),
            "v_east_mps": float(state[3]) if state.size > 3 else np.nan,
            "v_north_mps": float(state[4]) if state.size > 4 else np.nan,
            "v_up_mps": float(state[5]) if state.size > 5 else np.nan,
            "accepted": bool(record.get("accepted", False)),
            "update_action": record.get("update_action"),
            "nis": record.get("nis"),
            "gate_threshold": record.get("gate_threshold"),
            "association_mode": record.get("association_mode"),
        }
        for optional in ("track_id", "frame_index", "association_missed_detection_reason"):
            if optional in record:
                row[optional] = record[optional]
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
    segments: list[pd.DataFrame] = []
    if radar.empty or "track_id" not in radar.columns:
        return segments
    sort_key = "frame_index" if "frame_index" in radar.columns else "time_s"
    for _, track_rows in radar.groupby("track_id", sort=True):
        ordered = track_rows.sort_values([sort_key, "time_s"]).reset_index(drop=True)
        values = pd.to_numeric(ordered[sort_key], errors="coerce").to_numpy(dtype=float)
        if values.size == 0:
            continue
        splits = np.r_[0, np.where(np.diff(values) > _segment_gap_threshold(values))[0] + 1, len(ordered)]
        for start, end in zip(splits[:-1], splits[1:]):
            segment = ordered.iloc[int(start) : int(end)].copy()
            if not segment.empty:
                segments.append(segment)
    return segments


def _segment_gap_threshold(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("inf")
    diffs = np.diff(np.sort(finite))
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return float("inf")
    if np.allclose(finite, np.round(finite)):
        return 1.5
    return 1.5 * float(np.median(positive))


def _range_candidate_pool(
    candidates: pd.DataFrame,
    *,
    range_gate_m: float,
    require_range_m: bool,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    if require_range_m:
        ranges = pd.to_numeric(candidates["range_m"], errors="coerce").to_numpy(dtype=float)
    elif "range_m" in candidates.columns:
        ranges = pd.to_numeric(candidates["range_m"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(ranges).any():
            ranges = np.linalg.norm(candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float), axis=1)
    else:
        ranges = np.linalg.norm(candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float), axis=1)
    pool = candidates.loc[np.isfinite(ranges) & (ranges <= float(range_gate_m))].copy()
    pool["association_range_gate_m"] = float(range_gate_m)
    return pool


def _catprob_candidate_pool(candidates: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if candidates.empty or threshold is None:
        return candidates.copy()
    if "cat_prob_uav" not in candidates.columns:
        return candidates.iloc[0:0].copy()
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    pool = candidates.loc[catprob >= float(threshold)].copy()
    pool["association_catprob_threshold"] = float(threshold)
    return pool


def _mean_catprob(frame: pd.DataFrame) -> float:
    if "cat_prob_uav" not in frame.columns:
        return 1.0
    values = pd.to_numeric(frame["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return 0.0
    return float(np.nanmean(values))


def _radar_event_key(event: dict[str, object]) -> object:
    candidates = event["candidates"]
    assert isinstance(candidates, pd.DataFrame)
    if "frame_index" in candidates.columns:
        values = pd.to_numeric(candidates["frame_index"], errors="coerce").dropna()
        if not values.empty:
            return ("frame_index", int(values.iloc[0]))
    return ("time_s", round(float(event["time_s"]), 9))


def _radar_row_key(row: pd.Series) -> object:
    if "frame_index" in row.index and pd.notna(row["frame_index"]):
        value = float(row["frame_index"])
        if np.isfinite(value):
            return ("frame_index", int(value))
    return ("time_s", round(float(row["time_s"]), 9))


def _rf_row_at_time(rf: pd.DataFrame, time_s: float) -> pd.Series | None:
    if rf.empty:
        return None
    times = rf["time_s"].to_numpy(dtype=float)
    index = int(np.argmin(np.abs(times - float(time_s))))
    if abs(float(times[index]) - float(time_s)) > 1.0e-9:
        return None
    return rf.iloc[index].copy()


def _selected_rows_frame(reference: pd.DataFrame, rows: list[pd.Series]) -> pd.DataFrame:
    if not rows:
        return reference.iloc[0:0].copy()
    return _sort_radar_rows(pd.DataFrame(rows)).reset_index(drop=True)


def _sort_radar_rows(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ("time_s", "frame_index", "track_id", "track_index") if column in frame.columns]
    return frame.sort_values(sort_columns) if sort_columns else frame


def _safe_covariance(candidate: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    covariance = np.asarray(candidate, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    if covariance.shape != fallback.shape:
        return fallback
    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all():
        return fallback
    try:
        eigvals = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        return fallback
    if np.min(eigvals) <= 0.0:
        jitter = abs(float(np.min(eigvals))) + 1.0e-6
        covariance = covariance + np.eye(covariance.shape[0]) * jitter
    return covariance


def _projector_for_origin(
    *,
    enu_origin: str,
    enu_origin_lla: str | None,
    lw1_origin_lla: str | None,
    origin_config: Path | None = None,
) -> LocalENUProjector | None:
    if enu_origin == "truth-first":
        return None
    if enu_origin == "lla":
        if not enu_origin_lla:
            raise ValueError("--enu-origin lla requires --enu-origin-lla LAT,LON,ALT")
        return projector_from_lla(*_parse_lla(enu_origin_lla))
    if enu_origin == "lw1":
        origin = lw1_origin_lla or os.environ.get(PAPER_STRICT_LW1_ORIGIN_LLA_ENV)
        lla = _parse_lla(origin) if origin else _origin_lla_from_config(origin_config, "lw1")
        if lla is None:
            raise ValueError(
                "--enu-origin lw1 requires --lw1-origin-lla LAT,LON,ALT, "
                f"{PAPER_STRICT_LW1_ORIGIN_LLA_ENV}=LAT,LON,ALT, or an origin config"
            )
        return projector_from_lla(*lla)
    raise ValueError(f"unknown enu_origin {enu_origin!r}")


def _origin_lla_from_config(
    origin_config: Path | None,
    name: str,
) -> tuple[float, float, float] | None:
    raw_path = origin_config or os.environ.get(PAPER_STRICT_ORIGINS_FILE_ENV)
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"origin config does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    origins = payload.get("origins", payload) if isinstance(payload, dict) else {}
    entry = origins.get(name) if isinstance(origins, dict) else None
    if entry is None:
        return None
    return _origin_entry_to_lla(entry, name)


def _origin_entry_to_lla(entry: Any, name: str) -> tuple[float, float, float]:
    if isinstance(entry, (list, tuple)) and len(entry) == 3:
        return float(entry[0]), float(entry[1]), float(entry[2])
    if not isinstance(entry, dict):
        raise ValueError(f"origin {name!r} must be a mapping or a 3-item list")
    latitude = _origin_value(entry, "latitude_deg", "latitude", "lat")
    longitude = _origin_value(entry, "longitude_deg", "longitude", "lon", "lng")
    altitude = _origin_value(entry, "altitude_m", "altitude", "alt")
    missing = [
        label
        for label, value in (
            ("latitude_deg", latitude),
            ("longitude_deg", longitude),
            ("altitude_m", altitude),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"origin {name!r} missing required fields: {', '.join(missing)}")
    return float(latitude), float(longitude), float(altitude)


def _origin_value(entry: dict[str, Any], *names: str) -> float | None:
    for key in names:
        if key in entry and entry[key] is not None:
            return float(entry[key])
    return None


def _parse_lla(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError("LLA origin must have the form LAT,LON,ALT")
    lat, lon, alt = (float(part) for part in parts)
    return lat, lon, alt


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame.copy()
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


def _resolve_flights(dataset_root: Path, flights: Iterable[str] | None) -> list[str]:
    requested = list(flights or [])
    if requested:
        return [select_flight(dataset_root, name).name for name in requested]
    return [flight.name for flight in discover_flights(dataset_root)]


def _validate_config(config: PaperStrictConfig) -> None:
    if config.range_gate_m <= 0.0:
        raise ValueError("range_gate_m must be positive")
    if not 0.0 < config.nis_gate_probability < 1.0:
        raise ValueError("nis_gate_probability must be in (0, 1)")
    if config.rf_nis_gate_probability is not None:
        if not 0.0 < config.rf_nis_gate_probability < 1.0:
            raise ValueError("rf_nis_gate_probability must be in (0, 1) or None")
    if config.truth_time_gate_s <= 0.0:
        raise ValueError("truth_time_gate_s must be positive")
    if config.acceleration_std_mps2 <= 0.0:
        raise ValueError("acceleration_std_mps2 must be positive")
    if config.radar_catprob_threshold is not None and not 0.0 <= config.radar_catprob_threshold <= 1.0:
        raise ValueError("radar_catprob_threshold must be in [0, 1] or None")
    if config.bootstrap_source not in {"radar", "first-event"}:
        raise ValueError("bootstrap_source must be 'radar' or 'first-event'")


def _jsonable_config(config: PaperStrictConfig) -> dict[str, Any]:
    return {
        "range_gate_m": float(config.range_gate_m),
        "nis_gate_probability": float(config.nis_gate_probability),
        "rf_nis_gate_probability": config.rf_nis_gate_probability,
        "truth_time_gate_s": float(config.truth_time_gate_s),
        "acceleration_std_mps2": float(config.acceleration_std_mps2),
        "radar_catprob_threshold": config.radar_catprob_threshold,
        "empirical_covariance": bool(config.empirical_covariance),
        "require_radar_range_m": bool(config.require_radar_range_m),
        "bootstrap_source": config.bootstrap_source,
        "rf_default_std_m": float(config.rf_default_std_m),
        "radar_default_xy_std_m": float(config.radar_default_xy_std_m),
        "radar_default_z_std_m": float(config.radar_default_z_std_m),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
