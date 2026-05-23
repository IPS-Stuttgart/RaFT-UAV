"""Paper-reference parity scoring utilities.

The AERPAW Dataset-28 paper reports a count/error fingerprint, not just a
single RMSE.  This module makes that fingerprint available to the general
baseline path so result tables can detect reproduction drift before comparing
tracker variants.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

PAPER_REFERENCE_COUNTS: dict[str, int] = {
    "RF raw": 206,
    "RF after NIS": 125,
    "Radar raw": 3106,
    "Radar after NIS": 2403,
    "KF all steps": 2655,
    "KF updated": 2528,
    "KF coasted": 127,
}

PAPER_REFERENCE_ERROR_M: dict[str, dict[str, float]] = {
    "RF raw": {"mean": 471.8, "std": 885.2, "max": 4831.6},
    "RF after NIS": {"mean": 25.8, "std": 16.2, "max": 113.9},
    "Radar raw": {"mean": 26.2, "std": 25.5, "max": 195.6},
    "Radar after NIS": {"mean": 21.0, "std": 17.1, "max": 97.2},
    "KF all steps": {"mean": 21.9, "std": 17.9, "max": 109.1},
}


def build_baseline_paper_parity(
    *,
    stage_counts: Mapping[str, Any] | None,
    rf_rows: int,
    radar_rows: int,
    selected_radar_rows: int,
    posterior_records: int,
    accepted_by_source: Mapping[str, int],
    rejected_by_source: Mapping[str, int],
    paper_position_error_3d: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a compact paper-parity report for a baseline metrics payload.

    ``stage_counts`` is expected to come from ``selected_radar.attrs`` when the
    association mode exposes paper-compatible internal counts.  When it is not
    present, the function still returns best-effort counts so non-paper methods
    can be flagged as non-comparable instead of silently sorted against the
    reference paper.
    """

    stages = dict(stage_counts or {})
    accepted = {str(key): int(value) for key, value in accepted_by_source.items()}
    rejected = {str(key): int(value) for key, value in rejected_by_source.items()}
    updated = int(stages.get("kf_updated_rows", sum(accepted.values())))
    coasted = int(stages.get("kf_coasted_rows", sum(rejected.values())))
    observed_counts = {
        "RF raw": _optional_int(stages.get("rf_raw_rows", rf_rows)),
        "RF after NIS": _optional_int(stages.get("rf_after_nis_rows", accepted.get("rf", 0))),
        "Radar raw": _optional_int(
            stages.get("radar_raw_target_track_rows", stages.get("radar_raw_rows", radar_rows))
        ),
        "Radar after NIS": _optional_int(
            stages.get("radar_after_nis_rows", selected_radar_rows)
        ),
        "KF all steps": _optional_int(stages.get("kf_all_steps_rows", posterior_records)),
        "KF updated": _optional_int(stages.get("kf_updated_rows", updated)),
        "KF coasted": _optional_int(stages.get("kf_coasted_rows", coasted)),
    }
    count_checks = _count_checks(observed_counts)
    error_checks = _kf_error_checks(paper_position_error_3d)
    count_abs_delta_total = int(
        sum(abs(int(row["delta"])) for row in count_checks.values() if row["delta"] is not None)
    )
    error_abs_delta_total_m = float(
        sum(
            abs(float(row["delta_m"]))
            for row in error_checks.values()
            if row["delta_m"] is not None and np.isfinite(float(row["delta_m"]))
        )
    )
    missing_reference_counts = [
        method for method, row in count_checks.items() if row["observed"] is None
    ]
    count_matches = [
        bool(row["matches_reference"])
        for row in count_checks.values()
        if row["matches_reference"] is not None
    ]
    return {
        "reference_counts": PAPER_REFERENCE_COUNTS,
        "reference_errors_m": PAPER_REFERENCE_ERROR_M,
        "observed_counts": observed_counts,
        "count_checks": count_checks,
        "error_checks": error_checks,
        "count_abs_delta_total": count_abs_delta_total,
        "error_abs_delta_total_m": error_abs_delta_total_m,
        "score": float(1000.0 * count_abs_delta_total + error_abs_delta_total_m),
        "all_count_matches_reference": bool(count_matches and all(count_matches)),
        "missing_reference_counts": missing_reference_counts,
        "stage_count_source": "association_attrs" if stage_counts is not None else "fallback",
        "metric_protocol": "sample_time_nearest_or_interpolated_truth_mean_std_max",
    }


def paper_stage_counts_from_records(
    *,
    records: list[dict[str, object]],
    rf_raw_rows: int,
    radar_raw_target_track_rows: int,
    radar_after_range_gate_rows: int,
    radar_preselected_rows: int,
    radar_after_nis_rows: int,
    radar_all_track_rows: int,
) -> dict[str, int]:
    """Build the standard paper-count fingerprint from association records."""

    accepted_rf = 0
    updated = 0
    coasted = 0
    for record in records:
        accepted = bool(record.get("accepted", False))
        source = str(record.get("source", ""))
        if accepted:
            updated += 1
            if source == "rf":
                accepted_rf += 1
        else:
            coasted += 1
    return {
        "rf_raw_rows": int(rf_raw_rows),
        "rf_after_nis_rows": int(accepted_rf),
        "radar_all_track_rows": int(radar_all_track_rows),
        "radar_raw_target_track_rows": int(radar_raw_target_track_rows),
        "radar_after_range_gate_rows": int(radar_after_range_gate_rows),
        "radar_largest_continuous_track_rows": int(radar_preselected_rows),
        "radar_after_nis_rows": int(radar_after_nis_rows),
        "kf_all_steps_rows": int(len(records)),
        "kf_updated_rows": int(updated),
        "kf_coasted_rows": int(coasted),
    }


def parity_score_from_count_audit(
    count_audit_rows: list[Mapping[str, Any]],
    *,
    error_delta_m: float | None = None,
) -> float:
    """Return a scalar score for offset sweeps; lower is better."""

    count_penalty = 0.0
    for row in count_audit_rows:
        delta = _optional_float(row.get("delta"))
        if delta is not None:
            count_penalty += abs(delta)
    error_penalty = 0.0 if error_delta_m is None else abs(float(error_delta_m))
    return float(1000.0 * count_penalty + error_penalty)


def _count_checks(observed_counts: Mapping[str, int | None]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for method, reference in PAPER_REFERENCE_COUNTS.items():
        observed = observed_counts.get(method)
        delta = None if observed is None else int(observed) - int(reference)
        rows[method] = {
            "observed": observed,
            "reference": int(reference),
            "delta": delta,
            "matches_reference": None if delta is None else bool(delta == 0),
        }
    return rows


def _kf_error_checks(paper_position_error_3d: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    reference = PAPER_REFERENCE_ERROR_M["KF all steps"]
    observed = {
        "mean": _optional_float(paper_position_error_3d.get("mean_m")),
        "std": _optional_float(paper_position_error_3d.get("std_m")),
        "max": _optional_float(paper_position_error_3d.get("max_m")),
    }
    rows: dict[str, dict[str, Any]] = {}
    for key, observed_value in observed.items():
        reference_value = float(reference[key])
        delta = None if observed_value is None else float(observed_value) - reference_value
        rows[f"KF all steps {key}"] = {
            "observed_m": observed_value,
            "reference_m": reference_value,
            "delta_m": delta,
        }
    return rows


def _optional_int(value: Any) -> int | None:
    scalar = _optional_float(value)
    return None if scalar is None else int(round(scalar))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None
