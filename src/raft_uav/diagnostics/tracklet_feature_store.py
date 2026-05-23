"""Radar candidate feature-store and counterfactual association diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features
from raft_uav.diagnostics.paper_strict import (
    PAPER_STRICT_RANGE_GATE_M,
    load_paper_strict_inputs,
    select_paper_strict_radar_track,
)
from raft_uav.evaluation.metrics import interpolate_positions_at_times
from raft_uav.io.aerpaw import discover_flights, select_flight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-tracklet-feature-store",
        description=(
            "write radar candidate features, oracle ranks, and selected-vs-best regret rows"
        ),
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracklet-feature-store"))
    parser.add_argument(
        "--variant",
        choices=["auto", "original", "rerun"],
        default="auto",
        help="RF/radar/truth file variant; auto preserves historical rerun preference",
    )
    parser.add_argument(
        "--enu-origin",
        choices=["truth-first", "lla", "lw1"],
        default="lw1",
    )
    parser.add_argument("--enu-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lla")
    parser.add_argument("--lw1-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lw1")
    parser.add_argument("--origin-config", type=Path, default=None)
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--range-gate-m", type=float, default=PAPER_STRICT_RANGE_GATE_M)
    parser.add_argument("--radar-catprob-threshold", type=float, default=None)
    parser.add_argument(
        "--selected-radar-csv",
        type=Path,
        default=None,
        help=(
            "optional selected_radar.csv from any run. If omitted, the paper-strict largest "
            "continuous track is used as the selected path."
        ),
    )
    parser.add_argument("--rf-default-std-m", type=float, default=75.0)
    args = parser.parse_args(argv)

    result = run_tracklet_feature_store(
        dataset_root=args.dataset_root,
        flights=args.flight,
        output_dir=args.output_dir,
        variant=args.variant,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
        truth_time_gate_s=args.truth_time_gate_s,
        range_gate_m=args.range_gate_m,
        radar_catprob_threshold=args.radar_catprob_threshold,
        selected_radar_csv=args.selected_radar_csv,
        rf_default_std_m=args.rf_default_std_m,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"summary_csv={result['summary_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def run_tracklet_feature_store(
    *,
    dataset_root: Path,
    flights: Iterable[str] | None,
    output_dir: Path = Path("outputs/tracklet-feature-store"),
    variant: str = "auto",
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
    truth_time_gate_s: float = 2.0,
    range_gate_m: float = PAPER_STRICT_RANGE_GATE_M,
    radar_catprob_threshold: float | None = None,
    selected_radar_csv: Path | None = None,
    rf_default_std_m: float = 75.0,
) -> dict[str, Any]:
    """Write radar candidate feature stores and counterfactual regret reports."""

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
            origin_config=origin_config,
            rf_default_std_m=rf_default_std_m,
            variant=variant,
        )
        selected = _selected_radar_for_flight(
            inputs.radar,
            selected_radar_csv=selected_radar_csv,
            range_gate_m=range_gate_m,
            catprob_threshold=radar_catprob_threshold,
        )
        features = build_tracklet_candidate_feature_store(
            radar=inputs.radar,
            truth=inputs.truth,
            selected_radar=selected,
            truth_time_gate_s=truth_time_gate_s,
        )
        regret = build_counterfactual_association_dashboard(features)
        summary = summarize_counterfactual_regret(regret)
        row = {
            "flight": inputs.flight_name,
            **summary,
        }

        flight_dir = output / inputs.flight_name
        flight_dir.mkdir(parents=True, exist_ok=True)
        features_csv = flight_dir / "tracklet_candidate_features.csv"
        regret_csv = flight_dir / "counterfactual_association_dashboard.csv"
        selected_csv = flight_dir / "selected_radar_used.csv"
        manifest_json = flight_dir / "tracklet_feature_store_manifest.json"
        features.to_csv(features_csv, index=False)
        regret.to_csv(regret_csv, index=False)
        selected.to_csv(selected_csv, index=False)
        manifest = {
            **row,
            "features_csv": str(features_csv),
            "regret_csv": str(regret_csv),
            "selected_csv": str(selected_csv),
            "selected_source": (
                "external_csv"
                if selected_radar_csv is not None
                else "paper_strict_largest_track"
            ),
            "selected_radar_csv": None if selected_radar_csv is None else str(selected_radar_csv),
            "file_manifest": inputs.file_manifest,
        }
        manifest_json.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        row["manifest_json"] = str(manifest_json)
        row["features_csv"] = str(features_csv)
        row["regret_csv"] = str(regret_csv)
        summary_rows.append(row)
        manifests.append(manifest)

    summary_frame = pd.DataFrame.from_records(summary_rows)
    summary_csv = output / "tracklet_feature_store_summary.csv"
    summary_json = output / "tracklet_feature_store_summary.json"
    summary_frame.to_csv(summary_csv, index=False)
    payload = {
        "output_dir": str(output),
        "summary_csv": str(summary_csv),
        "variant": variant,
        "flights": manifests,
    }
    summary_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return {**payload, "summary_json": str(summary_json)}


def build_tracklet_candidate_feature_store(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    selected_radar: pd.DataFrame | None = None,
    truth_time_gate_s: float = 2.0,
) -> pd.DataFrame:
    """Return one row per radar candidate with oracle rank and selection labels.

    Ground truth columns are diagnostic labels only.  They are meant for LOFO
    scorer training, audit dashboards, and candidate-pool regret analysis, not
    for online association at inference time.
    """

    if radar.empty:
        return pd.DataFrame(columns=_FEATURE_COLUMNS)
    out = add_track_level_features(radar).copy()
    out = _append_frame_keys(out)
    out = _append_truth_errors(out, truth, truth_time_gate_s=truth_time_gate_s)
    out["candidate_count_in_frame"] = out.groupby(["frame_key_type", "frame_key"], dropna=False)[
        "time_s"
    ].transform("size")
    out["oracle_rank_in_frame"] = out.groupby(["frame_key_type", "frame_key"], dropna=False)[
        "oracle_error_m"
    ].rank(method="first", ascending=True, na_option="bottom")
    out["oracle_best_candidate"] = out["oracle_rank_in_frame"] == 1.0
    out["chosen_by_selected_radar"] = _selection_mask(out, selected_radar)
    out["selected_oracle_error_m"] = np.where(
        out["chosen_by_selected_radar"].astype(bool), out["oracle_error_m"], np.nan
    )
    return _ordered_feature_columns(out)


def build_counterfactual_association_dashboard(features: pd.DataFrame) -> pd.DataFrame:
    """Return one row per radar frame comparing selected and oracle-best candidates."""

    if features.empty:
        return pd.DataFrame(columns=_REGRET_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (key_type, key), group in features.groupby(
        ["frame_key_type", "frame_key"], sort=True, dropna=False
    ):
        ordered = group.sort_values("oracle_rank_in_frame", na_position="last")
        finite_errors = pd.to_numeric(ordered["oracle_error_m"], errors="coerce")
        best = ordered.loc[finite_errors.idxmin()] if finite_errors.notna().any() else None
        selected = ordered.loc[ordered["chosen_by_selected_radar"].fillna(False).astype(bool)]
        selected_row = selected.iloc[0] if not selected.empty else None
        best_error = _row_float(best, "oracle_error_m") if best is not None else np.nan
        selected_error = (
            _row_float(selected_row, "oracle_error_m")
            if selected_row is not None
            else np.nan
        )
        selected_rank = (
            _row_float(selected_row, "oracle_rank_in_frame")
            if selected_row is not None
            else np.nan
        )
        regret = (
            selected_error - best_error
            if np.isfinite([selected_error, best_error]).all()
            else np.nan
        )
        rows.append(
            {
                "frame_key_type": key_type,
                "frame_key": key,
                "time_s": float(pd.to_numeric(group["time_s"], errors="coerce").median()),
                "candidate_count": int(len(group)),
                "truth_available": bool(finite_errors.notna().any()),
                "best_candidate_error_m": best_error,
                "best_candidate_track_id": _row_value(best, "track_id"),
                "best_candidate_track_index": _row_value(best, "track_index"),
                "selected_present": selected_row is not None,
                "selected_candidate_error_m": selected_error,
                "selected_candidate_rank": selected_rank,
                "selected_candidate_track_id": _row_value(selected_row, "track_id"),
                "selected_candidate_track_index": _row_value(selected_row, "track_index"),
                "selection_regret_m": regret,
                "category": _regret_category(best_error, selected_error, selected_row is not None),
            }
        )
    return pd.DataFrame.from_records(rows, columns=_REGRET_COLUMNS)


def summarize_counterfactual_regret(regret: pd.DataFrame) -> dict[str, Any]:
    """Summarize counterfactual selected-vs-best radar association regret."""

    if regret.empty:
        return {
            "radar_frame_count": 0,
            "truth_matched_frame_count": 0,
            "selected_frame_count": 0,
        }
    truth_rows = regret.loc[regret["truth_available"].fillna(False).astype(bool)]
    selected_rows = truth_rows.loc[truth_rows["selected_present"].fillna(False).astype(bool)]
    finite_regret = pd.to_numeric(selected_rows["selection_regret_m"], errors="coerce").dropna()
    categories = regret["category"].value_counts(dropna=False).to_dict()
    out: dict[str, Any] = {
        "radar_frame_count": int(len(regret)),
        "truth_matched_frame_count": int(len(truth_rows)),
        "selected_frame_count": int(len(selected_rows)),
        "mean_regret_m": float(finite_regret.mean()) if len(finite_regret) else np.nan,
        "p95_regret_m": float(finite_regret.quantile(0.95)) if len(finite_regret) else np.nan,
        "max_regret_m": float(finite_regret.max()) if len(finite_regret) else np.nan,
        "selected_best_rate": (
            float((finite_regret <= 1.0e-9).mean())
            if len(finite_regret)
            else np.nan
        ),
    }
    for category, count in categories.items():
        out[f"category_{category}_count"] = int(count)
    return out


def _append_truth_errors(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    truth_time_gate_s: float,
) -> pd.DataFrame:
    out = radar.copy()
    truth_positions, valid = interpolate_positions_at_times(
        truth["time_s"].to_numpy(dtype=float),
        truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        out["time_s"].to_numpy(dtype=float),
        max_time_delta_s=truth_time_gate_s,
    )
    positions = out[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = positions - truth_positions
    errors = np.linalg.norm(residuals, axis=1)
    out["truth_available"] = valid.astype(bool)
    out["oracle_error_m"] = np.where(valid & np.isfinite(residuals).all(axis=1), errors, np.nan)
    out["oracle_residual_east_m"] = np.where(valid, residuals[:, 0], np.nan)
    out["oracle_residual_north_m"] = np.where(valid, residuals[:, 1], np.nan)
    out["oracle_residual_up_m"] = np.where(valid, residuals[:, 2], np.nan)
    return out


def _append_frame_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "frame_index" in out.columns:
        values = pd.to_numeric(out["frame_index"], errors="coerce")
        out["frame_key_type"] = "frame_index"
        out["frame_key"] = values.round().astype("Int64").astype(str)
    else:
        times = pd.to_numeric(out["time_s"], errors="coerce")
        out["frame_key_type"] = "time_s"
        out["frame_key"] = times.round(9).astype(str)
    return out


def _selection_mask(features: pd.DataFrame, selected_radar: pd.DataFrame | None) -> np.ndarray:
    if selected_radar is None or selected_radar.empty:
        return np.zeros(len(features), dtype=bool)
    selected = _append_frame_keys(selected_radar)
    selected_keys = {_candidate_match_key(row) for _, row in selected.iterrows()}
    return np.array([_candidate_match_key(row) in selected_keys for _, row in features.iterrows()])


def _candidate_match_key(row: pd.Series) -> tuple[object, ...]:
    key = [row.get("frame_key_type"), row.get("frame_key")]
    track_id = _optional_int(row.get("track_id"))
    track_index = _optional_int(row.get("track_index"))
    key.append(track_id)
    if track_index is not None:
        key.append(track_index)
    return tuple(key)


def _selected_radar_for_flight(
    radar: pd.DataFrame,
    *,
    selected_radar_csv: Path | None,
    range_gate_m: float,
    catprob_threshold: float | None,
) -> pd.DataFrame:
    if selected_radar_csv is not None:
        return pd.read_csv(selected_radar_csv)
    return select_paper_strict_radar_track(
        radar,
        range_gate_m=range_gate_m,
        catprob_threshold=catprob_threshold,
        require_range_m="range_m" in radar.columns,
    )


def _ordered_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    front = [column for column in _FEATURE_COLUMNS if column in frame.columns]
    rest = [column for column in frame.columns if column not in front]
    return frame[front + rest]


def _regret_category(best_error: float, selected_error: float, selected_present: bool) -> str:
    if not np.isfinite(best_error):
        return "no_truth"
    if not selected_present:
        return "best_candidate_not_selected"
    if not np.isfinite(selected_error):
        return "selected_candidate_without_truth"
    if selected_error <= best_error + 1.0e-9:
        return "oracle_best_selected"
    return "wrong_candidate_selected"


def _row_float(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row.index:
        return float("nan")
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _row_value(row: pd.Series | None, column: str) -> object:
    if row is None or column not in row.index:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    if column in {"track_id", "track_index"}:
        number = _optional_int(value)
        return "" if number is None else number
    return value


def _optional_int(value: object) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if np.isfinite(number) else None


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


_FEATURE_COLUMNS = [
    "frame_key_type",
    "frame_key",
    "time_s",
    "frame_index",
    "track_index",
    "track_id",
    "east_m",
    "north_m",
    "up_m",
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_velocity_mps",
    "rcs_dbsm",
    "cat_prob_uav",
    "track_age",
    "candidate_count_in_frame",
    "truth_available",
    "oracle_error_m",
    "oracle_rank_in_frame",
    "oracle_best_candidate",
    "chosen_by_selected_radar",
    "selected_oracle_error_m",
]
_REGRET_COLUMNS = [
    "frame_key_type",
    "frame_key",
    "time_s",
    "candidate_count",
    "truth_available",
    "best_candidate_error_m",
    "best_candidate_track_id",
    "best_candidate_track_index",
    "selected_present",
    "selected_candidate_error_m",
    "selected_candidate_rank",
    "selected_candidate_track_id",
    "selected_candidate_track_index",
    "selection_regret_m",
    "category",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
