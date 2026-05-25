"""Truth-oracle diagnostics for raw Fortem radar candidate pools.

This module is deliberately diagnostic-only.  It answers whether the raw radar
candidate pool can produce a paper-like target stream under plausible timestamp
and geometry interpretations before any non-oracle association or Kalman tuning
is attempted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_strict import load_paper_strict_inputs
from raft_uav.diagnostics.radar_geometry import AZIMUTH_CONVENTIONS, polar_offset_enu
from raft_uav.evaluation.metrics import interpolate_positions_at_times
from raft_uav.io.aerpaw import DEFAULT_RADAR_CLOCK_OFFSET_S, DEFAULT_RF_CLOCK_OFFSET_S

POSITION_SOURCES = (
    "fortem-lla",
    "polar-from-lw1",
    "polar-from-radar-origin",
)
ELEVATION_MODES = ("as-is", "inverted", "zero")
EVALUATION_WINDOW_HELP = (
    "truth-window or explicit:START_S:END_S.  Explicit windows are applied to "
    "radar times after the candidate audit's additional clock delta."
)


@dataclass(frozen=True)
class EvaluationWindow:
    """Radar-candidate audit time window in truth-relative seconds."""

    mode: str
    start_s: float | None = None
    end_s: float | None = None

    def contains(self, times_s: np.ndarray) -> np.ndarray:
        times = np.asarray(times_s, dtype=float)
        keep = np.isfinite(times)
        if self.start_s is not None:
            keep &= times >= float(self.start_s)
        if self.end_s is not None:
            keep &= times <= float(self.end_s)
        return keep

    def to_record(self) -> dict[str, float | str | None]:
        return {"mode": self.mode, "start_s": self.start_s, "end_s": self.end_s}


def run_radar_candidate_audit(
    *,
    dataset_root: Path,
    flight: str,
    output_dir: Path = Path("outputs/radar-candidate-audit"),
    variant: str = "auto",
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
    rf_clock_offset_s: float = DEFAULT_RF_CLOCK_OFFSET_S,
    radar_clock_offset_s: float = DEFAULT_RADAR_CLOCK_OFFSET_S,
    evaluation_window: str = "truth-window",
    radar_clock_delta_s: Sequence[float] = (0.0,),
    position_sources: Sequence[str] = ("fortem-lla",),
    azimuth_conventions: Sequence[str] = ("north-clockwise",),
    elevation_modes: Sequence[str] = ("as-is",),
    radar_origin_lla: str | None = None,
    range_gate_m: float = 800.0,
    max_truth_time_delta_s: float = 2.0,
    top_k: int = 25,
) -> dict[str, Any]:
    """Run per-frame truth-oracle radar-candidate diagnostics and write artifacts.

    ``radar_clock_delta_s`` is applied on top of the base radar clock offset used
    by the normal loader.  No fusion tracker is run here.
    """

    _validate_choices(position_sources, POSITION_SOURCES, "position source")
    _validate_choices(azimuth_conventions, AZIMUTH_CONVENTIONS, "azimuth convention")
    _validate_choices(elevation_modes, ELEVATION_MODES, "elevation mode")

    inputs = load_paper_strict_inputs(
        dataset_root=Path(dataset_root),
        flight_name=flight,
        enu_origin=enu_origin,
        enu_origin_lla=enu_origin_lla,
        lw1_origin_lla=lw1_origin_lla,
        rf_default_std_m=75.0,
        origin_config=origin_config,
        variant=variant,
        rf_clock_offset_s=rf_clock_offset_s,
        radar_clock_offset_s=radar_clock_offset_s,
    )
    if inputs.projector is None:
        raise RuntimeError("paper-strict loader returned no ENU projector")

    window = parse_evaluation_window(evaluation_window)
    radar_origin_enu = (
        _radar_origin_enu(inputs.projector, radar_origin_lla)
        if radar_origin_lla is not None
        else np.zeros(3, dtype=float)
    )
    deltas = _unique_float_list(radar_clock_delta_s)

    summaries: list[dict[str, Any]] = []
    frames_by_config: dict[str, pd.DataFrame] = {}
    current_config_residuals: pd.DataFrame | None = None

    for delta_s in deltas:
        for source in position_sources:
            conventions = azimuth_conventions if source.startswith("polar") else ("not-applicable",)
            modes = elevation_modes if source.startswith("polar") else ("not-applicable",)
            for convention in conventions:
                for elevation_mode in modes:
                    residuals = build_candidate_residual_frame(
                        radar=inputs.radar,
                        truth=inputs.truth,
                        position_source=source,
                        radar_clock_delta_s=delta_s,
                        evaluation_window=window,
                        range_gate_m=range_gate_m,
                        max_truth_time_delta_s=max_truth_time_delta_s,
                        azimuth_convention=convention,
                        elevation_mode=elevation_mode,
                        radar_origin_enu_m=radar_origin_enu,
                    )
                    selected = select_oracle_candidate_per_frame(residuals)
                    config_key = _config_key(source, delta_s, convention, elevation_mode)
                    frames_by_config[config_key] = selected
                    summaries.append(
                        summarize_oracle_selection(
                            selected,
                            residuals,
                            position_source=source,
                            radar_clock_delta_s=delta_s,
                            azimuth_convention=convention,
                            elevation_mode=elevation_mode,
                            range_gate_m=range_gate_m,
                        )
                    )
                    if source == "fortem-lla" and np.isclose(delta_s, 0.0):
                        current_config_residuals = residuals.copy()

    summary_frame = pd.DataFrame.from_records(summaries)
    best_summary = _best_summary_row(summary_frame)
    best_key = _config_key(
        str(best_summary["position_source"]),
        float(best_summary["radar_clock_delta_s"]),
        str(best_summary["azimuth_convention"]),
        str(best_summary["elevation_mode"]),
    )
    best_rows = frames_by_config.get(best_key, pd.DataFrame())
    if current_config_residuals is None:
        current_config_residuals = pd.DataFrame()

    output = Path(output_dir) / inputs.flight_name
    output.mkdir(parents=True, exist_ok=True)
    summary_csv = output / "candidate_oracle_summary.csv"
    by_offset_csv = output / "candidate_oracle_by_time_offset.csv"
    top_configs_csv = output / "candidate_oracle_top_configs.csv"
    best_rows_csv = output / "candidate_oracle_best_rows.csv"
    current_residuals_csv = output / "candidate_residuals_current_config.csv"
    field_sanity_json = output / "radar_field_sanity.json"
    payload_json = output / "radar_candidate_audit_summary.json"

    summary_frame.to_csv(summary_csv, index=False)
    summarize_by_time_offset(summary_frame).to_csv(by_offset_csv, index=False)
    _top_summary_rows(summary_frame, top_k=top_k).to_csv(top_configs_csv, index=False)
    best_rows.to_csv(best_rows_csv, index=False)
    current_config_residuals.to_csv(current_residuals_csv, index=False)
    field_sanity = radar_field_sanity(inputs.radar, evaluation_window=window)
    field_sanity_json.write_text(json.dumps(_jsonable(field_sanity), indent=2), encoding="utf-8")

    payload = {
        "flight": inputs.flight_name,
        "summary_csv": str(summary_csv),
        "by_time_offset_csv": str(by_offset_csv),
        "top_configs_csv": str(top_configs_csv),
        "best_rows_csv": str(best_rows_csv),
        "current_config_residuals_csv": str(current_residuals_csv),
        "field_sanity_json": str(field_sanity_json),
        "summary_json": str(payload_json),
        "best_config": _jsonable(dict(best_summary)),
        "evaluation_window": window.to_record(),
        "range_gate_m": float(range_gate_m),
        "max_truth_time_delta_s": float(max_truth_time_delta_s),
        "base_rf_clock_offset_s": float(rf_clock_offset_s),
        "base_radar_clock_offset_s": float(radar_clock_offset_s),
        "radar_clock_delta_s": [float(value) for value in deltas],
        "position_sources": list(position_sources),
        "azimuth_conventions": list(azimuth_conventions),
        "elevation_modes": list(elevation_modes),
        "radar_origin_enu_m": [float(value) for value in radar_origin_enu],
        "enu_origin_mode": inputs.enu_origin_mode,
        "truth_origin_time": str(inputs.truth_origin_time),
        "file_manifest": inputs.file_manifest,
        "field_sanity": field_sanity,
    }
    payload_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def build_candidate_residual_frame(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    position_source: str,
    radar_clock_delta_s: float,
    evaluation_window: EvaluationWindow,
    range_gate_m: float,
    max_truth_time_delta_s: float,
    azimuth_convention: str = "north-clockwise",
    elevation_mode: str = "as-is",
    radar_origin_enu_m: Iterable[float] | np.ndarray = (0.0, 0.0, 0.0),
) -> pd.DataFrame:
    """Return candidate-level truth residuals under one geometry/time hypothesis."""

    source = _validate_choice(position_source, POSITION_SOURCES, "position source")
    if source.startswith("polar"):
        azimuth_convention = _validate_choice(
            azimuth_convention,
            AZIMUTH_CONVENTIONS,
            "azimuth convention",
        )
        elevation_mode = _validate_choice(elevation_mode, ELEVATION_MODES, "elevation mode")
    else:
        azimuth_convention = "not-applicable"
        elevation_mode = "not-applicable"

    out = radar.copy()
    out["audit_time_s"] = pd.to_numeric(out["time_s"], errors="coerce") + float(
        radar_clock_delta_s
    )
    out = out.loc[evaluation_window.contains(out["audit_time_s"].to_numpy(dtype=float))].copy()
    if out.empty:
        return _empty_residual_frame(
            out,
            source,
            radar_clock_delta_s,
            azimuth_convention,
            elevation_mode,
        )

    candidate_positions = _candidate_positions(
        out,
        position_source=source,
        radar_origin_enu_m=radar_origin_enu_m,
        azimuth_convention=azimuth_convention,
        elevation_mode=elevation_mode,
    )
    truth_positions, valid = interpolate_positions_at_times(
        truth["time_s"].to_numpy(dtype=float),
        truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        out["audit_time_s"].to_numpy(dtype=float),
        max_time_delta_s=max_truth_time_delta_s,
    )
    finite = (
        valid
        & np.isfinite(candidate_positions).all(axis=1)
        & np.isfinite(truth_positions).all(axis=1)
    )
    residual = candidate_positions - truth_positions
    error_2d = np.linalg.norm(residual[:, :2], axis=1)
    error_3d = np.linalg.norm(residual, axis=1)

    out["position_source"] = source
    out["radar_clock_delta_s"] = float(radar_clock_delta_s)
    out["azimuth_convention"] = azimuth_convention
    out["elevation_mode"] = elevation_mode
    out["truth_match_valid"] = finite
    out["candidate_east_m"] = candidate_positions[:, 0]
    out["candidate_north_m"] = candidate_positions[:, 1]
    out["candidate_up_m"] = candidate_positions[:, 2]
    out["truth_east_m"] = truth_positions[:, 0]
    out["truth_north_m"] = truth_positions[:, 1]
    out["truth_up_m"] = truth_positions[:, 2]
    out["residual_east_m"] = residual[:, 0]
    out["residual_north_m"] = residual[:, 1]
    out["residual_up_m"] = residual[:, 2]
    out["error_2d_m"] = np.where(finite, error_2d, np.nan)
    out["error_3d_m"] = np.where(finite, error_3d, np.nan)
    out["abs_z_error_m"] = np.where(finite, np.abs(residual[:, 2]), np.nan)
    out["range_gate_m"] = float(range_gate_m)
    if "range_m" in out.columns:
        ranges = pd.to_numeric(out["range_m"], errors="coerce")
        out["range_le_gate"] = ranges <= float(range_gate_m)
    else:
        out["range_le_gate"] = False
    out["frame_key"] = _frame_key(out)
    return out


def select_oracle_candidate_per_frame(residuals: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest-3D-error candidate in each radar frame."""

    if residuals.empty:
        return residuals.copy()
    valid = residuals.loc[residuals["truth_match_valid"].fillna(False).astype(bool)].copy()
    if valid.empty:
        return valid
    sort_columns = ["frame_key", "error_3d_m", "error_2d_m"]
    ascending = [True, True, True]
    if "cat_prob_uav" in valid.columns:
        valid["_sort_cat_prob_uav"] = pd.to_numeric(valid["cat_prob_uav"], errors="coerce")
        sort_columns.append("_sort_cat_prob_uav")
        ascending.append(False)
    for column in ("track_id", "track_index"):
        if column in valid.columns:
            sort_columns.append(column)
            ascending.append(True)
    selected = valid.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    selected = selected.groupby("frame_key", sort=False, dropna=False).head(1).copy()
    selected["oracle_selected"] = True
    return selected.drop(columns=["_sort_cat_prob_uav"], errors="ignore").reset_index(drop=True)


def summarize_oracle_selection(
    selected: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    position_source: str,
    radar_clock_delta_s: float,
    azimuth_convention: str,
    elevation_mode: str,
    range_gate_m: float,
) -> dict[str, Any]:
    """Return one compact summary row for a candidate-oracle configuration."""

    summary: dict[str, Any] = {
        "position_source": position_source,
        "radar_clock_delta_s": float(radar_clock_delta_s),
        "azimuth_convention": azimuth_convention,
        "elevation_mode": elevation_mode,
        "range_gate_m": float(range_gate_m),
        "count_candidates": int(len(residuals)),
        "count_valid_candidates": int(residuals["truth_match_valid"].fillna(False).sum())
        if "truth_match_valid" in residuals.columns
        else 0,
        "count_frames": int(selected["frame_key"].nunique()) if "frame_key" in selected else 0,
    }
    summary.update(_error_summary(selected))
    summary.update(_range_gate_summary(selected, residuals, range_gate_m=range_gate_m))
    return summary


def summarize_by_time_offset(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best geometry row for each tested radar clock delta."""

    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for delta_s, group in summary.groupby("radar_clock_delta_s", sort=True):
        best = _best_summary_row(group)
        rows.append(
            {
                "radar_clock_delta_s": float(delta_s),
                "best_position_source": best.get("position_source"),
                "best_azimuth_convention": best.get("azimuth_convention"),
                "best_elevation_mode": best.get("elevation_mode"),
                "best_mean_3d_m": best.get("mean_3d_m"),
                "best_mean_2d_m": best.get("mean_2d_m"),
                "best_mean_abs_z_m": best.get("mean_abs_z_m"),
                "best_count_frames": best.get("count_frames"),
                "best_recall_3d_le_50m": best.get("recall_3d_le_50m"),
            }
        )
    return pd.DataFrame.from_records(rows)


def radar_field_sanity(
    radar: pd.DataFrame,
    *,
    evaluation_window: EvaluationWindow | None = None,
) -> dict[str, Any]:
    """Return schema/count summaries for the raw normalized radar candidates."""

    frame = radar.copy()
    if evaluation_window is not None and "time_s" in frame.columns:
        times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
        keep = evaluation_window.contains(times)
        frame = frame.loc[keep].copy()
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }
    for column in (
        "frame_index",
        "track_id",
        "range_m",
        "azimuth_deg",
        "elevation_deg",
        "cat_prob_uav",
        "confidence",
        "track_age",
        "east_m",
        "north_m",
        "up_m",
        "time_s",
    ):
        if column not in frame.columns:
            summary[f"{column}_present"] = False
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        summary[f"{column}_present"] = True
        summary[f"{column}_finite_fraction"] = (
            float(len(finite) / len(frame)) if len(frame) else 0.0
        )
        if len(finite):
            summary[f"{column}_min"] = float(finite.min())
            summary[f"{column}_p50"] = float(np.percentile(finite, 50.0))
            summary[f"{column}_max"] = float(finite.max())
    if "frame_index" in frame.columns:
        frame_index = pd.to_numeric(frame["frame_index"], errors="coerce")
        summary["unique_frames"] = int(frame_index.nunique())
        counts = frame.groupby("frame_index", dropna=False).size().to_numpy(dtype=float)
        if counts.size:
            summary["candidates_per_frame_mean"] = float(np.mean(counts))
            summary["candidates_per_frame_p50"] = float(np.percentile(counts, 50.0))
            summary["candidates_per_frame_p95"] = float(np.percentile(counts, 95.0))
    return summary


def parse_evaluation_window(value: str) -> EvaluationWindow:
    """Parse a candidate-audit evaluation window string."""

    raw = str(value).strip()
    if raw == "truth-window":
        return EvaluationWindow(mode=raw)
    if raw.startswith("explicit:"):
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError("explicit evaluation windows must be explicit:START_S:END_S")
        start_s = float(parts[1])
        end_s = float(parts[2])
        if not np.isfinite([start_s, end_s]).all() or start_s > end_s:
            raise ValueError("explicit evaluation window start/end must be finite and ordered")
        return EvaluationWindow(mode=raw, start_s=start_s, end_s=end_s)
    raise ValueError(EVALUATION_WINDOW_HELP)


def parse_clock_delta_values(
    values: Sequence[str] | None,
    ranges: Sequence[str] | None,
) -> list[float]:
    """Parse repeated comma-list and START:STOP:STEP clock-delta arguments."""

    parsed: list[float] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(float(part))
    for spec in ranges or []:
        parts = [part.strip() for part in str(spec).split(":")]
        if len(parts) != 3:
            raise ValueError("clock delta ranges must have the form START:STOP:STEP")
        start, stop, step = (float(part) for part in parts)
        if step == 0.0:
            raise ValueError("clock delta range step must be nonzero")
        # Inclusive stop, with a small epsilon for decimal steps.
        count = int(np.floor((stop - start) / step + 1.0e-9)) + 1
        if count <= 0:
            raise ValueError("clock delta range has no samples; check START/STOP/STEP signs")
        parsed.extend(float(start + index * step) for index in range(count))
    return _unique_float_list(parsed or [0.0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-radar-candidate-audit",
        description="diagnostic truth-oracle audit of raw Fortem radar candidate pools",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True, help="flight name or unique substring")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar-candidate-audit"))
    parser.add_argument("--variant", choices=["auto", "original", "rerun"], default="auto")
    parser.add_argument(
        "--enu-origin",
        choices=["truth-first", "lla", "lw1"],
        default="lw1",
        help="same ENU origin semantics as raft-uav-paper-strict",
    )
    parser.add_argument("--enu-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lla")
    parser.add_argument("--lw1-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lw1")
    parser.add_argument("--origin-config", type=Path, default=None)
    parser.add_argument("--rf-clock-offset-s", type=float, default=DEFAULT_RF_CLOCK_OFFSET_S)
    parser.add_argument("--radar-clock-offset-s", type=float, default=DEFAULT_RADAR_CLOCK_OFFSET_S)
    parser.add_argument("--evaluation-window", default="truth-window", help=EVALUATION_WINDOW_HELP)
    parser.add_argument(
        "--radar-clock-delta-s",
        action="append",
        default=None,
        help="extra radar clock delta(s) in seconds, comma-separated; repeatable",
    )
    parser.add_argument(
        "--radar-clock-delta-range",
        action="append",
        default=None,
        help="inclusive START:STOP:STEP extra radar clock delta sweep; repeatable",
    )
    parser.add_argument(
        "--position-source",
        action="append",
        choices=POSITION_SOURCES,
        default=None,
        help="candidate position interpretation; repeatable; defaults to fortem-lla",
    )
    parser.add_argument(
        "--azimuth-convention",
        action="append",
        choices=AZIMUTH_CONVENTIONS,
        default=None,
        help="polar azimuth convention; repeatable for polar position sources",
    )
    parser.add_argument(
        "--elevation-mode",
        action="append",
        choices=ELEVATION_MODES,
        default=None,
        help="polar elevation interpretation; repeatable for polar position sources",
    )
    parser.add_argument(
        "--radar-origin-lla",
        default=None,
        help="LAT,LON,ALT for radar-origin polar mode",
    )
    parser.add_argument("--range-gate-m", type=float, default=800.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args(argv)

    payload = run_radar_candidate_audit(
        dataset_root=args.dataset_root,
        flight=args.flight,
        output_dir=args.output_dir,
        variant=args.variant,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
        rf_clock_offset_s=args.rf_clock_offset_s,
        radar_clock_offset_s=args.radar_clock_offset_s,
        evaluation_window=args.evaluation_window,
        radar_clock_delta_s=parse_clock_delta_values(
            args.radar_clock_delta_s,
            args.radar_clock_delta_range,
        ),
        position_sources=tuple(args.position_source or ["fortem-lla"]),
        azimuth_conventions=tuple(args.azimuth_convention or ["north-clockwise"]),
        elevation_modes=tuple(args.elevation_mode or ["as-is"]),
        radar_origin_lla=args.radar_origin_lla,
        range_gate_m=args.range_gate_m,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        top_k=args.top_k,
    )
    print(f"summary_json={payload['summary_json']}")
    print(f"summary_csv={payload['summary_csv']}")
    print(f"by_time_offset_csv={payload['by_time_offset_csv']}")
    print(f"best_rows_csv={payload['best_rows_csv']}")
    return 0


def _candidate_positions(
    radar: pd.DataFrame,
    *,
    position_source: str,
    radar_origin_enu_m: Iterable[float] | np.ndarray,
    azimuth_convention: str,
    elevation_mode: str,
) -> np.ndarray:
    if position_source == "fortem-lla":
        return radar[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    required = {"range_m", "azimuth_deg", "elevation_deg"}
    missing = sorted(required.difference(radar.columns))
    if missing:
        raise KeyError(f"polar candidate audit missing required columns: {missing}")
    elevation = pd.to_numeric(radar["elevation_deg"], errors="coerce").to_numpy(dtype=float)
    if elevation_mode == "inverted":
        elevation = -elevation
    elif elevation_mode == "zero":
        elevation = np.zeros_like(elevation)
    origin = np.zeros(3, dtype=float) if position_source == "polar-from-lw1" else np.asarray(
        tuple(radar_origin_enu_m),
        dtype=float,
    ).reshape(3)
    offset = polar_offset_enu(
        pd.to_numeric(radar["range_m"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(radar["azimuth_deg"], errors="coerce").to_numpy(dtype=float),
        elevation,
        azimuth_convention=azimuth_convention,
    )
    return offset + origin.reshape(1, 3)


def _error_summary(selected: pd.DataFrame) -> dict[str, float | int | None]:
    if selected.empty:
        return {
            "mean_3d_m": None,
            "mean_2d_m": None,
            "mean_abs_z_m": None,
            "median_3d_m": None,
            "p95_3d_m": None,
            "max_3d_m": None,
            "recall_3d_le_25m": None,
            "recall_3d_le_50m": None,
            "recall_3d_le_100m": None,
            "recall_3d_le_200m": None,
            "median_east_residual_m": None,
            "median_north_residual_m": None,
            "median_up_residual_m": None,
        }
    err3 = _finite(selected["error_3d_m"])
    err2 = _finite(selected["error_2d_m"])
    z = _finite(selected["abs_z_error_m"])
    return {
        "mean_3d_m": _mean(err3),
        "mean_2d_m": _mean(err2),
        "mean_abs_z_m": _mean(z),
        "median_3d_m": _percentile(err3, 50.0),
        "p95_3d_m": _percentile(err3, 95.0),
        "max_3d_m": float(np.max(err3)) if err3.size else None,
        "recall_3d_le_25m": _recall(err3, 25.0),
        "recall_3d_le_50m": _recall(err3, 50.0),
        "recall_3d_le_100m": _recall(err3, 100.0),
        "recall_3d_le_200m": _recall(err3, 200.0),
        "median_east_residual_m": _percentile(_finite(selected["residual_east_m"]), 50.0),
        "median_north_residual_m": _percentile(_finite(selected["residual_north_m"]), 50.0),
        "median_up_residual_m": _percentile(_finite(selected["residual_up_m"]), 50.0),
    }


def _range_gate_summary(
    selected: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    range_gate_m: float,
) -> dict[str, int | float | None]:
    if (
        residuals.empty
        or "range_m" not in residuals.columns
        or "frame_key" not in residuals.columns
    ):
        return {
            "frames_oracle_range_le_gate": None,
            "frames_highest_catprob_range_le_gate": None,
            "frames_any_range_le_gate": None,
            "frames_all_candidates_range_le_gate": None,
        }
    gate = float(range_gate_m)
    frame_groups = residuals.groupby("frame_key", dropna=False, sort=False)
    any_le = int(
        frame_groups["range_m"]
        .apply(lambda col: (pd.to_numeric(col, errors="coerce") <= gate).any())
        .sum()
    )
    all_le = int(
        frame_groups["range_m"]
        .apply(lambda col: (pd.to_numeric(col, errors="coerce") <= gate).all())
        .sum()
    )
    oracle_le = (
        int(
            (
                pd.to_numeric(
                    selected.get("range_m", pd.Series(dtype=float)),
                    errors="coerce",
                )
                <= gate
            ).sum()
        )
        if not selected.empty
        else 0
    )
    highest_catprob_le = _highest_catprob_range_gate_count(residuals, gate)
    frames = int(residuals["frame_key"].nunique())
    return {
        "frames_oracle_range_le_gate": oracle_le,
        "frames_highest_catprob_range_le_gate": highest_catprob_le,
        "frames_any_range_le_gate": any_le,
        "frames_all_candidates_range_le_gate": all_le,
        "fraction_oracle_range_le_gate": _safe_ratio(oracle_le, frames),
        "fraction_any_range_le_gate": _safe_ratio(any_le, frames),
    }


def _highest_catprob_range_gate_count(residuals: pd.DataFrame, gate_m: float) -> int | None:
    if "cat_prob_uav" not in residuals.columns or "range_m" not in residuals.columns:
        return None
    frame = residuals.copy()
    frame["_cat_prob_sort"] = pd.to_numeric(frame["cat_prob_uav"], errors="coerce")
    frame["_range_sort"] = pd.to_numeric(frame["range_m"], errors="coerce")
    ranked = frame.sort_values(["frame_key", "_cat_prob_sort"], ascending=[True, False])
    selected = ranked.groupby("frame_key", dropna=False, sort=False).head(1)
    return int((selected["_range_sort"] <= float(gate_m)).sum())


def _top_summary_rows(summary: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    frame = summary.copy()
    frame["_sort_mean_3d"] = pd.to_numeric(frame["mean_3d_m"], errors="coerce")
    frame["_sort_count_frames"] = pd.to_numeric(frame["count_frames"], errors="coerce")
    frame = frame.sort_values(["_sort_mean_3d", "_sort_count_frames"], ascending=[True, False])
    return frame.drop(columns=["_sort_mean_3d", "_sort_count_frames"]).head(int(top_k))


def _best_summary_row(summary: pd.DataFrame) -> pd.Series:
    if summary.empty:
        raise ValueError("candidate audit produced no summary rows")
    frame = summary.copy()
    frame["_sort_mean_3d"] = pd.to_numeric(frame["mean_3d_m"], errors="coerce")
    frame["_sort_count_frames"] = pd.to_numeric(frame["count_frames"], errors="coerce")
    frame = frame.sort_values(["_sort_mean_3d", "_sort_count_frames"], ascending=[True, False])
    return frame.iloc[0]


def _radar_origin_enu(projector: Any, radar_origin_lla: str | None) -> np.ndarray:
    if radar_origin_lla is None:
        return np.zeros(3, dtype=float)
    lat, lon, alt = _parse_lla(radar_origin_lla)
    return np.asarray(projector.transform(lat, lon, alt), dtype=float).reshape(3)


def _frame_key(frame: pd.DataFrame) -> np.ndarray:
    if "frame_index" in frame.columns:
        return pd.to_numeric(frame["frame_index"], errors="coerce").to_numpy(dtype=float)
    return pd.to_numeric(frame["audit_time_s"], errors="coerce").to_numpy(dtype=float)


def _empty_residual_frame(
    frame: pd.DataFrame,
    position_source: str,
    radar_clock_delta_s: float,
    azimuth_convention: str,
    elevation_mode: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["position_source"] = position_source
    out["radar_clock_delta_s"] = float(radar_clock_delta_s)
    out["azimuth_convention"] = azimuth_convention
    out["elevation_mode"] = elevation_mode
    out["truth_match_valid"] = False
    out["frame_key"] = np.empty(0, dtype=float)
    return out


def _config_key(
    position_source: str,
    radar_clock_delta_s: float,
    azimuth_convention: str,
    elevation_mode: str,
) -> str:
    return (
        f"{position_source}|{float(radar_clock_delta_s):.9g}|"
        f"{azimuth_convention}|{elevation_mode}"
    )


def _validate_choices(values: Sequence[str], allowed: Sequence[str], label: str) -> None:
    for value in values:
        _validate_choice(value, allowed, label)


def _validate_choice(value: str, allowed: Sequence[str], label: str) -> str:
    parsed = str(value).strip()
    if parsed not in allowed:
        raise ValueError(f"{label} must be one of {tuple(allowed)}, got {value!r}")
    return parsed


def _parse_lla(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError("LLA must have the form LAT,LON,ALT")
    lat, lon, alt = (float(part) for part in parts)
    if not np.isfinite([lat, lon, alt]).all():
        raise ValueError("LLA values must be finite")
    return lat, lon, alt


def _unique_float_list(values: Sequence[float]) -> list[float]:
    unique = sorted({round(float(value), 9) for value in values})
    return [float(value) for value in unique]


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _recall(values: np.ndarray, threshold_m: float) -> float | None:
    return float(np.mean(values <= float(threshold_m))) if values.size else None


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(inner) for inner in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
