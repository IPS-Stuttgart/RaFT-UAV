"""Truth-backed attribution for MMUAD candidate-mixture assignment failures.

The MMUAD reservoir/mixture experiments now retain many candidate branches and
write per-candidate responsibilities.  This diagnostic module compares those
assignments to local validation truth so we can tell whether a remaining pose
error is caused by a missing candidate, a good candidate that received too little
mixture weight, or a trajectory smoother that failed to recover from a weighted
candidate.  It is diagnostic only: truth is required and this should not be used
for hidden-test inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.schema import normalize_truth_columns

FRAME_CSV = "mmuad_candidate_mixture_assignment_attribution.csv"
SUMMARY_CSV = "mmuad_candidate_mixture_assignment_attribution_summary.csv"
SUMMARY_JSON = "mmuad_candidate_mixture_assignment_attribution_summary.json"


def build_assignment_attribution_tables(
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_truth_time_delta_s: float = 0.5,
    good_candidate_threshold_m: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frame-level attribution rows and sequence/pooled summaries."""

    assignment_rows = _normalize_assignments(assignments)
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if assignment_rows.empty or truth_rows.empty:
        empty = pd.DataFrame()
        return empty, empty
    records: list[dict[str, Any]] = []
    for sequence_id, group in assignment_rows.groupby("sequence_id", sort=True):
        sequence_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == str(sequence_id)]
        if sequence_truth.empty:
            continue
        truth_times = sequence_truth["time_s"].to_numpy(float)
        truth_xyz = sequence_truth[["x_m", "y_m", "z_m"]].to_numpy(float)
        order = np.argsort(truth_times)
        truth_times = truth_times[order]
        truth_xyz = truth_xyz[order]
        for time_s, frame in group.groupby("time_s", sort=True):
            match = _nearest_truth(float(time_s), truth_times, truth_xyz)
            if match is None:
                continue
            truth_time, truth_position, truth_delta = match
            if abs(truth_delta) > float(max_truth_time_delta_s):
                continue
            frame = frame.sort_values("candidate_rank").reset_index(drop=True)
            records.append(
                _attribution_record(
                    sequence_id=str(sequence_id),
                    time_s=float(time_s),
                    truth_time_s=float(truth_time),
                    truth_time_delta_s=float(truth_delta),
                    truth_xyz=truth_position,
                    frame=frame,
                    good_candidate_threshold_m=float(good_candidate_threshold_m),
                )
            )
    frame_rows = pd.DataFrame.from_records(records)
    summary = _summary_table(frame_rows)
    return frame_rows, summary


def write_assignment_attribution_outputs(
    *,
    output_dir: Path,
    frame_rows: pd.DataFrame,
    summary: pd.DataFrame,
    provenance: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write attribution artifacts and return path strings."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_csv": output / FRAME_CSV,
        "summary_csv": output / SUMMARY_CSV,
        "summary_json": output / SUMMARY_JSON,
    }
    frame_rows.to_csv(paths["frame_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    payload = dict(provenance or {})
    payload.update(
        {
            "frame_count": int(len(frame_rows)),
            "summary": summary.to_dict(orient="records"),
        }
    )
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-mixture-assignment-attribution",
        description="attribute local MMUAD candidate-mixture assignment failures against truth",
    )
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--good-candidate-threshold-m", type=float, default=5.0)
    args = parser.parse_args(argv)

    assignments = pd.read_csv(args.assignments_csv)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    frame_rows, summary = build_assignment_attribution_tables(
        assignments,
        truth,
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
        good_candidate_threshold_m=float(args.good_candidate_threshold_m),
    )
    paths = write_assignment_attribution_outputs(
        output_dir=args.output_dir,
        frame_rows=frame_rows,
        summary=summary,
        provenance={
            "assignments_csv": str(args.assignments_csv),
            "truth_csv": str(args.truth_csv),
            "max_truth_time_delta_s": float(args.max_truth_time_delta_s),
            "good_candidate_threshold_m": float(args.good_candidate_threshold_m),
        },
    )
    pooled = summary.loc[summary["sequence_id"] == "__pooled__"] if not summary.empty else summary
    print("mmuad_candidate_mixture_assignment_attribution=ok")
    print(f"frame_count={len(frame_rows)}")
    if not pooled.empty:
        row = pooled.iloc[0]
        print(f"state_mse_3d={row.get('state_3d_m_mse')}")
        print(f"oracle_mse_3d={row.get('oracle_3d_m_mse')}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _normalize_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(assignments).copy()
    required = {
        "sequence_id",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "state_x_m",
        "state_y_m",
        "state_z_m",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"assignment rows missing required columns: {missing}")
    if "candidate_rank" not in rows.columns:
        rows["candidate_rank"] = rows.groupby(["sequence_id", "time_s"]).cumcount() + 1
    if "mixture_final_weight" not in rows.columns:
        rows["mixture_final_weight"] = 1.0
    if "mixture_dominant" not in rows.columns:
        rows["mixture_dominant"] = False
    for column in (
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "candidate_rank",
        "mixture_final_weight",
    ):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["mixture_dominant"] = _bool_series(rows["mixture_dominant"])
    finite_columns = [
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "state_x_m",
        "state_y_m",
        "state_z_m",
    ]
    rows = rows.loc[np.isfinite(rows[finite_columns].to_numpy(float)).all(axis=1)].copy()
    return rows.sort_values(["sequence_id", "time_s", "candidate_rank"]).reset_index(drop=True)


def _nearest_truth(
    time_s: float,
    truth_times: np.ndarray,
    truth_xyz: np.ndarray,
) -> tuple[float, np.ndarray, float] | None:
    if len(truth_times) == 0:
        return None
    index = int(np.argmin(np.abs(truth_times - float(time_s))))
    truth_time = float(truth_times[index])
    return truth_time, truth_xyz[index], float(time_s - truth_time)


def _attribution_record(
    *,
    sequence_id: str,
    time_s: float,
    truth_time_s: float,
    truth_time_delta_s: float,
    truth_xyz: np.ndarray,
    frame: pd.DataFrame,
    good_candidate_threshold_m: float,
) -> dict[str, Any]:
    candidate_xyz = frame[["x_m", "y_m", "z_m"]].to_numpy(float)
    weights = pd.to_numeric(frame["mixture_final_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    if float(weights.sum()) <= 1.0e-12:
        weights = np.ones(len(frame), dtype=float) / float(len(frame))
    else:
        weights = weights / float(weights.sum())
    candidate_errors = np.linalg.norm(candidate_xyz - truth_xyz.reshape(1, 3), axis=1)
    oracle_index = int(np.argmin(candidate_errors))
    dominant_mask = _bool_series(frame["mixture_dominant"])
    dominant_index = int(np.where(dominant_mask.to_numpy())[0][0]) if dominant_mask.any() else int(np.argmax(weights))
    state_xyz = frame[["state_x_m", "state_y_m", "state_z_m"]].iloc[0].to_numpy(float)
    pseudo_xyz = np.sum(candidate_xyz * weights[:, None], axis=0)
    good_mask = candidate_errors <= float(good_candidate_threshold_m)
    good_weight_mass = float(weights[good_mask].sum())
    state_error = float(np.linalg.norm(state_xyz - truth_xyz))
    oracle_error = float(candidate_errors[oracle_index])
    dominant_error = float(candidate_errors[dominant_index])
    pseudo_error = float(np.linalg.norm(pseudo_xyz - truth_xyz))
    return {
        "sequence_id": sequence_id,
        "time_s": float(time_s),
        "truth_time_s": truth_time_s,
        "truth_time_delta_s": truth_time_delta_s,
        "candidate_count": int(len(frame)),
        "state_3d_m": state_error,
        "pseudo_measurement_3d_m": pseudo_error,
        "dominant_candidate_3d_m": dominant_error,
        "oracle_candidate_3d_m": oracle_error,
        "state_minus_oracle_3d_m": float(state_error - oracle_error),
        "dominant_minus_oracle_3d_m": float(dominant_error - oracle_error),
        "pseudo_minus_oracle_3d_m": float(pseudo_error - oracle_error),
        "oracle_candidate_rank": int(frame["candidate_rank"].iloc[oracle_index]),
        "oracle_candidate_weight": float(weights[oracle_index]),
        "oracle_candidate_source": _value(frame, oracle_index, "source"),
        "oracle_candidate_branch": _value(frame, oracle_index, "candidate_branch"),
        "dominant_candidate_rank": int(frame["candidate_rank"].iloc[dominant_index]),
        "dominant_candidate_weight": float(weights[dominant_index]),
        "dominant_candidate_source": _value(frame, dominant_index, "source"),
        "dominant_candidate_branch": _value(frame, dominant_index, "candidate_branch"),
        "dominant_is_oracle": bool(dominant_index == oracle_index),
        "good_candidate_available": bool(good_mask.any()),
        "good_candidate_weight_mass": good_weight_mass,
        "assignment_failure_mode": _failure_mode(
            good_candidate_available=bool(good_mask.any()),
            dominant_error=dominant_error,
            state_error=state_error,
            good_weight_mass=good_weight_mass,
            threshold=float(good_candidate_threshold_m),
        ),
    }


def _failure_mode(
    *,
    good_candidate_available: bool,
    dominant_error: float,
    state_error: float,
    good_weight_mass: float,
    threshold: float,
) -> str:
    if not good_candidate_available:
        return "missing_good_candidate"
    if dominant_error <= threshold:
        return "dominant_good_candidate"
    if state_error <= threshold:
        return "smoother_recovered"
    if good_weight_mass < 0.05:
        return "good_candidate_unweighted"
    return "good_candidate_weighted_but_not_selected"


def _summary_table(frame_rows: pd.DataFrame) -> pd.DataFrame:
    if frame_rows.empty:
        return pd.DataFrame()
    records = [_summarize_group(frame_rows, sequence_id="__pooled__")]
    for sequence_id, group in frame_rows.groupby("sequence_id", sort=True):
        records.append(_summarize_group(group, sequence_id=str(sequence_id)))
    return pd.DataFrame.from_records(records)


def _summarize_group(group: pd.DataFrame, *, sequence_id: str) -> dict[str, Any]:
    state = pd.to_numeric(group["state_3d_m"], errors="coerce")
    oracle = pd.to_numeric(group["oracle_candidate_3d_m"], errors="coerce")
    dominant = pd.to_numeric(group["dominant_candidate_3d_m"], errors="coerce")
    pseudo = pd.to_numeric(group["pseudo_measurement_3d_m"], errors="coerce")
    return {
        "sequence_id": sequence_id,
        "frame_count": int(len(group)),
        "state_3d_m_mean": _mean(state),
        "state_3d_m_mse": _mse(state),
        "state_3d_m_p95": _quantile(state, 0.95),
        "oracle_3d_m_mean": _mean(oracle),
        "oracle_3d_m_mse": _mse(oracle),
        "oracle_3d_m_p95": _quantile(oracle, 0.95),
        "dominant_3d_m_mse": _mse(dominant),
        "pseudo_measurement_3d_m_mse": _mse(pseudo),
        "state_minus_oracle_3d_m_mean": _mean(group["state_minus_oracle_3d_m"]),
        "dominant_minus_oracle_3d_m_mean": _mean(group["dominant_minus_oracle_3d_m"]),
        "good_candidate_available_fraction": _fraction(group["good_candidate_available"]),
        "dominant_is_oracle_fraction": _fraction(group["dominant_is_oracle"]),
        "good_candidate_weight_mass_mean": _mean(group["good_candidate_weight_mass"]),
        "oracle_candidate_rank_p50": _quantile(group["oracle_candidate_rank"], 0.50),
        "oracle_candidate_rank_p95": _quantile(group["oracle_candidate_rank"], 0.95),
        "dominant_failure_mode": _mode(group["assignment_failure_mode"]),
    }


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.fillna(False).astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _value(frame: pd.DataFrame, index: int, column: str) -> str:
    if column not in frame.columns:
        return ""
    value = frame[column].iloc[index]
    return "" if pd.isna(value) else str(value)


def _mean(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(series.mean()) if not series.empty else float("nan")


def _mse(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.mean(np.square(series.to_numpy(float)))) if not series.empty else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(series.quantile(quantile)) if not series.empty else float("nan")


def _fraction(values: pd.Series) -> float:
    series = _bool_series(values)
    return float(series.mean()) if not series.empty else float("nan")


def _mode(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    return str(cleaned.value_counts().index[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
