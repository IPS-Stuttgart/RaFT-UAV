"""Prioritize MMUAD candidate-assignment failures into next experiments.

The assignment diagnostics separate frames into modes such as buried oracle
candidates, wrong dominant assignments, and smoothing gaps.  This module turns
contiguous assignment-failure blocks into a compact action plan so reservoir and
mixture-MAP runs can be triaged without manually inspecting long CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ACTION_BY_MODE = {
    "missing_good_candidate_in_assignments": "expand_candidate_pool_or_reservoir",
    "good_candidate_buried": "increase_reservoir_recall_or_balance",
    "wrong_dominant_assignment": "retune_candidate_weights_or_sigma",
    "smoothing_assignment_gap": "retune_mixture_smoother",
    "covered": "keep_current_settings",
}

_METHOD_BY_MODE = {
    "missing_good_candidate_in_assignments": (
        "preserve raw/dynamic/source-translated branches, relax pruning, or add extraction branches"
    ),
    "good_candidate_buried": (
        "increase per-branch/per-source quotas, add score offsets, or use branch/source balancing"
    ),
    "wrong_dominant_assignment": (
        "retrain or calibrate learned sigma, reduce score dominance, and inspect branch/source priors"
    ),
    "smoothing_assignment_gap": (
        "adjust Huber/smoothness/responsibility floors or initialize from a stronger trajectory"
    ),
    "covered": "do not tune this mode first; prioritize uncovered failure modes",
}

ACTION_PLAN_CSV = "mmuad_candidate_assignment_action_plan.csv"
ACTION_SUMMARY_CSV = "mmuad_candidate_assignment_action_summary.csv"
ACTION_SUMMARY_JSON = "mmuad_candidate_assignment_action_summary.json"


def build_candidate_assignment_action_plan(
    blocks: pd.DataFrame,
    *,
    top_n_blocks: int = 20,
    duration_weight: float = 1.0,
    frame_weight: float = 1.0,
    error_weight: float = 1.0,
    regret_weight: float = 1.0,
    buried_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prioritized assignment-block action rows and an action summary."""

    rows = pd.DataFrame(blocks).copy()
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    _require_columns(rows, ["sequence_id", "assignment_failure_mode", "frame_count"])
    rows["frame_count"] = _numeric(rows, "frame_count", default=0.0)
    rows["duration_s"] = _numeric(rows, "duration_s", default=0.0)
    rows["state_error_3d_m_max"] = _numeric(rows, "state_error_3d_m_max", default=0.0)
    rows["state_regret_m_p95"] = _numeric(rows, "state_regret_m_p95", default=0.0)
    rows["oracle_in_topk_by_weight_rate"] = _numeric(
        rows,
        "oracle_in_topk_by_weight_rate",
        default=0.0,
    )
    rows["dominant_matches_oracle_rate"] = _numeric(
        rows,
        "dominant_matches_oracle_rate",
        default=0.0,
    )
    rows["recommended_action"] = rows["assignment_failure_mode"].map(_ACTION_BY_MODE).fillna(
        "inspect_assignment_block",
    )
    rows["recommended_method"] = rows["assignment_failure_mode"].map(_METHOD_BY_MODE).fillna(
        "inspect assignment diagnostics, branch summaries, and candidate rows",
    )
    rows["assignment_action_priority_score"] = _priority_score(
        rows,
        duration_weight=duration_weight,
        frame_weight=frame_weight,
        error_weight=error_weight,
        regret_weight=regret_weight,
        buried_weight=buried_weight,
    )
    rows = rows.sort_values(
        ["assignment_action_priority_score", "frame_count", "duration_s"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    action_rows = rows.head(int(top_n_blocks)).copy() if top_n_blocks > 0 else rows.copy()
    action_rows["action_rank"] = np.arange(1, len(action_rows) + 1, dtype=int)
    summary = _action_summary(rows)
    return action_rows, summary


def write_candidate_assignment_action_plan_outputs(
    *,
    output_dir: Path,
    action_rows: pd.DataFrame,
    action_summary: pd.DataFrame,
) -> dict[str, Path]:
    """Write candidate-assignment action-plan artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "action_rows_csv": output / ACTION_PLAN_CSV,
        "action_summary_csv": output / ACTION_SUMMARY_CSV,
        "action_summary_json": output / ACTION_SUMMARY_JSON,
    }
    action_rows.to_csv(paths["action_rows_csv"], index=False)
    action_summary.to_csv(paths["action_summary_csv"], index=False)
    payload = {
        "schema": "raft-uav-mmuad-candidate-assignment-action-plan-v1",
        "action_count": int(len(action_rows)),
        "summary_row_count": int(len(action_summary)),
        "actions": action_rows.to_dict(orient="records"),
        "summary": action_summary.to_dict(orient="records"),
    }
    paths["action_summary_json"].write_text(
        json.dumps(_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-assignment-action-plan",
        description="prioritize MMUAD candidate-assignment failure blocks into experiment actions",
    )
    parser.add_argument("--blocks-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n-blocks", type=int, default=20)
    parser.add_argument("--duration-weight", type=float, default=1.0)
    parser.add_argument("--frame-weight", type=float, default=1.0)
    parser.add_argument("--error-weight", type=float, default=1.0)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--buried-weight", type=float, default=0.5)
    args = parser.parse_args(argv)

    blocks = pd.read_csv(args.blocks_csv)
    action_rows, action_summary = build_candidate_assignment_action_plan(
        blocks,
        top_n_blocks=int(args.top_n_blocks),
        duration_weight=float(args.duration_weight),
        frame_weight=float(args.frame_weight),
        error_weight=float(args.error_weight),
        regret_weight=float(args.regret_weight),
        buried_weight=float(args.buried_weight),
    )
    paths = write_candidate_assignment_action_plan_outputs(
        output_dir=args.output_dir,
        action_rows=action_rows,
        action_summary=action_summary,
    )
    print("mmuad_candidate_assignment_action_plan=ok")
    print(f"action_rows={len(action_rows)}")
    print(f"action_summary_rows={len(action_summary)}")
    if not action_summary.empty:
        first = action_summary.iloc[0]
        print(f"top_recommended_action={first['recommended_action']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _priority_score(
    rows: pd.DataFrame,
    *,
    duration_weight: float,
    frame_weight: float,
    error_weight: float,
    regret_weight: float,
    buried_weight: float,
) -> pd.Series:
    missing_topk_rate = 1.0 - rows["oracle_in_topk_by_weight_rate"].clip(0.0, 1.0)
    wrong_dominant_rate = 1.0 - rows["dominant_matches_oracle_rate"].clip(0.0, 1.0)
    buried = 0.5 * (missing_topk_rate + wrong_dominant_rate)
    return (
        float(duration_weight) * _normalize(rows["duration_s"])
        + float(frame_weight) * _normalize(rows["frame_count"])
        + float(error_weight) * _normalize(rows["state_error_3d_m_max"])
        + float(regret_weight) * _normalize(rows["state_regret_m_p95"])
        + float(buried_weight) * buried
    )


def _action_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for action, group in rows.groupby("recommended_action", sort=False):
        records.append(
            {
                "recommended_action": str(action),
                "recommended_method": str(group["recommended_method"].iloc[0]),
                "block_count": int(len(group)),
                "frame_count": int(group["frame_count"].sum()),
                "duration_s_sum": float(group["duration_s"].sum()),
                "state_error_3d_m_max": _max(group["state_error_3d_m_max"]),
                "state_regret_m_p95": _quantile(group["state_regret_m_p95"], 0.95),
                "oracle_in_topk_by_weight_rate_mean": _mean(
                    group["oracle_in_topk_by_weight_rate"],
                ),
                "dominant_matches_oracle_rate_mean": _mean(
                    group["dominant_matches_oracle_rate"],
                ),
                "priority_score_sum": float(group["assignment_action_priority_score"].sum()),
            },
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values(
        ["priority_score_sum", "frame_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _require_columns(rows: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"assignment block rows missing required columns: {missing}")


def _numeric(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=rows.index, dtype=float)


def _normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    max_value = float(values.max()) if len(values) else 0.0
    if not np.isfinite(max_value) or max_value <= 0.0:
        return pd.Series(0.0, index=values.index, dtype=float)
    return values / max_value


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _max(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(quantile)) if not values.empty else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
