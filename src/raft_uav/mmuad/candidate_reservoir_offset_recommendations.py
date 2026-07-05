"""Recommend branch/source reservoir score offsets from assignment diagnostics.

The branch-preserving reservoir and offset-grid tools make it easy to keep
multiple MMUAD candidate streams alive, but they still require choosing which
branches/sources to promote before mixture-MAP.  This diagnostic consumes the
truth-backed frame rows from ``raft-uav-mmuad-candidate-assignment-diagnostics``
and turns recurring assignment failures into bounded additive offset suggestions.

It is deliberately a train/validation diagnostic: the output can seed train-CV
or public-val experiments, but hidden-test inference should consume only frozen
settings selected without hidden truth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

RECOMMENDATIONS_CSV = "mmuad_candidate_reservoir_offset_recommendations.csv"
RECOMMENDATIONS_JSON = "mmuad_candidate_reservoir_offset_recommendations.json"
RECOMMENDATIONS_CLI_TXT = "mmuad_candidate_reservoir_offset_cli.txt"
PROMOTABLE_FAILURE_MODES = (
    "good_candidate_buried",
    "wrong_dominant_assignment",
    "smoothing_assignment_gap",
)


@dataclass(frozen=True)
class ReservoirOffsetRecommendationConfig:
    """Configuration for assignment-derived reservoir offset suggestions."""

    max_abs_offset: float = 1.0
    min_frame_count: int = 1
    failure_modes: tuple[str, ...] = PROMOTABLE_FAILURE_MODES
    regret_column: str = "state_regret_m"
    fallback_regret_columns: tuple[str, ...] = ("dominant_regret_m", "weighted_regret_m")


def build_reservoir_offset_recommendations(
    frame_rows: pd.DataFrame,
    *,
    config: ReservoirOffsetRecommendationConfig | None = None,
) -> pd.DataFrame:
    """Return branch/source offset recommendations from assignment diagnostics."""

    config = config or ReservoirOffsetRecommendationConfig()
    rows = _normalized_rows(frame_rows, config=config)
    if rows.empty:
        return pd.DataFrame(columns=_output_columns())
    records: list[dict[str, Any]] = []
    for label_type, oracle_column, dominant_column in (
        ("branch", "oracle_candidate_branch", "dominant_candidate_branch"),
        ("source", "oracle_source", "dominant_source"),
    ):
        records.extend(
            _recommendation_records(
                rows,
                label_type=label_type,
                oracle_column=oracle_column,
                dominant_column=dominant_column,
                config=config,
            )
        )
    out = pd.DataFrame.from_records(records, columns=_output_columns())
    if out.empty:
        return out
    out = out.loc[out["frame_count"] >= int(config.min_frame_count)].copy()
    return out.sort_values(
        ["label_type", "recommended_offset", "frame_count", "label"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def write_reservoir_offset_recommendations(
    *,
    output_dir: Path,
    recommendations: pd.DataFrame,
    config: ReservoirOffsetRecommendationConfig,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write recommendation CSV, JSON, and CLI snippet artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "recommendations_csv": output / RECOMMENDATIONS_CSV,
        "recommendations_json": output / RECOMMENDATIONS_JSON,
        "recommendations_cli_txt": output / RECOMMENDATIONS_CLI_TXT,
    }
    recommendations.to_csv(paths["recommendations_csv"], index=False)
    cli_lines = _cli_lines(recommendations)
    paths["recommendations_cli_txt"].write_text("\n".join(cli_lines) + "\n", encoding="utf-8")
    payload = dict(provenance or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-reservoir-offset-recommendations-v1",
            "config": asdict(config),
            "recommendation_count": int(len(recommendations)),
            "cli_lines": cli_lines,
            "recommendations": recommendations.to_dict(orient="records"),
        }
    )
    paths["recommendations_json"].write_text(
        json.dumps(_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-reservoir-offset-recommendations",
        description="recommend reservoir branch/source score offsets from assignment diagnostics",
    )
    parser.add_argument("--frame-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-abs-offset", type=float, default=1.0)
    parser.add_argument("--min-frame-count", type=int, default=1)
    parser.add_argument("--failure-mode", action="append", default=[])
    parser.add_argument("--regret-column", default="state_regret_m")
    args = parser.parse_args(argv)

    failure_modes = tuple(args.failure_mode) or PROMOTABLE_FAILURE_MODES
    config = ReservoirOffsetRecommendationConfig(
        max_abs_offset=float(args.max_abs_offset),
        min_frame_count=int(args.min_frame_count),
        failure_modes=failure_modes,
        regret_column=str(args.regret_column),
    )
    frame_rows = pd.read_csv(args.frame_csv)
    recommendations = build_reservoir_offset_recommendations(frame_rows, config=config)
    paths = write_reservoir_offset_recommendations(
        output_dir=args.output_dir,
        recommendations=recommendations,
        config=config,
        provenance={"frame_csv": str(args.frame_csv)},
    )
    print("mmuad_candidate_reservoir_offset_recommendations=ok")
    print(f"recommendation_count={len(recommendations)}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _normalized_rows(
    frame_rows: pd.DataFrame,
    *,
    config: ReservoirOffsetRecommendationConfig,
) -> pd.DataFrame:
    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty:
        return rows
    if "assignment_failure_mode" not in rows.columns:
        rows["assignment_failure_mode"] = "unknown"
    for column in (
        "oracle_candidate_branch",
        "dominant_candidate_branch",
        "oracle_source",
        "dominant_source",
    ):
        if column not in rows.columns:
            rows[column] = "unknown"
        rows[column] = _clean_text(rows[column])
    rows["assignment_failure_mode"] = _clean_text(rows["assignment_failure_mode"])
    rows["_recommendation_weight"] = _recommendation_weight(rows, config=config)
    mode_mask = rows["assignment_failure_mode"].isin(set(config.failure_modes))
    useful = mode_mask & (rows["_recommendation_weight"] > 0.0)
    return rows.loc[useful].copy()


def _recommendation_weight(
    rows: pd.DataFrame,
    *,
    config: ReservoirOffsetRecommendationConfig,
) -> pd.Series:
    columns = (config.regret_column, *config.fallback_regret_columns)
    weight = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        weight = weight.where(weight.notna(), values)
    weight = weight.fillna(0.0).clip(lower=0.0)
    return weight


def _recommendation_records(
    rows: pd.DataFrame,
    *,
    label_type: str,
    oracle_column: str,
    dominant_column: str,
    config: ReservoirOffsetRecommendationConfig,
) -> list[dict[str, Any]]:
    promote: dict[str, float] = {}
    demote: dict[str, float] = {}
    frame_counts: dict[str, int] = {}
    for _, row in rows.iterrows():
        weight = float(row["_recommendation_weight"])
        oracle_label = str(row[oracle_column])
        dominant_label = str(row[dominant_column])
        if oracle_label and oracle_label != "unknown":
            promote[oracle_label] = promote.get(oracle_label, 0.0) + weight
            frame_counts[oracle_label] = frame_counts.get(oracle_label, 0) + 1
        if dominant_label and dominant_label not in {"unknown", oracle_label}:
            demote[dominant_label] = demote.get(dominant_label, 0.0) + weight
            frame_counts[dominant_label] = frame_counts.get(dominant_label, 0) + 1
    labels = sorted(set(promote) | set(demote))
    if not labels:
        return []
    net = {label: promote.get(label, 0.0) - demote.get(label, 0.0) for label in labels}
    max_abs_net = max(max(abs(value) for value in net.values()), 1.0e-12)
    records: list[dict[str, Any]] = []
    for label in labels:
        recommended = float(config.max_abs_offset) * net[label] / max_abs_net
        records.append(
            {
                "label_type": label_type,
                "label": label,
                "recommended_offset": recommended,
                "net_weight": net[label],
                "promote_weight": promote.get(label, 0.0),
                "demote_weight": demote.get(label, 0.0),
                "frame_count": int(frame_counts.get(label, 0)),
                "grid_seed": _grid_seed(recommended),
                "cli_flag": _cli_flag(label_type, label, recommended),
            }
        )
    return records


def _grid_seed(offset: float) -> str:
    values = sorted({0.0, float(offset), 0.5 * float(offset)})
    return ",".join(_format_float(value) for value in values)


def _cli_flag(label_type: str, label: str, offset: float) -> str:
    name = "branch-score-offset-grid" if label_type == "branch" else "source-score-offset-grid"
    return f"--{name} {label}={_grid_seed(offset)}"


def _cli_lines(recommendations: pd.DataFrame) -> list[str]:
    if recommendations.empty:
        return []
    return [str(flag) for flag in recommendations["cli_flag"].dropna().astype(str).tolist()]


def _format_float(value: float) -> str:
    value = float(value)
    if abs(value) < 1.0e-12:
        value = 0.0
    text = f"{value:.6g}"
    return text.replace("-0", "0") if text.startswith("-0") and value == 0.0 else text


def _output_columns() -> list[str]:
    return [
        "label_type",
        "label",
        "recommended_offset",
        "net_weight",
        "promote_weight",
        "demote_weight",
        "frame_count",
        "grid_seed",
        "cli_flag",
    ]


def _clean_text(values: pd.Series) -> pd.Series:
    text = values.where(values.notna(), "unknown").astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, "unknown")


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
