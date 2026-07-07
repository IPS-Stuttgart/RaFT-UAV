"""Compare multiple MMUAD Track 5 scorecard JSON files.

The local Track 5 workflow now produces many candidate ZIPs/scorecards. This
helper turns a set of scorecard JSON files into one ranked table and an optional
pairwise delta table so public-validation diagnostics can be reviewed without
hand-copying metrics from JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_POSE_REFERENCE_MSE = 56.88
DEFAULT_TOP3_REFERENCE_MSE = 24.51
DEFAULT_CLASS_REFERENCE_ACCURACY = 0.322


def compare_track5_scorecards(
    scorecard_paths: Iterable[Path],
    *,
    pose_reference_mse: float = DEFAULT_POSE_REFERENCE_MSE,
    top3_reference_mse: float = DEFAULT_TOP3_REFERENCE_MSE,
    class_reference_accuracy: float = DEFAULT_CLASS_REFERENCE_ACCURACY,
) -> pd.DataFrame:
    """Return a ranked table for Track 5 local scorecard JSON files."""

    records = [
        _scorecard_record(
            Path(path),
            pose_reference_mse=pose_reference_mse,
            top3_reference_mse=top3_reference_mse,
            class_reference_accuracy=class_reference_accuracy,
        )
        for path in scorecard_paths
    ]
    if not records:
        raise ValueError("at least one scorecard path is required")
    table = pd.DataFrame.from_records(records)
    for column in (
        "pose_mse_loss_m2",
        "public_rmse_3d_m",
        "public_p95_3d_m",
        "public_max_3d_m",
        "uav_type_accuracy",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values(
        ["pose_mse_loss_m2", "public_p95_3d_m", "public_max_3d_m", "scorecard_label"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=int))
    best_mse = table["pose_mse_loss_m2"].iloc[0]
    table["pose_mse_delta_to_best"] = table["pose_mse_loss_m2"] - best_mse
    table["pose_mse_ratio_to_best"] = table["pose_mse_loss_m2"] / best_mse
    return table


def build_pairwise_scorecard_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise metric deltas against the ranked best row."""

    table = pd.DataFrame(comparison).copy()
    if table.empty:
        return pd.DataFrame(columns=_pairwise_columns())
    best = table.sort_values("rank").iloc[0]
    records: list[dict[str, Any]] = []
    for _, row in table.sort_values("rank").iterrows():
        records.append(
            {
                "baseline_label": str(best["scorecard_label"]),
                "comparison_label": str(row["scorecard_label"]),
                "comparison_rank": int(row["rank"]),
                "pose_mse_delta": _numeric(row.get("pose_mse_loss_m2"))
                - _numeric(best.get("pose_mse_loss_m2")),
                "rmse_delta_m": _numeric(row.get("public_rmse_3d_m"))
                - _numeric(best.get("public_rmse_3d_m")),
                "p95_delta_m": _numeric(row.get("public_p95_3d_m"))
                - _numeric(best.get("public_p95_3d_m")),
                "max_delta_m": _numeric(row.get("public_max_3d_m"))
                - _numeric(best.get("public_max_3d_m")),
                "class_accuracy_delta": _numeric(row.get("uav_type_accuracy"))
                - _numeric(best.get("uav_type_accuracy")),
            }
        )
    return pd.DataFrame.from_records(records, columns=_pairwise_columns())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.track5_scorecard_compare",
        description="rank local MMUAD Track 5 scorecard JSON files",
    )
    parser.add_argument("scorecard_json", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--pairwise-delta-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--pose-reference-mse", type=float, default=DEFAULT_POSE_REFERENCE_MSE)
    parser.add_argument("--top3-reference-mse", type=float, default=DEFAULT_TOP3_REFERENCE_MSE)
    parser.add_argument(
        "--class-reference-accuracy",
        type=float,
        default=DEFAULT_CLASS_REFERENCE_ACCURACY,
    )
    args = parser.parse_args(argv)

    comparison = compare_track5_scorecards(
        args.scorecard_json,
        pose_reference_mse=args.pose_reference_mse,
        top3_reference_mse=args.top3_reference_mse,
        class_reference_accuracy=args.class_reference_accuracy,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)
    paths: dict[str, str] = {"comparison_csv": str(args.output_csv)}
    if args.pairwise_delta_csv is not None:
        pairwise = build_pairwise_scorecard_deltas(comparison)
        args.pairwise_delta_csv.parent.mkdir(parents=True, exist_ok=True)
        pairwise.to_csv(args.pairwise_delta_csv, index=False)
        paths["pairwise_delta_csv"] = str(args.pairwise_delta_csv)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(_comparison_summary(comparison, paths=paths), indent=2),
            encoding="utf-8",
        )
        paths["summary_json"] = str(args.summary_json)
    print("mmuad_track5_scorecard_compare=ok")
    print(f"scorecard_count={len(comparison)}")
    print(f"best_label={comparison.iloc[0]['scorecard_label']}")
    print(f"best_pose_mse={comparison.iloc[0]['pose_mse_loss_m2']}")
    print(f"output_csv={args.output_csv}")
    return 0


def _scorecard_record(
    path: Path,
    *,
    pose_reference_mse: float,
    top3_reference_mse: float,
    class_reference_accuracy: float,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    public = payload.get("public_track5", {}) if isinstance(payload, dict) else {}
    pooled = public.get("pooled", {}) if isinstance(public, dict) else {}
    validation = payload.get("validation", {}) if isinstance(payload, dict) else {}
    pose_mse = _numeric(
        pooled.get("pose_mse_loss_m2", pooled.get("mse_3d_m2", pooled.get("mse")))
    )
    class_acc = _numeric(
        pooled.get("uav_type_accuracy", pooled.get("classification_accuracy"))
    )
    return {
        "scorecard_label": _scorecard_label(path, payload),
        "scorecard_json": str(path),
        "results_path": payload.get("results_path"),
        "scorecard_leaderboard_ready": payload.get("scorecard_leaderboard_ready"),
        "codabench_upload_ready": payload.get("codabench_upload_ready"),
        "validation_leaderboard_ready": validation.get("leaderboard_ready"),
        "pose_mse_loss_m2": pose_mse,
        "public_rmse_3d_m": _numeric(pooled.get("rmse_3d_m")),
        "public_p95_3d_m": _numeric(pooled.get("p95_3d_m")),
        "public_max_3d_m": _numeric(pooled.get("max_3d_m")),
        "uav_type_accuracy": class_acc,
        "pose_mse_delta_to_reference": pose_mse - float(pose_reference_mse),
        "pose_mse_delta_to_top3_reference": pose_mse - float(top3_reference_mse),
        "class_accuracy_delta_to_reference": class_acc - float(class_reference_accuracy),
        "beats_pose_reference": bool(np.isfinite(pose_mse) and pose_mse < pose_reference_mse),
        "beats_top3_pose_reference": bool(np.isfinite(pose_mse) and pose_mse < top3_reference_mse),
        "beats_class_reference": bool(
            np.isfinite(class_acc) and class_acc > class_reference_accuracy
        ),
        "leaderboard_blocking_reasons": ";".join(
            str(item) for item in payload.get("leaderboard_blocking_reasons", [])
        ),
    }


def _scorecard_label(path: Path, payload: dict[str, Any]) -> str:
    results_path = payload.get("results_path")
    if results_path:
        return Path(str(results_path)).stem
    return Path(path).stem


def _comparison_summary(comparison: pd.DataFrame, *, paths: dict[str, str]) -> dict[str, Any]:
    best = comparison.sort_values("rank").iloc[0]
    return {
        "schema": "raft-uav-mmuad-track5-scorecard-comparison-v1",
        "scorecard_count": int(len(comparison)),
        "best_label": str(best["scorecard_label"]),
        "best_pose_mse_loss_m2": _numeric(best.get("pose_mse_loss_m2")),
        "best_public_rmse_3d_m": _numeric(best.get("public_rmse_3d_m")),
        "best_public_p95_3d_m": _numeric(best.get("public_p95_3d_m")),
        "best_public_max_3d_m": _numeric(best.get("public_max_3d_m")),
        "best_uav_type_accuracy": _numeric(best.get("uav_type_accuracy")),
        "best_scorecard_leaderboard_ready": _bool_or_none(
            best.get("scorecard_leaderboard_ready")
        ),
        "best_codabench_upload_ready": _bool_or_none(best.get("codabench_upload_ready")),
        "paths": dict(paths),
    }


def _numeric(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)):
        return bool(float(numeric) != 0.0)

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "<na>", "nat"}:
        return False
    return None


def _pairwise_columns() -> list[str]:
    return [
        "baseline_label",
        "comparison_label",
        "comparison_rank",
        "pose_mse_delta",
        "rmse_delta_m",
        "p95_delta_m",
        "max_delta_m",
        "class_accuracy_delta",
    ]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
