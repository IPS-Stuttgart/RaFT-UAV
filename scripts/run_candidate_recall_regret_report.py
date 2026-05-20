#!/usr/bin/env python3
"""Report candidate-set recall and association regret from normalized CSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.research.diagnostics import (
    association_regret,
    association_regret_summary,
    candidate_set_recall,
    track_switch_metrics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar", type=Path, required=True, help="Normalized radar candidates CSV")
    parser.add_argument("--truth", type=Path, required=True, help="Normalized truth CSV")
    parser.add_argument("--selected", type=Path, default=None, help="Selected radar CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-gate-m", type=float, default=150.0)
    parser.add_argument("--max-time-delta-s", type=float, default=1.0)
    parser.add_argument("--catprob-threshold", type=float, default=None)
    args = parser.parse_args(argv)

    radar = pd.read_csv(args.radar)
    truth = pd.read_csv(args.truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def preselector(frame: pd.DataFrame) -> pd.DataFrame:
        if args.catprob_threshold is None or "cat_prob_uav" not in frame.columns:
            return frame
        return frame.loc[pd.to_numeric(frame["cat_prob_uav"], errors="coerce") >= args.catprob_threshold]

    recall = candidate_set_recall(
        radar,
        truth,
        distance_gate_m=args.distance_gate_m,
        max_time_delta_s=args.max_time_delta_s,
        preselector=preselector,
    )
    recall_path = args.output_dir / "candidate_set_recall.csv"
    recall.to_csv(recall_path, index=False)
    summary: dict[str, object] = {
        "radar_csv": str(args.radar),
        "truth_csv": str(args.truth),
        "candidate_set_recall_csv": str(recall_path),
        "radar_frame_count": int(len(recall)),
        "target_present_rate": float(recall["target_present"].mean()) if len(recall) else float("nan"),
        "mean_best_candidate_error_m": float(recall["best_candidate_error_m"].mean()) if len(recall) else float("nan"),
    }
    if args.selected is not None:
        selected = pd.read_csv(args.selected)
        regret = association_regret(
            selected,
            radar,
            truth,
            max_time_delta_s=args.max_time_delta_s,
        )
        regret_path = args.output_dir / "association_regret.csv"
        regret.to_csv(regret_path, index=False)
        summary["association_regret_csv"] = str(regret_path)
        summary.update(association_regret_summary(regret))
        summary.update(track_switch_metrics(selected))
    summary_path = args.output_dir / "candidate_recall_regret_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_json={summary_path}")
    print(f"candidate_set_recall_csv={recall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
