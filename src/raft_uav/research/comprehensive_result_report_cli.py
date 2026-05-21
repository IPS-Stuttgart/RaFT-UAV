"""CLI for comprehensive RaFT-UAV result-improvement reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.io.aerpaw import DEFAULT_RADAR_CLOCK_OFFSET_S, DEFAULT_RF_CLOCK_OFFSET_S
from raft_uav.research.comprehensive_improvements import (
    comprehensive_run_report,
    load_normalized_flight_frames,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comprehensive_result_report"))
    parser.add_argument("--catprob-threshold", type=float, default=0.4)
    parser.add_argument("--truth-gate-m", type=float, default=150.0)
    parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    parser.add_argument("--time-offset-min-s", type=float, default=-2.0)
    parser.add_argument("--time-offset-max-s", type=float, default=2.0)
    parser.add_argument("--time-offset-step-s", type=float, default=0.05)
    parser.add_argument("--rf-clock-offset-s", type=float, default=DEFAULT_RF_CLOCK_OFFSET_S)
    parser.add_argument("--radar-clock-offset-s", type=float, default=DEFAULT_RADAR_CLOCK_OFFSET_S)
    args = parser.parse_args(argv)

    frames = load_normalized_flight_frames(
        args.dataset_root,
        args.flight,
        rf_clock_offset_s=args.rf_clock_offset_s,
        radar_clock_offset_s=args.radar_clock_offset_s,
    )
    selected_radar = _read_optional_csv(args.run_dir / "selected_radar.csv") if args.run_dir else None
    estimates = _read_optional_csv(args.run_dir / "estimates.csv") if args.run_dir else None
    offset_grid = np.arange(
        float(args.time_offset_min_s),
        float(args.time_offset_max_s) + 0.5 * float(args.time_offset_step_s),
        float(args.time_offset_step_s),
    )
    report = comprehensive_run_report(
        radar=frames.radar,
        rf=frames.rf,
        truth=frames.truth,
        selected_radar=selected_radar,
        estimates=estimates,
        catprob_threshold=args.catprob_threshold,
        truth_gate_m=args.truth_gate_m,
        truth_time_gate_s=args.truth_time_gate_s,
        offset_grid_s=offset_grid,
    )

    output_dir = args.output_dir / _slug(args.flight)
    if args.run_dir is not None:
        output_dir = output_dir / _slug(args.run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in report["tables"].items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            table.to_csv(output_dir / f"{name}.csv", index=False)
        else:
            (output_dir / f"{name}.csv").write_text("\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(report["summary"], indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "recommendations.md").write_text(
        _recommendations_markdown(args.flight, args.run_dir, report["summary"]),
        encoding="utf-8",
    )
    print(f"wrote comprehensive result-improvement report to {output_dir}")
    return 0


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path is not None and path.exists() else None


def _recommendations_markdown(
    flight: str,
    run_dir: Path | None,
    summary: dict[str, Any],
) -> str:
    lines = [
        f"# Comprehensive RaFT-UAV result-improvement report for {flight}",
        "",
    ]
    if run_dir is not None:
        lines.extend([f"Run directory: `{run_dir}`", ""])
    lines.extend(["## Summary", ""])
    for key in sorted(k for k in summary if k != "recommended_next_actions"):
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6g}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Recommended next actions", ""])
    for index, action in enumerate(summary.get("recommended_next_actions", []), start=1):
        lines.append(f"{index}. {action}")
    lines.append("")
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _slug(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


if __name__ == "__main__":
    raise SystemExit(main())
