"""Collect ungated/gated baseline metrics into a paper-evidence CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_common import empty_if_none, error_metric_columns, load_metrics, write_summary_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--gated-dir", type=Path, required=True)
    parser.add_argument("--inflated-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    args = parser.parse_args()

    rows = _collect_rows(args)
    write_summary_csv(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


def _collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = [("cv", args.baseline_dir), ("cv_nis_gated", args.gated_dir)]
    if args.inflated_dir is not None:
        methods.append(("cv_nis_inflated", args.inflated_dir))
    for method, root in methods:
        for flight in args.flights:
            metrics_path = root / flight / "metrics.json"
            if metrics_path.exists():
                rows.append(_row(method, metrics_path, load_metrics(metrics_path)))
    return rows


def _row(method: str, metrics_path: Path, metrics: dict[str, object]) -> dict[str, object]:
    gating = metrics.get("gating") or {}
    robust_update = metrics.get("robust_update") or {}
    source_counts = metrics.get("source_counts") or {}
    accepted_by_source = metrics.get("accepted_by_source") or {}
    rejected_by_source = metrics.get("rejected_by_source") or {}
    reweighted_by_source = metrics.get("reweighted_by_source") or {}
    posterior_records = int(metrics.get("posterior_records", 0))
    rejected = int(metrics.get("rejected_measurements", 0))

    row = {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": method,
        "radar_association": empty_if_none(
            metrics.get("radar_association", metrics.get("radar_selection"))
        ),
        "gating_enabled": _dict_get(gating, "enabled", False),
        "robust_update": empty_if_none(_dict_get(robust_update, "method")),
        "smoother": empty_if_none(_nested_metric(metrics, "smoother", "method")),
        "smoother_lag_s": empty_if_none(_nested_metric(metrics, "smoother", "lag_s")),
        "rf_gate_probability": empty_if_none(
            _gate_probability(gating, robust_update, "rf_gate_probability")
        ),
        "radar_gate_probability": empty_if_none(
            _gate_probability(gating, robust_update, "radar_gate_probability")
        ),
        "rf_inflation_alpha": empty_if_none(_dict_get(robust_update, "rf_inflation_alpha")),
        "radar_inflation_alpha": empty_if_none(
            _dict_get(robust_update, "radar_inflation_alpha")
        ),
        "posterior_records": posterior_records,
        "accepted_measurements": int(metrics.get("accepted_measurements", posterior_records)),
        "rejected_measurements": rejected,
        "reweighted_measurements": int(metrics.get("reweighted_measurements", 0)),
        "accepted_rf": _accepted_count(accepted_by_source, source_counts, "rf", rejected),
        "accepted_radar": _accepted_count(accepted_by_source, source_counts, "radar", rejected),
        "rejected_rf": _source_count(rejected_by_source, "rf"),
        "rejected_radar": _source_count(rejected_by_source, "radar"),
        "reweighted_rf": _source_count(reweighted_by_source, "rf"),
        "reweighted_radar": _source_count(reweighted_by_source, "radar"),
    }
    row.update(error_metric_columns(metrics))
    row["metrics_path"] = str(metrics_path)
    return row


def _accepted_count(
    accepted_by_source: object,
    source_counts: object,
    source: str,
    rejected: int,
) -> int:
    fallback = _source_count(source_counts, source) if rejected == 0 else 0
    if not isinstance(accepted_by_source, dict):
        return fallback
    return int(accepted_by_source.get(source, fallback))


def _gate_probability(gating: object, robust_update: object, key: str) -> object:
    value = _dict_get(gating, key)
    return _dict_get(robust_update, key) if value is None else value


def _nested_metric(metrics: dict[str, object], section: str, key: str) -> object:
    return _dict_get(metrics.get(section), key)


def _source_count(counts: object, source: str) -> int:
    return int(counts.get(source, 0)) if isinstance(counts, dict) else 0


def _dict_get(mapping: object, key: str, default: object = None) -> object:
    return mapping.get(key, default) if isinstance(mapping, dict) else default


if __name__ == "__main__":
    raise SystemExit(main())
