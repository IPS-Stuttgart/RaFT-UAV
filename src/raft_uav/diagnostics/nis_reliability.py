"""NIS reliability reports for RF/radar update diagnostics.

The existing covariance-calibration command fits scalar covariance multipliers.
This companion report keeps the full reliability picture visible: acceptance
rates, chi-square CDF agreement, high quantiles, and the covariance scale implied
by both mean and tail NIS statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

DEFAULT_GATE_PROBABILITIES = (0.95, 0.99)
DEFAULT_GROUP_COLUMNS = ("source", "measurement_dim")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-nis-reliability",
        description="summarize NIS reliability by source, dimension, and optional run metadata",
    )
    parser.add_argument(
        "diagnostics",
        nargs="+",
        type=Path,
        help="diagnostics.csv/paper_strict_estimates.csv files or directories containing them",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nis-reliability"))
    parser.add_argument("--output-name", default="nis_reliability")
    parser.add_argument(
        "--group-by",
        action="append",
        default=None,
        help=(
            "extra or replacement group column; repeatable. Defaults to source and "
            "measurement_dim. Use --replace-default-groups to group only by these columns."
        ),
    )
    parser.add_argument(
        "--replace-default-groups",
        action="store_true",
        help="use only --group-by columns instead of appending to source/measurement_dim",
    )
    parser.add_argument(
        "--gate-probability",
        action="append",
        type=float,
        default=None,
        help="chi-square gate probability to report; repeatable; defaults to 0.95 and 0.99",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="drop rejected updates before reliability statistics",
    )
    args = parser.parse_args(argv)

    if args.replace_default_groups:
        group_columns = tuple(args.group_by or DEFAULT_GROUP_COLUMNS)
    else:
        group_columns = tuple(dict.fromkeys((*DEFAULT_GROUP_COLUMNS, *(args.group_by or []))))
    gate_probabilities = tuple(args.gate_probability or DEFAULT_GATE_PROBABILITIES)
    result = run_nis_reliability_report(
        inputs=args.diagnostics,
        output_dir=args.output_dir,
        output_name=args.output_name,
        group_columns=group_columns,
        gate_probabilities=gate_probabilities,
        accepted_only=args.accepted_only,
    )
    print(f"summary_csv={result['summary_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def run_nis_reliability_report(
    *,
    inputs: Iterable[Path | str],
    output_dir: Path = Path("outputs/nis-reliability"),
    output_name: str = "nis_reliability",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    gate_probabilities: Sequence[float] = DEFAULT_GATE_PROBABILITIES,
    accepted_only: bool = False,
) -> dict[str, Any]:
    """Read diagnostics files, write NIS reliability CSV/JSON, and return paths."""

    paths = discover_nis_diagnostics_paths(inputs)
    frame = read_nis_diagnostics(paths)
    summary = nis_reliability_summary(
        frame,
        group_columns=group_columns,
        gate_probabilities=gate_probabilities,
        accepted_only=accepted_only,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_csv = output / f"{output_name}.csv"
    summary_json = output / f"{output_name}.json"
    summary.to_csv(summary_csv, index=False)
    payload = {
        "summary_csv": str(summary_csv),
        "input_paths": [str(path) for path in paths],
        "group_columns": list(group_columns),
        "gate_probabilities": [float(value) for value in gate_probabilities],
        "accepted_only": bool(accepted_only),
        "rows": summary.to_dict(orient="records"),
    }
    summary_json.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {**payload, "summary_json": str(summary_json)}


def discover_nis_diagnostics_paths(inputs: Iterable[Path | str]) -> list[Path]:
    """Return candidate diagnostics CSV files from explicit files or directories."""

    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
            continue
        if path.is_dir():
            paths.extend(sorted(path.rglob("diagnostics.csv")))
            paths.extend(sorted(path.rglob("paper_strict_estimates.csv")))
            continue
        raise FileNotFoundError(f"diagnostics input does not exist: {path}")
    unique = sorted(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError("no NIS diagnostics files found")
    return unique


def read_nis_diagnostics(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Read diagnostics CSV files and normalize source/dimension/NIS columns."""

    frames: list[pd.DataFrame] = []
    for path_like in paths:
        path = Path(path_like)
        frame = pd.read_csv(path)
        frame["diagnostics_path"] = str(path)
        if "measurement_dim" not in frame.columns:
            frame["measurement_dim"] = _infer_measurement_dim(frame)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def nis_reliability_summary(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    gate_probabilities: Sequence[float] = DEFAULT_GATE_PROBABILITIES,
    accepted_only: bool = False,
) -> pd.DataFrame:
    """Return reliability statistics for normalized innovation squared samples."""

    work = _normalized_nis_frame(frame, group_columns=group_columns, accepted_only=accepted_only)
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    groupers = list(group_columns)
    for group_key, group in work.groupby(groupers, sort=True, dropna=False):
        row = _group_key_record(groupers, group_key)
        values = group["nis"].to_numpy(dtype=float)
        dim = _single_int_or_nan(group["measurement_dim"])
        row.update(_nis_stats(values, dim=dim, gate_probabilities=gate_probabilities))
        if "accepted" in group.columns:
            accepted = group["accepted"].map(_truthy)
            row["accepted_count"] = int(accepted.sum())
            row["accepted_fraction"] = float(accepted.mean()) if len(accepted) else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _normalized_nis_frame(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    accepted_only: bool,
) -> pd.DataFrame:
    if "nis" not in frame.columns:
        raise KeyError("diagnostics frame is missing required column 'nis'")
    work = frame.copy()
    if "source" not in work.columns:
        work["source"] = "unknown"
    if "measurement_dim" not in work.columns:
        work["measurement_dim"] = _infer_measurement_dim(work)
    if accepted_only and "accepted" in work.columns:
        work = work.loc[work["accepted"].map(_truthy)].copy()
    work["nis"] = pd.to_numeric(work["nis"], errors="coerce")
    work["measurement_dim"] = pd.to_numeric(work["measurement_dim"], errors="coerce")
    for column in group_columns:
        if column not in work.columns:
            work[column] = "missing"
    nis_values = work["nis"].to_numpy(dtype=float)
    dim_values = work["measurement_dim"].to_numpy(dtype=float)
    integer_dim = np.isclose(dim_values, np.rint(dim_values))
    finite = (
        np.isfinite(nis_values)
        & (nis_values >= 0.0)
        & np.isfinite(dim_values)
        & (dim_values > 0.0)
        & integer_dim
    )
    return work.loc[finite].copy()


def _infer_measurement_dim(frame: pd.DataFrame) -> pd.Series:
    if "source" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    source = frame["source"].astype(str).str.lower()
    values = np.where(source.eq("rf"), 2, np.where(source.eq("radar"), 3, np.nan))
    return pd.Series(values, index=frame.index)


def _nis_stats(
    values: np.ndarray,
    *,
    dim: int | None,
    gate_probabilities: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    out: dict[str, Any] = {"count": int(values.size)}
    if values.size == 0:
        return out
    out.update(
        {
            "nis_mean": float(np.mean(values)),
            "nis_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "nis_median": float(np.percentile(values, 50.0)),
            "nis_p90": float(np.percentile(values, 90.0)),
            "nis_p95": float(np.percentile(values, 95.0)),
            "nis_p99": float(np.percentile(values, 99.0)),
            "nis_max": float(np.max(values)),
        }
    )
    if dim is None:
        return out
    out["chi2_mean_expected"] = float(dim)
    out["mean_covariance_scale"] = float(out["nis_mean"] / max(float(dim), 1.0e-12))
    sorted_values = np.sort(values)
    empirical_cdf_upper = np.arange(1, values.size + 1, dtype=float) / float(values.size)
    empirical_cdf_lower = np.arange(values.size, dtype=float) / float(values.size)
    theoretical_cdf = chi2.cdf(sorted_values, df=int(dim))
    out["chi2_ks_distance"] = float(
        max(
            np.max(empirical_cdf_upper - theoretical_cdf),
            np.max(theoretical_cdf - empirical_cdf_lower),
        )
    )
    for probability in gate_probabilities:
        probability = _validate_probability(probability)
        threshold = float(chi2.ppf(probability, df=int(dim)))
        suffix = _probability_suffix(probability)
        actual = float(np.mean(values <= threshold))
        observed_quantile = float(np.quantile(values, probability))
        out[f"gate_threshold_{suffix}"] = threshold
        out[f"expected_under_gate_{suffix}"] = float(probability)
        out[f"actual_under_gate_{suffix}"] = actual
        out[f"acceptance_gap_{suffix}"] = actual - float(probability)
        out[f"observed_quantile_{suffix}"] = observed_quantile
        out[f"tail_covariance_scale_{suffix}"] = observed_quantile / max(threshold, 1.0e-12)
    return out


def _group_key_record(group_columns: Sequence[str], group_key: object) -> dict[str, object]:
    if len(group_columns) == 1:
        values = (group_key,)
    else:
        values = tuple(group_key if isinstance(group_key, tuple) else (group_key,))
    return {column: values[index] for index, column in enumerate(group_columns)}


def _single_int_or_nan(series: pd.Series) -> int | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    values = numeric.to_numpy(dtype=float)
    integer_like = np.isfinite(values) & np.isclose(values, np.rint(values))
    if not integer_like.all():
        return None
    unique = np.unique(np.rint(values).astype(int))
    if len(unique) != 1:
        return None
    return int(unique[0])


def _validate_probability(value: float) -> float:
    probability = float(value)
    if not 0.0 < probability < 1.0:
        raise ValueError("gate probability must be in (0, 1)")
    return probability


def _probability_suffix(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "accepted"}


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
