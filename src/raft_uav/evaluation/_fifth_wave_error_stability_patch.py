"""Keep fifth-wave error metrics finite for large representable inputs."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd


_fifth_wave = import_module("raft_uav.evaluation.fifth_wave_diagnostics")
_IMPL = _fifth_wave._IMPL
_PATCH_MARKER = "_raft_uav_fifth_wave_error_stability_patch_applied"
_ORIGINAL_METRIC_FUNCTION = _IMPL._metric_function


def _stable_mean(values: np.ndarray) -> float:
    """Return the ordinary mean unless finite summation overflow requires scaling."""

    array = np.asarray(values, dtype=float).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = float(np.mean(array))
    if array.size == 0 or np.isfinite(direct) or not np.isfinite(array).all():
        return direct

    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return direct
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.mean(array / scale))


def _stable_median(values: np.ndarray) -> float:
    """Return the ordinary median unless its midpoint arithmetic overflows."""

    array = np.asarray(values, dtype=float).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = float(np.median(array))
    if array.size == 0 or np.isfinite(direct) or not np.isfinite(array).all():
        return direct

    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return direct
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.median(array / scale))


def _stable_percentile(values: np.ndarray, percentile: float) -> float:
    """Preserve ordinary percentile arithmetic and scale only after overflow."""

    array = np.asarray(values, dtype=float).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = float(np.percentile(array, percentile))
    if array.size == 0 or np.isfinite(direct) or not np.isfinite(array).all():
        return direct

    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return direct
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.percentile(array / scale, percentile))


def _stable_rmse(values: np.ndarray) -> float:
    """Preserve ordinary RMSE arithmetic and scale only after squaring overflow."""

    array = np.asarray(values, dtype=float).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = float(np.sqrt(np.mean(np.square(array))))
    if array.size == 0 or np.isfinite(direct) or not np.isfinite(array).all():
        return direct

    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return direct
    normalized = array / scale
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.sqrt(np.mean(np.square(normalized))))


def _stable_mae(values: np.ndarray) -> float:
    """Return mean absolute error without overflowing a finite sum."""

    array = np.asarray(values, dtype=float).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        absolute = np.abs(array)
    return _stable_mean(absolute)


def _stable_row_norms(deltas: np.ndarray) -> np.ndarray:
    """Return row-wise Euclidean norms without changing ordinary results."""

    matrix = np.asarray(deltas, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("deltas must be a two-dimensional array")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(matrix, axis=1)
    finite_rows = np.isfinite(matrix).all(axis=1)
    fallback = finite_rows & ~np.isfinite(norms)
    if bool(fallback.any()):
        norms = norms.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            norms[fallback] = np.hypot.reduce(np.abs(matrix[fallback]), axis=1)
    return norms


def _stable_paired_error_delta_frame(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
    dimensions: int = 3,
    label_a: str = "method_a",
    label_b: str = "method_b",
) -> pd.DataFrame:
    """Legacy paired alignment with overflow-stable position-error norms."""

    _IMPL._validate_position_frame(method_a, "method_a")
    _IMPL._validate_position_frame(method_b, "method_b")
    _IMPL._validate_position_frame(truth, "truth")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_xyz = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    a_idx, a_dt = _IMPL._nearest_indices_and_deltas(
        method_a["time_s"].to_numpy(dtype=float), truth_times
    )
    b_idx, b_dt = _IMPL._nearest_indices_and_deltas(
        method_b["time_s"].to_numpy(dtype=float), truth_times
    )
    keep = (
        (a_idx >= 0)
        & (b_idx >= 0)
        & (a_dt <= float(max_time_delta_s))
        & (b_dt <= float(max_time_delta_s))
    )
    a_pos = method_a[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[
        a_idx[keep]
    ]
    b_pos = method_b[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[
        b_idx[keep]
    ]
    ref = truth_xyz[keep]
    with np.errstate(over="ignore", invalid="ignore"):
        a_delta = a_pos[:, :dimensions] - ref[:, :dimensions]
        b_delta = b_pos[:, :dimensions] - ref[:, :dimensions]
    error_a = _stable_row_norms(a_delta)
    error_b = _stable_row_norms(b_delta)
    return pd.DataFrame(
        {
            "time_s": truth_times[keep],
            "error_a_m": error_a,
            "error_b_m": error_b,
            "delta_error_m": error_a - error_b,
            "abs_delta_error_m": np.abs(error_a - error_b),
            "method_a": label_a,
            "method_b": label_b,
            "a_time_delta_s": a_dt[keep],
            "b_time_delta_s": b_dt[keep],
        }
    )


def _stable_aligned_error_components(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> pd.DataFrame:
    """Legacy alignment with overflow-stable horizontal position errors."""

    _IMPL._validate_position_frame(estimates, "estimates")
    _IMPL._validate_position_frame(truth, "truth")
    truth_times = truth["time_s"].to_numpy(dtype=float)
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    idx, dt = _IMPL._nearest_indices_and_deltas(estimate_times, truth_times)
    keep = (idx >= 0) & (dt <= float(max_time_delta_s))
    est = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[
        idx[keep]
    ]
    ref = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[keep]
    with np.errstate(over="ignore", invalid="ignore"):
        residual = est - ref
    return pd.DataFrame(
        {
            "time_s": truth_times[keep],
            "horizontal_error_m": _stable_row_norms(residual[:, :2]),
            "vertical_error_m": residual[:, 2],
            "time_delta_s": dt[keep],
        }
    )


def _stable_estimate_error_frame(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
) -> pd.DataFrame:
    """Legacy estimate alignment with overflow-stable 2D and 3D errors."""

    _IMPL._validate_position_frame(estimates, "estimates")
    _IMPL._validate_position_frame(truth, "truth")
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    idx, dt = _IMPL._nearest_indices_and_deltas(truth_times, estimate_times)
    keep = (idx >= 0) & (dt <= float(max_time_delta_s))
    est = estimates.loc[keep, ["east_m", "north_m", "up_m"]].to_numpy(
        dtype=float
    )
    ref = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[idx[keep]]
    with np.errstate(over="ignore", invalid="ignore"):
        residual = est - ref
    out = estimates.loc[keep].copy().reset_index(drop=True)
    out["truth_time_delta_s"] = dt[keep]
    out["error_2d_m"] = _stable_row_norms(residual[:, :2])
    out["error_3d_m"] = _stable_row_norms(residual)
    out["vertical_error_m"] = residual[:, 2]
    return out


def _stable_block_bootstrap_interval(
    values,
    *,
    metric="mean",
    block_size: int = 50,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
):
    """Legacy block bootstrap with overflow-stable metrics and quantiles."""

    x = _IMPL._finite_vector(values)
    if x.size == 0:
        return _IMPL.BootstrapInterval(
            _IMPL._metric_name(metric),
            np.nan,
            np.nan,
            np.nan,
            confidence,
            0,
            block_size,
            resamples,
        )
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    fn = _IMPL._metric_function(metric)
    estimate = fn(x)
    blocks = _IMPL._contiguous_blocks(x, min(int(block_size), x.size))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(resamples), dtype=float)
    for idx in range(int(resamples)):
        sampled = [
            blocks[int(i)]
            for i in rng.integers(0, len(blocks), size=len(blocks))
        ]
        draws[idx] = fn(np.concatenate(sampled)[: x.size])
    alpha = 1.0 - float(confidence)
    return _IMPL.BootstrapInterval(
        metric=_IMPL._metric_name(metric),
        estimate=float(estimate),
        lower=_stable_percentile(draws, 100.0 * alpha / 2.0),
        upper=_stable_percentile(draws, 100.0 * (1.0 - alpha / 2.0)),
        confidence=float(confidence),
        samples=int(x.size),
        block_size=int(block_size),
        resamples=int(resamples),
    )


def _metric_function(metric):
    """Return the established metric with overflow-safe finite fallbacks."""

    if callable(metric):
        return metric
    if metric == "mean":
        return _stable_mean
    if metric == "median":
        return _stable_median
    if metric == "rmse":
        return _stable_rmse
    if metric == "mae":
        return _stable_mae
    if metric == "p95":
        return lambda values: _stable_percentile(values, 95.0)
    return _ORIGINAL_METRIC_FUNCTION(metric)


def bad_segment_table(
    times_s,
    errors_m,
    *,
    window_s: float = 20.0,
    stride_s: float = 5.0,
    top_k: int = 10,
) -> pd.DataFrame:
    """Mine worst windows without overflowing finite error summaries."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    finite = np.isfinite(times) & np.isfinite(errors)
    times = times[finite]
    errors = errors[finite]
    if times.size == 0:
        return pd.DataFrame()
    rows: list[dict[str, float | int]] = []
    start = float(times[0])
    end_time = float(times[-1])
    while start <= end_time:
        end = start + float(window_s)
        keep = (times >= start) & (times < end)
        if np.any(keep):
            values = errors[keep]
            rows.append(
                {
                    "start_time_s": start,
                    "end_time_s": end,
                    "sample_count": int(values.size),
                    "rmse_m": _stable_rmse(values),
                    "mae_m": _stable_mae(values),
                    "p95_m": _stable_percentile(values, 95.0),
                    "max_m": float(np.max(values)),
                }
            )
        start += float(stride_s)
    return pd.DataFrame.from_records(rows).sort_values(
        ["p95_m", "rmse_m"], ascending=[False, False]
    ).head(int(top_k))


def vertical_horizontal_error_summary(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
) -> dict[str, float | int]:
    """Return horizontal and vertical metrics without finite-value overflow."""

    aligned = _IMPL._aligned_error_components(
        estimates,
        truth,
        max_time_delta_s=max_time_delta_s,
    )
    if aligned.empty:
        return {"matched_count": 0}
    horizontal = aligned["horizontal_error_m"].to_numpy(dtype=float)
    vertical = np.abs(aligned["vertical_error_m"].to_numpy(dtype=float))
    horizontal_rmse = _stable_rmse(horizontal)
    vertical_rmse = _stable_rmse(vertical)
    return {
        "matched_count": int(len(aligned)),
        "horizontal_rmse_m": horizontal_rmse,
        "horizontal_p95_m": _stable_percentile(horizontal, 95.0),
        "up_rmse_m": vertical_rmse,
        "up_p95_m": _stable_percentile(vertical, 95.0),
        "vertical_to_horizontal_rmse_ratio": float(
            vertical_rmse / max(horizontal_rmse, 1e-12)
        ),
    }


def oracle_replay_realistic_gap(
    real_errors_m,
    oracle_replay_errors_m,
) -> dict[str, float | int]:
    """Compare real and oracle replay errors without finite RMSE overflow."""

    real = _IMPL._finite_vector(real_errors_m)
    oracle = _IMPL._finite_vector(oracle_replay_errors_m)
    count = min(real.size, oracle.size)
    if count == 0:
        return {"paired_samples": 0}
    real = real[:count]
    oracle = oracle[:count]
    real_rmse = _stable_rmse(real)
    oracle_rmse = _stable_rmse(oracle)
    return {
        "paired_samples": int(count),
        "real_rmse_m": real_rmse,
        "oracle_replay_rmse_m": oracle_rmse,
        "association_gap_rmse_m": float(real_rmse - oracle_rmse),
        "real_p95_m": _stable_percentile(real, 95.0),
        "oracle_replay_p95_m": _stable_percentile(oracle, 95.0),
    }


def _nis_group_summary(
    frame: pd.DataFrame,
    *,
    source_column: str,
    nis_column: str,
) -> pd.DataFrame:
    """Summarize NIS groups without overflowing finite aggregates."""

    if frame.empty or nis_column not in frame.columns:
        return pd.DataFrame(
            columns=[source_column, "count", "mean_nis", "p95_nis"]
        )
    source = (
        frame[source_column]
        if source_column in frame.columns
        else pd.Series(["all"] * len(frame))
    )
    rows = []
    for name, group in frame.groupby(source, sort=True):
        values = _IMPL._finite_vector(group[nis_column].to_numpy(dtype=float))
        rows.append(
            {
                source_column: name,
                "count": int(values.size),
                "mean_nis": _stable_mean(values) if values.size else np.nan,
                "p95_nis": (
                    _stable_percentile(values, 95.0) if values.size else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _group_error_summary(
    frame: pd.DataFrame,
    group_column: str,
    error_column: str,
) -> pd.DataFrame:
    """Summarize grouped errors without overflowing finite aggregates."""

    rows = []
    for name, group in frame.groupby(group_column, sort=True):
        values = _IMPL._finite_vector(group[error_column].to_numpy(dtype=float))
        if values.size == 0:
            continue
        rows.append(
            {
                group_column: name,
                "count": int(values.size),
                "rmse_m": _stable_rmse(values),
                "mae_m": _stable_mae(values),
                "p95_m": _stable_percentile(values, 95.0),
                "max_m": float(np.max(values)),
            }
        )
    return pd.DataFrame.from_records(rows)


def install() -> None:
    """Install stable arithmetic while preserving compatibility-wrapper semantics."""

    if getattr(_fifth_wave, _PATCH_MARKER, False):
        return

    # The compatibility package keeps sequence-local alignment wrappers whose
    # captured legacy implementations are deliberately replaceable here.
    _fifth_wave._ORIGINAL_PAIRED_ERROR_DELTA_FRAME = (
        _stable_paired_error_delta_frame
    )
    _fifth_wave._ORIGINAL_ALIGNED_ERROR_COMPONENTS = (
        _stable_aligned_error_components
    )
    _fifth_wave._ORIGINAL_ESTIMATE_ERROR_FRAME = _stable_estimate_error_frame
    _fifth_wave._ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL = (
        _stable_block_bootstrap_interval
    )

    replacements = {
        "_metric_function": _metric_function,
        "bad_segment_table": bad_segment_table,
        "vertical_horizontal_error_summary": vertical_horizontal_error_summary,
        "oracle_replay_realistic_gap": oracle_replay_realistic_gap,
        "_nis_group_summary": _nis_group_summary,
        "_group_error_summary": _group_error_summary,
    }
    for name, replacement in replacements.items():
        setattr(_IMPL, name, replacement)
        setattr(_fifth_wave, name, replacement)

    setattr(_fifth_wave, _PATCH_MARKER, True)
    setattr(_IMPL, _PATCH_MARKER, True)


install()
