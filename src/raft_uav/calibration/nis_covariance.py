"""NIS-based measurement-covariance calibration utilities.

The Kalman and IMM baselines already record normalized innovation squared (NIS)
for every RF and radar update.  This module turns those diagnostics into
source/dimension-specific covariance multipliers and exposes a small runtime hook
that can scale newly constructed tracking measurements from a calibration JSON.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

ENV_NIS_COVARIANCE_CALIBRATION_JSON = "RAFT_UAV_NIS_COVARIANCE_CALIBRATION_JSON"
NIS_COVARIANCE_CALIBRATION_SCHEMA = "raft-uav-nis-covariance-calibration-v1"
NIS_COVARIANCE_CALIBRATION_METHODS = ("mean", "quantile")

_CACHED_PATH: str | None = None
_CACHED_MTIME_NS: int | None = None
_CACHED_CALIBRATION: dict[str, Any] | None = None


@dataclass(frozen=True)
class NISCovarianceCalibrationGroup:
    """One fitted covariance multiplier for one measurement source and dimension."""

    source: str
    measurement_dim: int
    count: int
    method: str
    statistic: float
    target: float
    raw_scale: float
    applied_scale: float
    enabled: bool
    accepted_only: bool
    quantile: float | None = None

    def to_record(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return asdict(self)


def discover_diagnostics_paths(inputs: Iterable[Path | str]) -> list[Path]:
    """Return diagnostics CSV files from explicit files or recursively from folders."""

    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
            continue
        if path.is_dir():
            paths.extend(sorted(path.rglob("diagnostics.csv")))
            continue
        raise FileNotFoundError(f"diagnostics input does not exist: {path}")
    return sorted(dict.fromkeys(paths))


def read_diagnostics_frames(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Read and concatenate diagnostics CSV files, tagging each source path."""

    diagnostics_paths = discover_diagnostics_paths(paths)
    if not diagnostics_paths:
        raise FileNotFoundError("no diagnostics.csv files found")
    frames: list[pd.DataFrame] = []
    for path in diagnostics_paths:
        frame = pd.read_csv(path)
        frame["diagnostics_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def fit_nis_covariance_calibration_from_paths(
    inputs: Iterable[Path | str],
    *,
    method: str = "mean",
    quantile: float = 0.95,
    min_samples: int = 20,
    min_scale: float = 0.25,
    max_scale: float = 25.0,
    accepted_only: bool = True,
) -> dict[str, Any]:
    """Fit a calibration payload from one or more diagnostics files or folders."""

    diagnostics_paths = discover_diagnostics_paths(inputs)
    frame = read_diagnostics_frames(diagnostics_paths)
    payload = fit_nis_covariance_calibration_from_frame(
        frame,
        method=method,
        quantile=quantile,
        min_samples=min_samples,
        min_scale=min_scale,
        max_scale=max_scale,
        accepted_only=accepted_only,
    )
    payload["input_diagnostics"] = [str(path) for path in diagnostics_paths]
    return payload


def fit_nis_covariance_calibration_from_frame(
    frame: pd.DataFrame,
    *,
    method: str = "mean",
    quantile: float = 0.95,
    min_samples: int = 20,
    min_scale: float = 0.25,
    max_scale: float = 25.0,
    accepted_only: bool = True,
) -> dict[str, Any]:
    """Fit per-source covariance multipliers by matching NIS to chi-square targets.

    With ``method='mean'``, the observed mean NIS is matched to the measurement
    dimension.  With ``method='quantile'``, the observed quantile is matched to
    the corresponding chi-square quantile.  The fitted multiplier scales the
    measurement covariance before future updates; values greater than one make
    the source less confident, values below one make it more confident.
    """

    method = _validate_method(method)
    quantile = _validate_quantile(quantile)
    min_samples = _validate_nonnegative_int(min_samples, "min_samples")
    min_scale = _validate_positive_float(min_scale, "min_scale")
    max_scale = _validate_positive_float(max_scale, "max_scale")
    if max_scale < min_scale:
        raise ValueError("max_scale must be >= min_scale")

    work = _normalized_diagnostics_frame(frame, accepted_only=accepted_only)
    groups: dict[str, dict[str, object]] = {}
    for (source, measurement_dim), group in work.groupby(["source", "measurement_dim"], sort=True):
        source_str = str(source)
        dim = int(measurement_dim)
        values = group["nis"].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values >= 0.0)]
        if values.size == 0:
            continue
        calibration = _fit_group(
            source=source_str,
            measurement_dim=dim,
            values=values,
            method=method,
            quantile=quantile,
            min_samples=min_samples,
            min_scale=min_scale,
            max_scale=max_scale,
            accepted_only=accepted_only,
        )
        groups[_group_key(source_str, dim)] = calibration.to_record()

    return {
        "schema": NIS_COVARIANCE_CALIBRATION_SCHEMA,
        "method": method,
        "quantile": float(quantile) if method == "quantile" else None,
        "min_samples": int(min_samples),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "accepted_only": bool(accepted_only),
        "groups": groups,
    }


def write_nis_covariance_calibration(payload: Mapping[str, Any], path: Path | str) -> Path:
    """Write a calibration payload as deterministic JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_jsonable_payload(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def load_nis_covariance_calibration(path: Path | str) -> dict[str, Any]:
    """Load and validate a NIS covariance calibration JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_nis_covariance_calibration(payload)
    return payload


def validate_nis_covariance_calibration(payload: Mapping[str, Any]) -> None:
    """Validate a calibration payload and raise ``ValueError`` if malformed."""

    if payload.get("schema") != NIS_COVARIANCE_CALIBRATION_SCHEMA:
        raise ValueError("unknown NIS covariance calibration schema")
    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("NIS covariance calibration must contain a groups object")
    for key, group in groups.items():
        if not isinstance(group, Mapping):
            raise ValueError(f"calibration group {key!r} must be an object")
        source = str(group.get("source", ""))
        dim = int(group.get("measurement_dim", 0))
        if key != _group_key(source, dim):
            raise ValueError(f"calibration group key {key!r} does not match source/dimension")
        scale = float(group.get("applied_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"calibration group {key!r} has invalid applied_scale")


def covariance_scale_for_source_dim(
    calibration: Mapping[str, Any] | None,
    source: str,
    measurement_dim: int,
) -> float:
    """Return the configured covariance multiplier for a source/dimension pair."""

    group = _calibration_group(calibration, source, int(measurement_dim))
    if group is None or not bool(group.get("enabled", False)):
        return 1.0
    scale = float(group.get("applied_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid covariance scale for {source}:{measurement_dim}")
    return scale


def runtime_nis_covariance_calibration() -> dict[str, Any] | None:
    """Return the calibration selected by ``RAFT_UAV_NIS_COVARIANCE_CALIBRATION_JSON``."""

    global _CACHED_CALIBRATION, _CACHED_MTIME_NS, _CACHED_PATH

    path_value = os.environ.get(ENV_NIS_COVARIANCE_CALIBRATION_JSON)
    if path_value is None or path_value.strip() == "":
        _CACHED_PATH = None
        _CACHED_MTIME_NS = None
        _CACHED_CALIBRATION = None
        return None

    path = Path(path_value)
    stat = path.stat()
    cache_path = str(path)
    if (
        _CACHED_CALIBRATION is None
        or _CACHED_PATH != cache_path
        or _CACHED_MTIME_NS != int(stat.st_mtime_ns)
    ):
        _CACHED_CALIBRATION = load_nis_covariance_calibration(path)
        _CACHED_PATH = cache_path
        _CACHED_MTIME_NS = int(stat.st_mtime_ns)
    return _CACHED_CALIBRATION


def scale_covariance_for_calibrated_source(
    source: str,
    measurement_dim: int,
    covariance: np.ndarray,
) -> np.ndarray:
    """Scale a measurement covariance using the runtime calibration if configured.

    Exact source/dimension calibration groups take precedence.  When a radar
    measurement is augmented with velocity and no explicit ``radar:6`` group is
    available, reuse a fitted ``radar:3`` position calibration for the leading
    position block.  LOFO NIS calibration is usually fitted from position-only
    diagnostics, while the result-oriented SOTA runner can later enable radar
    velocity updates; without this fallback those radar measurements would
    silently bypass the fitted radar covariance scale.
    """

    array = np.asarray(covariance, dtype=float)
    source_str = str(source)
    measurement_dim_int = int(measurement_dim)
    calibration = runtime_nis_covariance_calibration()

    scale = covariance_scale_for_source_dim(calibration, source_str, measurement_dim_int)
    if scale != 1.0:
        return array * float(scale)

    if (
        source_str == "radar"
        and measurement_dim_int == 6
        and _calibration_group(calibration, source_str, measurement_dim_int) is None
    ):
        position_scale = covariance_scale_for_source_dim(calibration, source_str, 3)
        if position_scale != 1.0:
            return _scale_leading_position_block(array, position_dim=3, scale=position_scale)

    return array


def environment_assignment(path: Path | str) -> str:
    """Return a shell-friendly environment assignment for a calibration file."""

    return f"{ENV_NIS_COVARIANCE_CALIBRATION_JSON}={Path(path)}"


def _normalized_diagnostics_frame(frame: pd.DataFrame, *, accepted_only: bool) -> pd.DataFrame:
    required = {"source", "measurement_dim", "nis"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"diagnostics frame is missing required columns: {missing}")

    work = frame.copy()
    if accepted_only and "accepted" in work.columns:
        work = work.loc[work["accepted"].map(_truthy)].copy()
    work["source"] = work["source"].astype(str)
    work["measurement_dim"] = pd.to_numeric(work["measurement_dim"], errors="coerce")
    work["nis"] = pd.to_numeric(work["nis"], errors="coerce")
    work = work.dropna(subset=["source", "measurement_dim", "nis"])
    work = work.loc[np.isfinite(work["nis"].to_numpy(dtype=float))]
    work = work.loc[work["nis"].to_numpy(dtype=float) >= 0.0]
    work["measurement_dim"] = work["measurement_dim"].astype(int)
    work = work.loc[work["measurement_dim"] > 0]
    return work


def _fit_group(
    *,
    source: str,
    measurement_dim: int,
    values: np.ndarray,
    method: str,
    quantile: float,
    min_samples: int,
    min_scale: float,
    max_scale: float,
    accepted_only: bool,
) -> NISCovarianceCalibrationGroup:
    count = int(values.size)
    if method == "mean":
        statistic = float(np.mean(values))
        target = float(measurement_dim)
        quantile_value: float | None = None
    else:
        statistic = float(np.quantile(values, quantile))
        target = float(chi2.ppf(quantile, df=measurement_dim))
        quantile_value = float(quantile)
    raw_scale = statistic / target if target > 0.0 else 1.0
    enabled = bool(count >= min_samples and np.isfinite(raw_scale) and raw_scale > 0.0)
    applied_scale = float(np.clip(raw_scale, min_scale, max_scale)) if enabled else 1.0
    return NISCovarianceCalibrationGroup(
        source=source,
        measurement_dim=int(measurement_dim),
        count=count,
        method=method,
        statistic=statistic,
        target=target,
        raw_scale=float(raw_scale),
        applied_scale=applied_scale,
        enabled=enabled,
        accepted_only=bool(accepted_only),
        quantile=quantile_value,
    )


def _calibration_group(
    calibration: Mapping[str, Any] | None,
    source: str,
    measurement_dim: int,
) -> Mapping[str, Any] | None:
    if calibration is None:
        return None
    groups = calibration.get("groups", {})
    if not isinstance(groups, Mapping):
        return None
    group = groups.get(_group_key(source, int(measurement_dim)))
    return group if isinstance(group, Mapping) else None


def _scale_leading_position_block(
    covariance: np.ndarray,
    *,
    position_dim: int,
    scale: float,
) -> np.ndarray:
    array = np.asarray(covariance, dtype=float)
    factors = np.ones(array.shape[0], dtype=float)
    factors[: int(position_dim)] = np.sqrt(float(scale))
    return array * np.outer(factors, factors)


def _group_key(source: str, measurement_dim: int) -> str:
    return f"{str(source)}:{int(measurement_dim)}"


def _validate_method(method: str) -> str:
    parsed = str(method).strip().lower()
    if parsed not in NIS_COVARIANCE_CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {NIS_COVARIANCE_CALIBRATION_METHODS}")
    return parsed


def _validate_quantile(value: float) -> float:
    quantile = float(value)
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    return quantile


def _validate_positive_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _validate_nonnegative_int(value: int, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _jsonable_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): convert(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(inner) for inner in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return convert(dict(payload))
