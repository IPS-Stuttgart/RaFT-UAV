"""Radar-native geometry diagnostics for Fortem ``lla`` versus polar fields.

The Fortem JSON exposes both a geodetic target estimate (``lla``) and native
radar observables (``range``, ``azimuth``, ``elevation``). A disagreement
between those two coordinate sources is one of the fastest ways to produce a
100 m class reproduction error while every downstream tracker looks reasonable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

AZIMUTH_CONVENTIONS = ("north-clockwise", "east-counterclockwise", "east-clockwise")


def polar_offset_enu(
    range_m: Iterable[float] | np.ndarray,
    azimuth_deg: Iterable[float] | np.ndarray,
    elevation_deg: Iterable[float] | np.ndarray,
    *,
    azimuth_convention: str = "north-clockwise",
) -> np.ndarray:
    """Return ENU offsets implied by radar range/azimuth/elevation.

    ``north-clockwise`` means azimuth 0 points north and +90 points east.
    ``east-counterclockwise`` means azimuth 0 points east and +90 points north.
    ``east-clockwise`` is included as a sign-convention diagnostic.
    """

    convention = _validate_azimuth_convention(azimuth_convention)
    slant_range = np.asarray(range_m, dtype=float)
    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    horizontal = slant_range * np.cos(elevation)

    if convention == "north-clockwise":
        east = horizontal * np.sin(azimuth)
        north = horizontal * np.cos(azimuth)
    elif convention == "east-counterclockwise":
        east = horizontal * np.cos(azimuth)
        north = horizontal * np.sin(azimuth)
    else:  # east-clockwise
        east = horizontal * np.cos(azimuth)
        north = -horizontal * np.sin(azimuth)
    up = slant_range * np.sin(elevation)
    return np.column_stack([east, north, up])


def build_radar_geometry_audit_frame(
    radar: pd.DataFrame,
    *,
    radar_origin_enu_m: Iterable[float] | np.ndarray = (0.0, 0.0, 0.0),
    azimuth_convention: str = "north-clockwise",
) -> pd.DataFrame:
    """Append polar-backprojected ENU coordinates and disagreement columns."""

    required = {"east_m", "north_m", "up_m", "range_m", "azimuth_deg", "elevation_deg"}
    missing = sorted(required.difference(radar.columns))
    if missing:
        raise KeyError(f"radar geometry audit missing required columns: {missing}")

    origin = np.asarray(tuple(radar_origin_enu_m), dtype=float).reshape(3)
    out = radar.copy()
    for column in sorted(required):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    polar_offset = polar_offset_enu(
        out["range_m"].to_numpy(dtype=float),
        out["azimuth_deg"].to_numpy(dtype=float),
        out["elevation_deg"].to_numpy(dtype=float),
        azimuth_convention=azimuth_convention,
    )
    polar_enu = polar_offset + origin.reshape(1, 3)
    lla_enu = out[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    delta = polar_enu - lla_enu
    lla_from_radar_origin = lla_enu - origin.reshape(1, 3)
    lla_slant_range = np.linalg.norm(lla_from_radar_origin, axis=1)

    out["polar_east_m"] = polar_enu[:, 0]
    out["polar_north_m"] = polar_enu[:, 1]
    out["polar_up_m"] = polar_enu[:, 2]
    out["geometry_delta_east_m"] = delta[:, 0]
    out["geometry_delta_north_m"] = delta[:, 1]
    out["geometry_delta_up_m"] = delta[:, 2]
    out["geometry_delta_horizontal_m"] = np.linalg.norm(delta[:, :2], axis=1)
    out["geometry_delta_3d_m"] = np.linalg.norm(delta, axis=1)
    out["lla_slant_range_from_radar_origin_m"] = lla_slant_range
    out["range_minus_lla_slant_range_m"] = out["range_m"].to_numpy(dtype=float) - lla_slant_range
    out["radar_origin_east_m"] = float(origin[0])
    out["radar_origin_north_m"] = float(origin[1])
    out["radar_origin_up_m"] = float(origin[2])
    out["azimuth_convention"] = azimuth_convention
    return out


def summarize_radar_geometry_audit(audit: pd.DataFrame) -> dict[str, Any]:
    """Return robust summary statistics for a geometry-audit frame."""

    summary: dict[str, Any] = {"rows": int(len(audit))}
    if audit.empty:
        return summary
    for column in (
        "geometry_delta_3d_m",
        "geometry_delta_horizontal_m",
        "geometry_delta_up_m",
        "range_minus_lla_slant_range_m",
    ):
        if column in audit.columns:
            summary[column] = _series_summary(audit[column])
    if "track_id" in audit.columns:
        summary["track_ids"] = int(pd.to_numeric(audit["track_id"], errors="coerce").nunique())
    if "frame_index" in audit.columns:
        summary["frames"] = int(pd.to_numeric(audit["frame_index"], errors="coerce").nunique())
    if "azimuth_convention" in audit.columns and len(audit):
        summary["azimuth_convention"] = str(audit["azimuth_convention"].iloc[0])
    return summary


def summarize_radar_geometry_by_track(audit: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per Fortem track ID."""

    if audit.empty or "track_id" not in audit.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for track_id, group in audit.groupby("track_id", dropna=False, sort=True):
        row: dict[str, Any] = {"track_id": track_id, "rows": int(len(group))}
        if "time_s" in group.columns:
            times = pd.to_numeric(group["time_s"], errors="coerce").dropna()
            row["time_s_min"] = float(times.min()) if len(times) else np.nan
            row["time_s_max"] = float(times.max()) if len(times) else np.nan
        for column in (
            "geometry_delta_3d_m",
            "geometry_delta_horizontal_m",
            "geometry_delta_up_m",
            "range_minus_lla_slant_range_m",
        ):
            stats = _series_summary(group[column]) if column in group.columns else {}
            for key, value in stats.items():
                row[f"{column}_{key}"] = value
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def run_radar_geometry_audit(
    *,
    dataset_root: Path,
    flight: str,
    output_dir: Path,
    variant: str = "auto",
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
    radar_origin_lla: str | None = None,
    azimuth_convention: str = "north-clockwise",
    top_k: int = 25,
) -> dict[str, Any]:
    """Load one flight, run the geometry audit, and write CSV/JSON artifacts."""

    from raft_uav.diagnostics.paper_strict import load_paper_strict_inputs

    inputs = load_paper_strict_inputs(
        dataset_root=Path(dataset_root),
        flight_name=flight,
        enu_origin=enu_origin,
        enu_origin_lla=enu_origin_lla,
        lw1_origin_lla=lw1_origin_lla,
        rf_default_std_m=75.0,
        origin_config=origin_config,
        variant=variant,
    )
    if inputs.projector is None:
        raise RuntimeError("paper-strict loader returned no ENU projector")

    if radar_origin_lla:
        lat, lon, alt = _parse_lla(radar_origin_lla)
        radar_origin_enu = inputs.projector.transform(lat, lon, alt)
        radar_origin_mode = "explicit-lla"
    else:
        radar_origin_enu = np.zeros(3, dtype=float)
        radar_origin_mode = "enu-origin"

    audit = build_radar_geometry_audit_frame(
        inputs.radar,
        radar_origin_enu_m=radar_origin_enu,
        azimuth_convention=azimuth_convention,
    )
    summary = summarize_radar_geometry_audit(audit)
    by_track = summarize_radar_geometry_by_track(audit)
    worst = audit.sort_values("geometry_delta_3d_m", ascending=False).head(int(top_k)).copy()

    flight_dir = Path(output_dir) / inputs.flight_name
    flight_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = flight_dir / "radar_geometry_audit.csv"
    by_track_csv = flight_dir / "radar_geometry_by_track.csv"
    worst_csv = flight_dir / "radar_geometry_worst_rows.csv"
    summary_json = flight_dir / "radar_geometry_summary.json"
    audit.to_csv(audit_csv, index=False)
    by_track.to_csv(by_track_csv, index=False)
    worst.to_csv(worst_csv, index=False)

    payload = {
        "flight": inputs.flight_name,
        "audit_csv": str(audit_csv),
        "by_track_csv": str(by_track_csv),
        "worst_csv": str(worst_csv),
        "summary_json": str(summary_json),
        "summary": summary,
        "radar_origin_mode": radar_origin_mode,
        "radar_origin_enu_m": [float(value) for value in radar_origin_enu],
        "enu_origin_mode": inputs.enu_origin_mode,
        "azimuth_convention": azimuth_convention,
        "file_manifest": inputs.file_manifest,
    }
    summary_json.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-radar-geometry-audit",
        description="compare Fortem LLA ENU coordinates against native polar backprojection",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True, help="flight name or unique substring")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar-geometry-audit"))
    parser.add_argument("--variant", choices=["auto", "original", "rerun"], default="auto")
    parser.add_argument(
        "--enu-origin",
        choices=["truth-first", "lla", "lw1"],
        default="lw1",
        help="same ENU origin semantics as raft-uav-paper-strict",
    )
    parser.add_argument("--enu-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lla")
    parser.add_argument("--lw1-origin-lla", default=None, help="LAT,LON,ALT for --enu-origin lw1")
    parser.add_argument("--origin-config", type=Path, default=None)
    parser.add_argument(
        "--radar-origin-lla",
        default=None,
        help="optional Fortem radar site LAT,LON,ALT; defaults to the ENU origin",
    )
    parser.add_argument(
        "--azimuth-convention",
        choices=AZIMUTH_CONVENTIONS,
        default="north-clockwise",
    )
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args(argv)

    payload = run_radar_geometry_audit(
        dataset_root=args.dataset_root,
        flight=args.flight,
        output_dir=args.output_dir,
        variant=args.variant,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
        radar_origin_lla=args.radar_origin_lla,
        azimuth_convention=args.azimuth_convention,
        top_k=args.top_k,
    )
    print(f"summary_json={payload['summary_json']}")
    print(f"audit_csv={payload['audit_csv']}")
    print(f"by_track_csv={payload['by_track_csv']}")
    return 0


def _series_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _validate_azimuth_convention(value: str) -> str:
    parsed = str(value).strip().lower()
    if parsed not in AZIMUTH_CONVENTIONS:
        raise ValueError(f"azimuth_convention must be one of {AZIMUTH_CONVENTIONS}")
    return parsed


def _parse_lla(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError("LLA must have the form LAT,LON,ALT")
    lat, lon, alt = (float(part) for part in parts)
    if not np.isfinite([lat, lon, alt]).all():
        raise ValueError("LLA values must be finite")
    return lat, lon, alt


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(inner) for inner in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
