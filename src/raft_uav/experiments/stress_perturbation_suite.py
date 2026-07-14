"""Generate stress-test perturbation CSVs for normalized measurement artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerturbationSpec:
    """One deterministic perturbation setting."""

    name: str
    drop_rate: float = 0.0
    false_track_count: int = 0
    time_jitter_std_s: float = 0.0
    catprob_scale: float = 1.0
    velocity_noise_std_mps: float = 0.0
    position_noise_std_m: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop_rate < 1.0:
            raise ValueError("drop_rate must be in [0, 1)")
        if self.false_track_count < 0:
            raise ValueError("false_track_count must be nonnegative")
        for name in ("time_jitter_std_s", "velocity_noise_std_mps", "position_noise_std_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        catprob_scale = float(self.catprob_scale)
        if not np.isfinite(catprob_scale) or catprob_scale < 0.0:
            raise ValueError("catprob_scale must be finite and nonnegative")


def perturb_measurements(frame: pd.DataFrame, spec: PerturbationSpec) -> pd.DataFrame:
    """Return a perturbed copy of a normalized RF/radar frame."""

    rng = np.random.default_rng(int(spec.seed))
    out = frame.copy()
    if spec.drop_rate > 0.0 and len(out):
        keep = rng.random(len(out)) >= float(spec.drop_rate)
        out = out.loc[keep].copy()
    if spec.time_jitter_std_s > 0.0 and "time_s" in out.columns:
        out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce") + rng.normal(
            0.0, float(spec.time_jitter_std_s), size=len(out)
        )
    for col in ("east_m", "north_m", "up_m"):
        if spec.position_noise_std_m > 0.0 and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") + rng.normal(
                0.0, float(spec.position_noise_std_m), size=len(out)
            )
    for col in ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps"):
        if spec.velocity_noise_std_mps > 0.0 and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") + rng.normal(
                0.0, float(spec.velocity_noise_std_mps), size=len(out)
            )
    if "cat_prob_uav" in out.columns:
        out["cat_prob_uav"] = np.clip(
            pd.to_numeric(out["cat_prob_uav"], errors="coerce") * float(spec.catprob_scale),
            0.0,
            1.0,
        )
    if spec.false_track_count > 0 and not out.empty:
        out = pd.concat([out, _false_tracks(out, spec, rng)], ignore_index=True)
    out["perturbation_name"] = spec.name
    return out.sort_values([c for c in ("time_s", "frame_index", "track_id") if c in out.columns]).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", action="append", required=True, help="JSON object or path to JSON spec")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input_csv)
    specs = [_load_spec(raw) for raw in args.spec]
    output_paths = _planned_output_paths(specs, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec, path in zip(specs, output_paths):
        perturbed = perturb_measurements(frame, spec)
        perturbed.to_csv(path, index=False)
        manifest.append({"path": str(path), "rows": int(len(perturbed)), "spec": asdict(spec)})
    manifest_path = args.output_dir / "stress_perturbation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest_json={manifest_path}")
    return 0


def _false_tracks(frame: pd.DataFrame, spec: PerturbationSpec, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    template = frame.sample(n=min(len(frame), max(1, spec.false_track_count)), random_state=int(spec.seed), replace=True)
    max_track = 0
    if "track_id" in frame.columns:
        track_values = pd.to_numeric(frame["track_id"], errors="coerce").dropna()
        if not track_values.empty:
            max_track = int(track_values.max())
    for index in range(int(spec.false_track_count)):
        row = template.iloc[index % len(template)].copy()
        for col in ("east_m", "north_m", "up_m"):
            if col in row.index:
                row[col] = float(row[col]) + rng.normal(0.0, max(float(spec.position_noise_std_m), 100.0))
        if "track_id" in row.index:
            row["track_id"] = max_track + index + 1
        if "cat_prob_uav" in row.index:
            row["cat_prob_uav"] = min(float(row.get("cat_prob_uav", 0.5)), 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_spec(raw: str) -> PerturbationSpec:
    path = Path(raw)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(raw)
    return PerturbationSpec(**payload)


def _planned_output_paths(
    specs: Sequence[PerturbationSpec],
    output_dir: Path,
) -> list[Path]:
    """Return collision-free output paths for all perturbation specifications."""

    output_dir = Path(output_dir)
    paths: list[Path] = []
    names_by_filename: dict[str, str] = {}
    for spec in specs:
        filename = f"{_slug(spec.name)}.csv"
        filename_key = filename.casefold()
        previous_name = names_by_filename.get(filename_key)
        if previous_name is not None:
            raise ValueError(
                "perturbation names must map to unique output filenames; "
                f"{previous_name!r} and {spec.name!r} both map to {filename!r}"
            )
        names_by_filename[filename_key] = spec.name
        paths.append(output_dir / filename)
    return paths


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
