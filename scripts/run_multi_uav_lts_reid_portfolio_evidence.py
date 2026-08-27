#!/usr/bin/env python3
"""Run the 102-sequence ReID portfolio and guarded-bridge evidence sweep."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import run_multi_uav_lts_fast_output_evidence as fast
import run_multi_uav_lts_improved_evidence as improved
import run_multi_uav_lts_public_evidence as evidence

from raft_uav.multi_uav_lts.reid_tracklet_portfolio import (
    FastReidAppearanceProvider,
    PortfolioParameters,
    fuse_prediction_portfolio,
)

_SOURCE_CALIBRATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source_raw_rts", ()),
    (
        "source_raw_rts_u050",
        (
            "--uncertainty-scale-x",
            "0.50",
            "--uncertainty-scale-y",
            "0.50",
            "--max-area-ratio",
            "1.50",
        ),
    ),
    (
        "source_raw_rts_u075",
        (
            "--uncertainty-scale-x",
            "0.75",
            "--uncertainty-scale-y",
            "0.75",
            "--max-area-ratio",
            "1.75",
        ),
    ),
)

_CANDIDATE_NAMES = (
    "bridge_geometry_gap1",
    "bridge_geometry_gap2",
    "bridge_reid_gap2",
    "bridge_reid_gap5_phase",
    "portfolio_geometry_w30",
    "portfolio_geometry_w60",
    "portfolio_reid_w30",
    "portfolio_reid_w60",
    "portfolio_reid_bridge2",
    "portfolio_reid_bridge5_phase",
)


def _runtime_inputs(arguments: list[str]) -> tuple[Path, Path, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parsed, _unknown = parser.parse_known_args(arguments)
    payload = evidence._load_json(parsed.inputs_json.expanduser().resolve())
    try:
        image_root = Path(payload["dataset"]["train_sequence_root"])
        checkout = Path(payload["upstream"]["checkout"])
    except (KeyError, TypeError) as error:
        raise ValueError("public inputs omit dataset/upstream paths") from error
    return image_root.expanduser().resolve(), (checkout / "BoT-SORT").resolve(), str(parsed.device)


def _complete_portfolio_candidate(
    root: Path,
    *,
    name: str,
    expected_sequences: int,
) -> Path | None:
    output_dir = root / "predictions"
    summary_path = root / "portfolio-candidate-summary.json"
    if not output_dir.is_dir() or not summary_path.is_file():
        return None
    try:
        summary = evidence._load_json(summary_path)
        digest, total_bytes, count = evidence._directory_digest(output_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        summary.get("candidate") != name
        or count != expected_sequences
        or summary.get("sequence_count") != expected_sequences
        or total_bytes <= 0
        or summary.get("prediction_content_bytes") != total_bytes
        or summary.get("prediction_content_sha256") != digest
    ):
        return None
    return output_dir


def _materialize_portfolio(
    name: str,
    sources: dict[str, Path],
    seed_dir: Path,
    *,
    run_dir: Path,
    expected_sequences: int,
    parameters: PortfolioParameters,
    provider: FastReidAppearanceProvider | None,
) -> Path:
    root = run_dir / name
    complete = _complete_portfolio_candidate(
        root,
        name=name,
        expected_sequences=expected_sequences,
    )
    if complete is not None:
        print(f"Reusing complete portfolio candidate {name}", flush=True)
        return complete

    output_dir = root / "predictions"
    summary = fuse_prediction_portfolio(
        sources,
        seed_dir,
        output_dir,
        parameters=parameters,
        appearance_provider=provider,
    )
    evidence._write_json(root / "portfolio-summary.json", asdict(summary))
    digest, total_bytes, count = evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"{name} covers {count} sequences, expected {expected_sequences}"
        )
    evidence._write_json(
        root / "portfolio-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-reid-portfolio-candidate-v1",
            "candidate": name,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            "source_paths": {key: str(value) for key, value in sources.items()},
            "parameters": asdict(parameters),
            "appearance_enabled": provider is not None,
        },
    )
    return output_dir


def _common_parameters(**changes: Any) -> PortfolioParameters:
    values: dict[str, Any] = {
        "window_frames": 60,
        "min_segment_rows": 1,
        "sample_count_per_window": 3,
        "coverage_weight": 1.0,
        "confidence_weight": 0.05,
        "appearance_weight": 0.0,
        "support_weight": 0.20,
        "smoothness_weight": 0.05,
        "source_switch_penalty": 0.04,
        "transition_speed_weight": 0.01,
        "transition_size_weight": 0.02,
        "max_transition_normalized_speed": 8.0,
        "bridge_max_gap_frames": 0,
        "bridge_max_normalized_speed": 5.0,
        "bridge_max_log_size_change": 1.0,
        "bridge_min_endpoint_confidence": 0.003,
        "bridge_confidence_decay": 0.85,
        "bridge_require_appearance": True,
        "bridge_endpoint_appearance_threshold": 0.30,
        "bridge_anchor_appearance_threshold": 0.40,
        "bridge_use_smoothed_endpoints": True,
    }
    values.update(changes)
    return PortfolioParameters(**values)


def _run_portfolio_candidates(
    proposal_dir: Path,
    seed_dir: Path,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Path]:
    image_root, botsort_root, device = _runtime_inputs(sys.argv[1:])
    raw_predictions = proposal_dir.parent / "predictions"
    _digest, raw_bytes, raw_count = evidence._directory_digest(raw_predictions)
    if raw_count != expected_sequences or raw_bytes <= 0:
        raise ValueError(
            f"raw control covers {raw_count} sequences, expected {expected_sequences}"
        )
    resolution_groups = improved._sequence_resolution_groups(image_root, seed_dir)
    source_root = run_dir / "source-bank"
    calibrated: dict[str, Path] = {}
    for name, arguments in _SOURCE_CALIBRATIONS:
        calibrated[name] = fast._materialize_calibration(
            raw_predictions,
            name,
            arguments,
            resolution_groups,
            run_dir=source_root,
            expected_sequences=expected_sequences,
        )
    raw_only = {"raw": raw_predictions}
    portfolio_sources = {"raw": raw_predictions, **calibrated}

    provider = FastReidAppearanceProvider(
        image_root,
        botsort_root,
        device=device,
        crop_scales=(1.0, 1.25),
        batch_size=32,
    )
    variants: tuple[
        tuple[str, dict[str, Path], PortfolioParameters, FastReidAppearanceProvider | None],
        ...,
    ] = (
        (
            "bridge_geometry_gap1",
            raw_only,
            _common_parameters(
                bridge_max_gap_frames=1,
                bridge_require_appearance=False,
            ),
            None,
        ),
        (
            "bridge_geometry_gap2",
            raw_only,
            _common_parameters(
                bridge_max_gap_frames=2,
                bridge_require_appearance=False,
            ),
            None,
        ),
        (
            "bridge_reid_gap2",
            raw_only,
            _common_parameters(bridge_max_gap_frames=2),
            provider,
        ),
        (
            "bridge_reid_gap5_phase",
            raw_only,
            _common_parameters(
                bridge_max_gap_frames=5,
                bridge_endpoint_appearance_threshold=0.30,
                bridge_endpoint_appearance_threshold_late=0.20,
                bridge_anchor_appearance_threshold=0.40,
                bridge_anchor_appearance_threshold_late=0.30,
            ),
            provider,
        ),
        (
            "portfolio_geometry_w30",
            portfolio_sources,
            _common_parameters(window_frames=30),
            None,
        ),
        (
            "portfolio_geometry_w60",
            portfolio_sources,
            _common_parameters(window_frames=60),
            None,
        ),
        (
            "portfolio_reid_w30",
            portfolio_sources,
            _common_parameters(window_frames=30, appearance_weight=0.75),
            provider,
        ),
        (
            "portfolio_reid_w60",
            portfolio_sources,
            _common_parameters(window_frames=60, appearance_weight=0.75),
            provider,
        ),
        (
            "portfolio_reid_bridge2",
            portfolio_sources,
            _common_parameters(
                window_frames=60,
                appearance_weight=0.75,
                bridge_max_gap_frames=2,
            ),
            provider,
        ),
        (
            "portfolio_reid_bridge5_phase",
            portfolio_sources,
            _common_parameters(
                window_frames=60,
                appearance_weight=0.75,
                bridge_max_gap_frames=5,
                bridge_endpoint_appearance_threshold=0.30,
                bridge_endpoint_appearance_threshold_late=0.20,
                bridge_anchor_appearance_threshold=0.40,
                bridge_anchor_appearance_threshold_late=0.30,
            ),
            provider,
        ),
    )
    outputs: dict[str, Path] = {}
    for name, sources, parameters, variant_provider in variants:
        outputs[name] = _materialize_portfolio(
            name,
            sources,
            seed_dir,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
            parameters=parameters,
            provider=variant_provider,
        )
    return outputs


def main() -> int:
    evidence.CANDIDATES = tuple((name, ()) for name in _CANDIDATE_NAMES)
    evidence._baseline_cache_key = improved._proposal_source_cache_key
    evidence._prepare_baseline = improved._prepare_baseline_resumable
    evidence._run_candidates = _run_portfolio_candidates
    return evidence.main()


if __name__ == "__main__":
    raise SystemExit(main())
