#!/usr/bin/env python3
"""Run leakage-safe temporal, stabilization, formation, and metric-aware evidence.

Every trainable component is fitted on complementary scenario folds. Held-out
predictions are assembled exactly once, all candidates retain the raw control,
and final selection is delegated to the guarded Multi-UAV LTS tournament.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from raft_uav.multi_uav_lts._full_stack_io import (
    image_index,
    prediction_files,
    read_rows,
    write_json,
    write_rows,
)
from raft_uav.multi_uav_lts.adaptive_kinematic import (
    AdaptiveKinematicConfig,
    smooth_directory as adaptive_smooth_directory,
)
from raft_uav.multi_uav_lts.formation_reassociation import (
    FormationConfig,
    refine_directory,
)
from raft_uav.multi_uav_lts.imm_trajectory import ImmConfig, smooth_directory
from raft_uav.multi_uav_lts.learned_motion_prior import (
    MotionPrior,
    combine_affinities,
    load_sequence_translations,
    make_affinity as make_motion_affinity,
    map_affinity,
    train_prior,
)
from raft_uav.multi_uav_lts.metric_aware_boxes import (
    MetricBoxConfig,
    transform_directory,
)
from raft_uav.multi_uav_lts.observable_expert_gate import (
    apply_gate,
    fit_gate,
    read_score_csv,
    sequence_features,
)
from raft_uav.multi_uav_lts.registered_temporal_proposals import (
    RegisteredTemporalConfig,
    generate_registered_temporal,
)
from raft_uav.multi_uav_lts.scene_stabilization import (
    StabilizationConfig,
    estimate_sequence_translations,
    make_stabilized_geometry,
)
from raft_uav.multi_uav_lts.seeded_multiscan import MultiScanConfig, associate_sequence
from raft_uav.multi_uav_lts.temporal_p2_detector import (
    TemporalP2Config,
    generate_temporal_p2_roi,
    train_temporal_detector,
)
from raft_uav.multi_uav_lts.thermal_edge_model import (
    ThermalModel,
    make_affinity as make_thermal_affinity,
    train_model,
)


CANDIDATE_NAMES = (
    "raw_imm",
    "raw_akkf",
    "raw_formation_bidirectional",
    "raw_formation_bidirectional_akkf",
    "raw_metric_fixed_125",
    "raw_akkf_metric_guarded",
    "motion_only",
    "registered_motion",
    "temporal_p2_motion",
    "full_temporal_motion",
    "full_stabilized_rawprior_motion",
    "full_stabilized_motion",
    "full_stabilized_thermal_motion",
    "full_stabilized_thermal_motion_imm",
    "full_stabilized_thermal_motion_akkf",
    "formation_forward",
    "formation_backward",
    "formation_bidirectional",
    "formation_bidirectional_akkf",
    "metric_fixed_125",
    "metric_fixed_135",
    "metric_guarded",
    "metric_stress",
    "metric_gap1",
    "formation_metric",
)


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}); see {log}: {' '.join(command)}"
        )


def read_folds(path: Path) -> dict[str, int]:
    assignments = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            sequence = row.get("sequence") or row.get("sequence_name")
            if not sequence:
                raise ValueError(f"fold row has no sequence field: {row}")
            assignments[sequence] = int(row["fold"])
    if not assignments:
        raise ValueError("empty fold assignment")
    return assignments


def merge_fold(source: Path, destination: Path, names: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_file = source / f"{name}.txt"
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        target = destination / source_file.name
        payload = source_file.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"conflicting held-out prediction: {target}")
        target.write_bytes(payload)


def copy_subset(source: Path, destination: Path, names: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_file = source / f"{name}.txt"
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        (destination / source_file.name).write_bytes(source_file.read_bytes())


def directory_complete(path: Path, names: list[str]) -> bool:
    return all((path / f"{name}.txt").is_file() for name in names)


def fuse_proposals(sources: list[Path], output: Path, names: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        rows = []
        next_id = 1
        for source in sources:
            file = source / f"{name}.txt"
            if not file.is_file():
                continue
            for row in read_rows(file):
                rows.append(
                    type(row)(
                        row.frame_id,
                        next_id,
                        row.x1,
                        row.y1,
                        row.width,
                        row.height,
                        row.confidence,
                        row.class_id,
                        row.visibility,
                    )
                )
                next_id += 1
        write_rows(output / f"{name}.txt", rows)


def score_candidate(
    python: str,
    candidate: Path,
    truth: Path,
    output: Path,
    log: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            python,
            "-m",
            "raft_uav.multi_uav_lts.metrics",
            str(candidate),
            "--truth-dir",
            str(truth),
            "--output-json",
            str(output.with_suffix(".json")),
            "--sequence-summary-csv",
            str(output),
        ],
        log,
    )


def normalize_score_csv(files: dict[str, Path], output: Path) -> None:
    rows = []
    for candidate, path in files.items():
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                sequence = (
                    row.get("sequence")
                    or row.get("sequence_name")
                    or row.get("SEQ")
                    or row.get("seq")
                )
                if not sequence or sequence.upper() == "COMBINED_SEQ":
                    continue
                value = None
                for key in (
                    "CODABENCH_HOTA",
                    "codabench_hota",
                    "HOTA(0)",
                    "hota_0",
                    "HOTA",
                ):
                    if key in row and row[key] not in ("", None):
                        value = float(row[key])
                        break
                if value is None:
                    raise ValueError(f"cannot find HOTA field in {path}: {sorted(row)}")
                rows.append(
                    {
                        "sequence": Path(sequence).stem,
                        "candidate": candidate,
                        "score": value,
                    }
                )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sequence", "candidate", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def ensure_training_stabilization_cache(
    image_root: Path,
    truth_dir: Path,
    cache_dir: Path,
    names: list[str],
    config: StabilizationConfig,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        output = cache_dir / f"{name}.json"
        if output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
            if payload.get("sequence") != name:
                raise ValueError(f"invalid stabilization cache entry: {output}")
            diagnostics = payload.get("diagnostics", {})
            summary[name] = {
                "accepted": int(diagnostics.get("accepted", 0)),
                "rejected": int(diagnostics.get("rejected", 0)),
            }
            continue
        truth_file = truth_dir / f"{name}.txt"
        if not truth_file.is_file():
            raise FileNotFoundError(truth_file)
        translations, diagnostics = estimate_sequence_translations(
            image_index(image_root / name),
            read_rows(truth_file),
            config,
        )
        write_json(
            output,
            {
                "format": "raft-uav-sequence-stabilization-v1",
                "sequence": name,
                "translations": {
                    str(frame): [dy, dx]
                    for frame, (dy, dx) in translations.items()
                },
                "diagnostics": diagnostics,
            },
        )
        summary[name] = {
            "accepted": int(diagnostics.get("accepted", 0)),
            "rejected": int(diagnostics.get("rejected", 0)),
        }
    return summary


def heldout_geometry(
    image_root: Path,
    raw_dir: Path,
    cache_dir: Path,
    name: str,
    config: StabilizationConfig,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{name}.json"
    if output.is_file():
        translations = load_sequence_translations(cache_dir, name)
        payload = json.loads(output.read_text(encoding="utf-8"))
        return translations, payload.get("diagnostics", {})
    translations, diagnostics = estimate_sequence_translations(
        image_index(image_root / name),
        read_rows(raw_dir / f"{name}.txt"),
        config,
    )
    write_json(
        output,
        {
            "format": "raft-uav-sequence-stabilization-v1",
            "sequence": name,
            "translations": {
                str(frame): [dy, dx] for frame, (dy, dx) in translations.items()
            },
            "diagnostics": diagnostics,
        },
    )
    return translations, diagnostics


def fixed_metric_config(scale: float) -> MetricBoxConfig:
    return MetricBoxConfig(
        base_scale=scale,
        tiny_gain=0.0,
        innovation_gain=0.0,
        gap_gain=0.0,
        maximum_scale=scale,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--base-proposal-dir", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--temporal-p2-epochs", type=int, default=4)
    parser.add_argument("--temporal-p2-max-samples", type=int, default=20_000)
    parser.add_argument("--require-improvement", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "raft-uav-temporal-formation-contract-v2",
        "image_root": str(args.image_root.resolve()),
        "truth_dir": str(args.truth_dir.resolve()),
        "seed_dir": str(args.seed_dir.resolve()),
        "raw_dir": str(args.raw_dir.resolve()),
        "base_proposal_dir": str(args.base_proposal_dir.resolve()),
        "fold_digest": hashlib.sha256(args.fold_assignments.read_bytes()).hexdigest(),
        "runner_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold_count": args.fold_count,
        "temporal_p2_epochs": args.temporal_p2_epochs,
        "temporal_p2_max_samples": args.temporal_p2_max_samples,
        "candidate_names": list(CANDIDATE_NAMES),
    }
    contract_path = args.run_dir / "experiment-contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("run directory belongs to a different experiment contract")
    else:
        write_json(contract_path, contract)

    folds = read_folds(args.fold_assignments)
    all_names = sorted(folds)
    raw_names = {path.stem for path in prediction_files(args.raw_dir)}
    if set(all_names) != raw_names:
        raise ValueError(
            f"fold/raw sequence mismatch: folds={len(all_names)} raw={len(raw_names)}"
        )
    candidate_dirs = {
        name: args.run_dir / "candidates" / name for name in CANDIDATE_NAMES
    }
    for directory in candidate_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    progress = {"stage": "folds", "completed_folds": []}
    write_json(args.run_dir / "progress.json", progress)
    association_config = MultiScanConfig(
        affinity_weight=0.18,
        min_birth_hits=5,
        min_birth_span=4,
    )
    stabilization_config = StabilizationConfig()
    temporal_residual_config = RegisteredTemporalConfig()
    temporal_p2_config = TemporalP2Config(
        max_training_samples=args.temporal_p2_max_samples,
    )
    formation_config = FormationConfig()
    training_stabilization_dir = args.run_dir / "shared" / "training-stabilization"

    for fold in range(args.fold_count):
        held = sorted(name for name, value in folds.items() if value == fold)
        train = sorted(set(all_names) - set(held))
        fold_dir = args.run_dir / "folds" / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            fold_dir / "training-stabilization.json",
            ensure_training_stabilization_cache(
                args.image_root,
                args.truth_dir,
                training_stabilization_dir,
                train,
                stabilization_config,
            ),
        )

        raw_motion_path = fold_dir / "raw-motion-prior.json"
        stabilized_motion_path = fold_dir / "stabilized-motion-prior.json"
        thermal_path = fold_dir / "thermal-affinity.json"
        temporal_checkpoint = fold_dir / "temporal-p2.pt"
        if not raw_motion_path.is_file():
            write_json(
                fold_dir / "raw-motion-train.json",
                train_prior(
                    args.truth_dir,
                    raw_motion_path,
                    train,
                    components=4,
                    max_gap=4,
                ),
            )
        if not stabilized_motion_path.is_file():
            write_json(
                fold_dir / "stabilized-motion-train.json",
                train_prior(
                    args.truth_dir,
                    stabilized_motion_path,
                    train,
                    components=4,
                    max_gap=4,
                    stabilization_file=training_stabilization_dir,
                ),
            )
        if not thermal_path.is_file():
            write_json(
                fold_dir / "thermal-train.json",
                train_model(
                    args.image_root,
                    args.truth_dir,
                    thermal_path,
                    train,
                ),
            )
        if not temporal_checkpoint.is_file():
            write_json(
                fold_dir / "temporal-p2-train.json",
                train_temporal_detector(
                    args.image_root,
                    args.truth_dir,
                    temporal_checkpoint,
                    train,
                    epochs=args.temporal_p2_epochs,
                    device=args.device,
                    config=temporal_p2_config,
                ),
            )

        registered = fold_dir / "registered-temporal-proposals"
        if not directory_complete(registered, held):
            write_json(
                fold_dir / "registered-temporal.json",
                generate_registered_temporal(
                    args.image_root,
                    args.raw_dir,
                    registered,
                    held,
                    temporal_residual_config,
                ),
            )
        temporal_p2 = fold_dir / "temporal-p2-proposals"
        if not directory_complete(temporal_p2, held):
            write_json(
                fold_dir / "temporal-p2-predict.json",
                generate_temporal_p2_roi(
                    temporal_checkpoint,
                    args.image_root,
                    args.raw_dir,
                    temporal_p2,
                    held,
                    device=args.device,
                    config=temporal_p2_config,
                ),
            )

        registered_fused = fold_dir / "registered-fused-proposals"
        temporal_p2_fused = fold_dir / "temporal-p2-fused-proposals"
        full_fused = fold_dir / "full-temporal-fused-proposals"
        fuse_proposals([args.base_proposal_dir, registered], registered_fused, held)
        fuse_proposals([args.base_proposal_dir, temporal_p2], temporal_p2_fused, held)
        fuse_proposals(
            [args.base_proposal_dir, registered, temporal_p2],
            full_fused,
            held,
        )

        raw_motion_prior = MotionPrior.load(raw_motion_path)
        stabilized_motion_prior = MotionPrior.load(stabilized_motion_path)
        thermal_model = ThermalModel.load(thermal_path)
        raw_motion_affinity = make_motion_affinity(raw_motion_prior)
        stabilized_motion_affinity = make_motion_affinity(stabilized_motion_prior)
        association_dirs = {
            "motion_only": fold_dir / "motion-only",
            "registered_motion": fold_dir / "registered-motion",
            "temporal_p2_motion": fold_dir / "temporal-p2-motion",
            "full_temporal_motion": fold_dir / "full-temporal-motion",
            "full_stabilized_rawprior_motion": fold_dir
            / "full-stabilized-rawprior-motion",
            "full_stabilized_motion": fold_dir / "full-stabilized-motion",
            "full_stabilized_thermal_motion": fold_dir
            / "full-stabilized-thermal-motion",
        }
        for directory in association_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        diagnostics = {}
        geometry_by_sequence = {}
        heldout_stabilization_dir = fold_dir / "heldout-stabilization"
        for name in held:
            seeds = read_rows(args.seed_dir / f"{name}.txt")
            image_paths = image_index(args.image_root / name)
            translations, stabilization_diagnostics = heldout_geometry(
                args.image_root,
                args.raw_dir,
                heldout_stabilization_dir,
                name,
                stabilization_config,
            )
            geometry = make_stabilized_geometry(translations)
            geometry_by_sequence[name] = geometry
            raw_stabilized_affinity = map_affinity(raw_motion_affinity, geometry)
            learned_stabilized_affinity = map_affinity(
                stabilized_motion_affinity,
                geometry,
            )
            thermal_affinity = make_thermal_affinity(thermal_model, image_paths)
            combined_affinity = combine_affinities(
                (0.45, learned_stabilized_affinity),
                (0.55, thermal_affinity),
            )
            source_rows = {
                "motion_only": read_rows(args.base_proposal_dir / f"{name}.txt"),
                "registered_motion": read_rows(registered_fused / f"{name}.txt"),
                "temporal_p2_motion": read_rows(temporal_p2_fused / f"{name}.txt"),
                "full_temporal_motion": read_rows(full_fused / f"{name}.txt"),
                "full_stabilized_rawprior_motion": read_rows(full_fused / f"{name}.txt"),
                "full_stabilized_motion": read_rows(full_fused / f"{name}.txt"),
                "full_stabilized_thermal_motion": read_rows(full_fused / f"{name}.txt"),
            }
            specifications = {
                "motion_only": (raw_motion_affinity, None),
                "registered_motion": (raw_motion_affinity, None),
                "temporal_p2_motion": (raw_motion_affinity, None),
                "full_temporal_motion": (raw_motion_affinity, None),
                "full_stabilized_rawprior_motion": (
                    raw_stabilized_affinity,
                    geometry,
                ),
                "full_stabilized_motion": (
                    learned_stabilized_affinity,
                    geometry,
                ),
                "full_stabilized_thermal_motion": (
                    combined_affinity,
                    geometry,
                ),
            }
            sequence_diagnostics = {}
            for candidate, proposals in source_rows.items():
                affinity, candidate_geometry = specifications[candidate]
                rows, candidate_diagnostics = associate_sequence(
                    proposals,
                    seeds,
                    association_config,
                    affinity,
                    candidate_geometry,
                )
                write_rows(association_dirs[candidate] / f"{name}.txt", rows)
                candidate_diagnostics["stabilization"] = {
                    "accepted": int(stabilization_diagnostics.get("accepted", 0)),
                    "rejected": int(stabilization_diagnostics.get("rejected", 0)),
                }
                sequence_diagnostics[candidate] = candidate_diagnostics
            diagnostics[name] = sequence_diagnostics
        write_json(fold_dir / "association.json", diagnostics)

        raw_subset = fold_dir / "raw"
        copy_subset(args.raw_dir, raw_subset, held)
        raw_imm = fold_dir / "raw-imm"
        raw_akkf = fold_dir / "raw-akkf"
        smooth_directory(raw_subset, raw_imm, held, ImmConfig())
        adaptive_smooth_directory(
            raw_subset,
            raw_akkf,
            held,
            AdaptiveKinematicConfig(),
            args.image_root,
        )
        raw_formation = fold_dir / "raw-formation-bidirectional"
        refine_directory(
            raw_subset,
            args.seed_dir,
            raw_formation,
            held,
            formation_config,
            mode="bidirectional",
            geometry_by_sequence=geometry_by_sequence,
        )
        raw_formation_akkf = fold_dir / "raw-formation-bidirectional-akkf"
        adaptive_smooth_directory(
            raw_formation,
            raw_formation_akkf,
            held,
            AdaptiveKinematicConfig(),
            args.image_root,
        )
        raw_metric_fixed = fold_dir / "raw-metric-fixed-125"
        raw_akkf_metric = fold_dir / "raw-akkf-metric-guarded"
        transform_directory(
            raw_subset,
            raw_metric_fixed,
            held,
            fixed_metric_config(1.25),
            args.image_root,
        )
        transform_directory(
            raw_akkf,
            raw_akkf_metric,
            held,
            MetricBoxConfig(
                tiny_gain=0.18,
                innovation_gain=0.10,
                gap_gain=0.05,
                maximum_scale=1.35,
            ),
            args.image_root,
        )

        full_associated = association_dirs["full_stabilized_thermal_motion"]
        full_imm = fold_dir / "full-stabilized-thermal-motion-imm"
        full_akkf = fold_dir / "full-stabilized-thermal-motion-akkf"
        smooth_directory(full_associated, full_imm, held, ImmConfig())
        adaptive_smooth_directory(
            full_associated,
            full_akkf,
            held,
            AdaptiveKinematicConfig(),
            args.image_root,
        )

        formation_dirs = {
            "formation_forward": fold_dir / "formation-forward",
            "formation_backward": fold_dir / "formation-backward",
            "formation_bidirectional": fold_dir / "formation-bidirectional",
        }
        for mode, candidate in (
            ("forward", "formation_forward"),
            ("backward", "formation_backward"),
            ("bidirectional", "formation_bidirectional"),
        ):
            refine_directory(
                full_associated,
                args.seed_dir,
                formation_dirs[candidate],
                held,
                formation_config,
                mode=mode,
                geometry_by_sequence=geometry_by_sequence,
            )
        formation_akkf = fold_dir / "formation-bidirectional-akkf"
        adaptive_smooth_directory(
            formation_dirs["formation_bidirectional"],
            formation_akkf,
            held,
            AdaptiveKinematicConfig(),
            args.image_root,
        )

        metric_dirs = {
            "metric_fixed_125": fold_dir / "metric-fixed-125",
            "metric_fixed_135": fold_dir / "metric-fixed-135",
            "metric_guarded": fold_dir / "metric-guarded",
            "metric_stress": fold_dir / "metric-stress",
            "metric_gap1": fold_dir / "metric-gap1",
            "formation_metric": fold_dir / "formation-metric",
        }
        transform_directory(
            full_akkf,
            metric_dirs["metric_fixed_125"],
            held,
            fixed_metric_config(1.25),
            args.image_root,
        )
        transform_directory(
            full_akkf,
            metric_dirs["metric_fixed_135"],
            held,
            fixed_metric_config(1.35),
            args.image_root,
        )
        transform_directory(
            full_akkf,
            metric_dirs["metric_guarded"],
            held,
            MetricBoxConfig(
                tiny_gain=0.18,
                innovation_gain=0.10,
                gap_gain=0.05,
                maximum_scale=1.35,
            ),
            args.image_root,
        )
        transform_directory(
            full_akkf,
            metric_dirs["metric_stress"],
            held,
            MetricBoxConfig(
                tiny_gain=0.38,
                innovation_gain=0.28,
                gap_gain=0.14,
                maximum_scale=2.20,
            ),
            args.image_root,
        )
        transform_directory(
            full_akkf,
            metric_dirs["metric_gap1"],
            held,
            MetricBoxConfig(
                tiny_gain=0.14,
                innovation_gain=0.08,
                gap_gain=0.04,
                maximum_scale=1.30,
                max_interpolation_gap=1,
            ),
            args.image_root,
        )
        transform_directory(
            formation_akkf,
            metric_dirs["formation_metric"],
            held,
            MetricBoxConfig(
                tiny_gain=0.18,
                innovation_gain=0.10,
                gap_gain=0.05,
                maximum_scale=1.35,
            ),
            args.image_root,
        )

        fold_sources = {
            "raw_imm": raw_imm,
            "raw_akkf": raw_akkf,
            "raw_formation_bidirectional": raw_formation,
            "raw_formation_bidirectional_akkf": raw_formation_akkf,
            "raw_metric_fixed_125": raw_metric_fixed,
            "raw_akkf_metric_guarded": raw_akkf_metric,
            **association_dirs,
            "full_stabilized_thermal_motion_imm": full_imm,
            "full_stabilized_thermal_motion_akkf": full_akkf,
            **formation_dirs,
            "formation_bidirectional_akkf": formation_akkf,
            **metric_dirs,
        }
        if set(fold_sources) != set(CANDIDATE_NAMES):
            missing = sorted(set(CANDIDATE_NAMES) - set(fold_sources))
            unexpected = sorted(set(fold_sources) - set(CANDIDATE_NAMES))
            raise ValueError(
                f"candidate source mismatch: missing={missing}, unexpected={unexpected}"
            )
        for candidate, source in fold_sources.items():
            merge_fold(source, candidate_dirs[candidate], held)
        progress["completed_folds"].append(fold)
        write_json(args.run_dir / "progress.json", progress)

    for candidate, directory in candidate_dirs.items():
        count = len(prediction_files(directory))
        if count != len(all_names):
            raise ValueError(f"incomplete {candidate}: {count}/{len(all_names)}")

    progress["stage"] = "scoring"
    write_json(args.run_dir / "progress.json", progress)
    scores = {"raw": args.run_dir / "scores" / "raw.csv"}
    score_candidate(
        args.python,
        args.raw_dir,
        args.truth_dir,
        scores["raw"],
        args.run_dir / "logs" / "score-raw.log",
    )
    for candidate, directory in candidate_dirs.items():
        scores[candidate] = args.run_dir / "scores" / f"{candidate}.csv"
        score_candidate(
            args.python,
            directory,
            args.truth_dir,
            scores[candidate],
            args.run_dir / "logs" / f"score-{candidate}.log",
        )

    normalized = args.run_dir / "gate-scores.csv"
    normalize_score_csv(scores, normalized)
    score_map = read_score_csv(normalized)
    gate_dir = args.run_dir / "candidates" / "observable_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate_evidence = {}
    for fold in range(args.fold_count):
        held = sorted(name for name, value in folds.items() if value == fold)
        train = sorted(set(all_names) - set(held))
        features = {
            name: sequence_features(
                name,
                args.image_root,
                args.seed_dir,
                args.raw_dir,
            )
            for name in train
        }
        training_scores = {name: score_map[name] for name in train}
        model = fit_gate(features, training_scores)
        model.save(args.run_dir / "folds" / f"fold-{fold}" / "expert-gate.json")
        fold_output = args.run_dir / "folds" / f"fold-{fold}" / "gate-output"
        result = apply_gate(
            model,
            args.image_root,
            args.seed_dir,
            args.raw_dir,
            candidate_dirs,
            fold_output,
            held,
        )
        merge_fold(fold_output, gate_dir, held)
        gate_evidence[str(fold)] = result["counts"]
    write_json(args.run_dir / "gate-evidence.json", gate_evidence)
    gate_score = args.run_dir / "scores" / "observable_gate.csv"
    score_candidate(
        args.python,
        gate_dir,
        args.truth_dir,
        gate_score,
        args.run_dir / "logs" / "score-observable-gate.log",
    )

    progress["stage"] = "tournament"
    write_json(args.run_dir / "progress.json", progress)
    tournament = args.run_dir / "tournament"
    command = [
        args.python,
        "-m",
        "raft_uav.multi_uav_lts.tournament",
        str(args.raw_dir),
        "--truth-dir",
        str(args.truth_dir),
        "--output-dir",
        str(tournament),
        "--fold-count",
        str(args.fold_count),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--expected-sequence-count",
        str(len(all_names)),
    ]
    for candidate, directory in [*candidate_dirs.items(), ("observable_gate", gate_dir)]:
        command.extend(["--candidate", f"{candidate}={directory}"])
    if args.require_improvement:
        command.append("--require-improvement")
    run(command, args.run_dir / "logs" / "tournament.log")

    selected_file = tournament / "selected_candidate.txt"
    selected = (
        selected_file.read_text(encoding="utf-8").strip()
        if selected_file.is_file()
        else "unknown"
    )
    summary = {
        "selected_candidate": selected,
        "candidate_dirs": {
            **{name: str(path) for name, path in candidate_dirs.items()},
            "observable_gate": str(gate_dir),
        },
        "fold_count": args.fold_count,
        "sequence_count": len(all_names),
        "candidate_count": len(candidate_dirs) + 1,
    }
    write_json(args.run_dir / "temporal-formation-summary.json", summary)
    progress["stage"] = "complete"
    progress["selected_candidate"] = selected
    write_json(args.run_dir / "progress.json", progress)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
