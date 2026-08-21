"""Run the experimental proposal graph after reliability-gated Sim(2) stabilization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ._proposal_sim2 import (
    SimilarityMotionConfig,
    SimilarityMotionStep,
    SimilarityTransform,
    cumulative_transforms,
    estimate_similarity_steps,
    restore_rows,
    stabilize_rows,
    step_summary,
)
from ._records import (
    Detection,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
)

_FORBIDDEN_OPTIONS = {
    "--enable-common-motion",
    "--common-motion-min-pairs",
    "--common-motion-max-normalized-step",
    "--common-motion-max-normalized-residual",
    "--birth-require-border-entry",
    "--image-width",
    "--image-height",
}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(os.sys.argv[1:] if argv is None else argv)
    parser = _custom_parser()
    if any(token in {"-h", "--help"} for token in arguments):
        parser.print_help()
        from . import experimental_proposal_graph_tracker

        return experimental_proposal_graph_tracker.main(["--help"])
    custom, remaining = parser.parse_known_args(arguments)
    config = SimilarityMotionConfig(
        min_pairs=custom.sim2_min_pairs,
        max_rows=custom.sim2_max_rows,
        max_normalized_step=custom.sim2_max_normalized_step,
        max_normalized_residual=custom.sim2_max_normalized_residual,
        max_scale_deviation=custom.sim2_max_scale_deviation,
        max_rotation_degrees=custom.sim2_max_rotation_degrees,
        min_spread_normalized=custom.sim2_min_spread_normalized,
        min_residual_improvement=custom.sim2_min_residual_improvement,
    )
    config.validate()
    _reject_incompatible_options(remaining)
    proposal_path, label_dir, output_dir, output_json = _base_paths(remaining)
    _guard_paths(proposal_path, label_dir, output_dir, output_json)

    text_by_name = prediction_texts(proposal_path)
    rows_by_sequence: dict[str, tuple[Detection, ...]] = {}
    steps_by_sequence: dict[str, dict[int, SimilarityMotionStep]] = {}
    transforms_by_sequence: dict[str, dict[int, SimilarityTransform]] = {}
    sequence_summaries: list[dict[str, object]] = []
    for name, text in sorted(text_by_name.items()):
        rows = tuple(parse_detection_text(text, source=f"{proposal_path}:{name}"))
        reject_duplicate_keys(list(rows), label="proposal")
        sequence = Path(name).stem
        steps = estimate_similarity_steps(rows, config)
        max_frame = max((row.frame_id for row in rows), default=1)
        transforms = cumulative_transforms(steps, max_frame)
        rows_by_sequence[sequence] = rows
        steps_by_sequence[sequence] = steps
        transforms_by_sequence[sequence] = transforms
        sequence_summaries.append(
            {"sequence": sequence, "frame_count": max_frame, **step_summary(steps)}
        )

    with tempfile.TemporaryDirectory(prefix="raft-uav-lts-sim2-") as temporary:
        root = Path(temporary)
        stabilized_dir = root / "proposals"
        stabilized_dir.mkdir()
        temporary_output = root / "predictions"
        temporary_summary = root / "summary.json"
        for sequence, rows in rows_by_sequence.items():
            stabilized = stabilize_rows(rows, transforms_by_sequence[sequence])
            _write_rows(stabilized_dir / f"{sequence}.txt", stabilized)
        delegated = _delegated_arguments(
            remaining,
            stabilized_dir,
            temporary_output,
            temporary_summary,
        )
        from . import experimental_proposal_graph_tracker

        return_code = experimental_proposal_graph_tracker.main(delegated)
        if return_code != 0:
            return int(return_code)
        restored: dict[str, str] = {}
        for name, text in sorted(prediction_texts(temporary_output).items()):
            sequence = Path(name).stem
            rows = tuple(parse_detection_text(text, source=f"{temporary_output}:{name}"))
            max_frame = max((row.frame_id for row in rows), default=1)
            transforms = transforms_by_sequence.get(sequence)
            if transforms is None or max_frame not in transforms:
                transforms = cumulative_transforms(
                    steps_by_sequence.get(sequence, {}), max_frame
                )
            native_rows = restore_rows(rows, transforms)
            restored[name] = "".join(
                format_detection(row) + "\n" for row in native_rows
            )
        _publish(restored, output_dir)
        underlying = json.loads(temporary_summary.read_text(encoding="utf-8"))

    summary = {
        "schema": "raft-uav-multi-uav-lts-sim2-proposal-graph-v1",
        "proposal_path": str(proposal_path.expanduser().resolve()),
        "first_frame_label_dir": str(label_dir.expanduser().resolve()),
        "output_dir": str(output_dir.expanduser().resolve()),
        "sim2_config": asdict(config),
        "sequence_count": len(sequence_summaries),
        "step_count": sum(int(row["step_count"]) for row in sequence_summaries),
        "sim2_step_count": sum(
            int(row["sim2_step_count"]) for row in sequence_summaries
        ),
        "translation_step_count": sum(
            int(row["translation_step_count"]) for row in sequence_summaries
        ),
        "identity_step_count": sum(
            int(row["identity_step_count"]) for row in sequence_summaries
        ),
        "sequences": sequence_summaries,
        "underlying_proposal_graph": underlying,
    }
    if output_json is not None:
        _write_json(output_json, summary)
    print(f"sim2_step_count={summary['sim2_step_count']}")
    print(f"translation_step_count={summary['translation_step_count']}")
    print(f"identity_step_count={summary['identity_step_count']}")
    print(f"output_dir={output_dir}")
    return 0


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sim2-min-pairs", type=int, default=4)
    parser.add_argument("--sim2-max-rows", type=int, default=96)
    parser.add_argument("--sim2-max-normalized-step", type=float, default=8.0)
    parser.add_argument("--sim2-max-normalized-residual", type=float, default=1.5)
    parser.add_argument("--sim2-max-scale-deviation", type=float, default=0.15)
    parser.add_argument("--sim2-max-rotation-degrees", type=float, default=15.0)
    parser.add_argument("--sim2-min-spread-normalized", type=float, default=2.0)
    parser.add_argument("--sim2-min-residual-improvement", type=float, default=0.05)
    return parser


def _base_paths(
    arguments: Sequence[str],
) -> tuple[Path, Path, Path, Path | None]:
    if not arguments or arguments[0].startswith("-"):
        raise ValueError("proposal_path must be the first non-Sim2 argument")
    output_json = _option(arguments, "--output-json", required=False)
    return (
        Path(arguments[0]),
        Path(_option(arguments, "--first-frame-label-dir", required=True)),
        Path(_option(arguments, "--output-dir", required=True)),
        None if output_json is None else Path(output_json),
    )


def _reject_incompatible_options(arguments: Sequence[str]) -> None:
    for token in arguments:
        option = token.split("=", 1)[0]
        if option in _FORBIDDEN_OPTIONS:
            raise ValueError(
                f"{option} is incompatible with similarity stabilization; "
                "evaluate it as a separate translation/border control"
            )


def _delegated_arguments(
    arguments: Sequence[str],
    stabilized_dir: Path,
    output_dir: Path,
    summary_path: Path,
) -> list[str]:
    delegated = list(arguments)
    delegated[0] = str(stabilized_dir)
    delegated = _replace_option(delegated, "--output-dir", str(output_dir))
    delegated = _remove_option(delegated, "--output-json")
    delegated.extend(["--output-json", str(summary_path)])
    if "--no-sequence-cache" not in delegated:
        delegated.append("--no-sequence-cache")
    return delegated


def _option(
    arguments: Sequence[str], option: str, *, required: bool
) -> str | None:
    for index, token in enumerate(arguments):
        if token == option:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("-"):
                raise ValueError(f"{option} requires a value")
            return arguments[index + 1]
        prefix = option + "="
        if token.startswith(prefix):
            value = token[len(prefix) :]
            if not value:
                raise ValueError(f"{option} requires a value")
            return value
    if required:
        raise ValueError(f"{option} is required")
    return None


def _replace_option(arguments: list[str], option: str, value: str) -> list[str]:
    result = list(arguments)
    for index, token in enumerate(result):
        if token == option:
            result[index + 1] = value
            return result
        if token.startswith(option + "="):
            result[index] = f"{option}={value}"
            return result
    raise ValueError(f"{option} is required")


def _remove_option(arguments: list[str], option: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            index += 2
        elif token.startswith(option + "="):
            index += 1
        else:
            result.append(token)
            index += 1
    return result


def _guard_paths(
    proposal_path: Path,
    label_dir: Path,
    output_dir: Path,
    output_json: Path | None,
) -> None:
    proposal = proposal_path.expanduser().resolve()
    labels = label_dir.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if _overlap(output, labels):
        raise ValueError("output directory must be disjoint from seed labels")
    if proposal_path.is_dir():
        collision = _overlap(output, proposal)
    else:
        collision = output == proposal or output in proposal.parents
    if collision:
        raise ValueError("output directory must be disjoint from proposals")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_json is not None:
        artifact = output_json.expanduser().resolve()
        if artifact == proposal or artifact == labels or labels in artifact.parents:
            raise ValueError("output JSON must not overwrite an input")
        if proposal_path.is_dir() and proposal in artifact.parents:
            raise ValueError("output JSON must be disjoint from proposals")


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _write_rows(path: Path, rows: Sequence[Detection]) -> None:
    path.write_text(
        "".join(format_detection(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _publish(prediction_text: dict[str, str], output_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.sim2-stage-{uuid.uuid4().hex}"
    backup = output.parent / f".{output.name}.sim2-backup-{uuid.uuid4().hex}"
    staging.mkdir()
    published = False
    try:
        for name, text in sorted(prediction_text.items()):
            if Path(name).name != name or not name.endswith(".txt"):
                raise ValueError(f"invalid prediction filename: {name}")
            (staging / name).write_text(text, encoding="utf-8")
        had_output = output.exists()
        if had_output:
            os.replace(output, backup)
        try:
            os.replace(staging, output)
            published = True
        except Exception:
            if had_output and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if published and backup.exists():
            shutil.rmtree(backup)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
