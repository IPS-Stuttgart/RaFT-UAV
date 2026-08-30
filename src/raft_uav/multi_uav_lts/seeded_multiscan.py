"""Global seeded-label multi-scan association by dual-decomposed path optimization.

Each supplied frame-one identity receives a sequence-level path through the
proposal DAG. Shared proposal conflicts are resolved by Lagrange prices. Late
births are recovered only from persistent unclaimed paths. Optional geometry
maps let association operate in stabilized coordinates while preserving source
boxes in the exported predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from ._full_stack_io import prediction_files, read_rows, write_json, write_rows
from ._records import Detection

Affinity = Callable[[Detection, Detection], float]
Geometry = Callable[[Detection], Detection]


@dataclass(frozen=True)
class MultiScanConfig:
    max_gap: int = 12
    motion_scale: float = 5.0
    size_scale: float = 0.7
    miss_cost: float = 0.22
    confidence_reward: float = 0.8
    start_gate: float = 8.0
    dual_iterations: int = 16
    dual_step: float = 0.8
    min_birth_hits: int = 5
    min_birth_span: int = 4
    min_birth_confidence: float = 0.01
    affinity_weight: float = 0.0


def edge_cost(
    a: Detection,
    b: Detection,
    config: MultiScanConfig,
    affinity: Affinity | None = None,
    geometry: Geometry | None = None,
) -> float:
    gap = b.frame_id - a.frame_id
    if gap <= 0 or gap > config.max_gap:
        return math.inf
    geometric_a = a if geometry is None else geometry(a)
    geometric_b = b if geometry is None else geometry(b)
    scale = max(
        math.sqrt(geometric_a.width * geometric_a.height),
        math.sqrt(geometric_b.width * geometric_b.height),
        2.0,
    )
    distance = math.hypot(
        geometric_b.center_x - geometric_a.center_x,
        geometric_b.center_y - geometric_a.center_y,
    ) / (scale * math.sqrt(gap))
    size = abs(
        math.log(max(geometric_b.width, 1e-6) / max(geometric_a.width, 1e-6))
    ) + abs(math.log(max(geometric_b.height, 1e-6) / max(geometric_a.height, 1e-6)))
    cost = (
        (distance / config.motion_scale) ** 2
        + (size / config.size_scale) ** 2
        + config.miss_cost * (gap - 1)
        - config.confidence_reward
        * math.log(max(b.confidence, 1e-5) / 1e-5)
        / math.log(100000.0)
    )
    if affinity is not None:
        cost -= config.affinity_weight * float(affinity(a, b))
    return cost


def _best_path(
    anchor: Detection,
    nodes: list[Detection],
    prices: dict[tuple[int, int], float],
    config: MultiScanConfig,
    affinity: Affinity | None = None,
    geometry: Geometry | None = None,
    forbid_start: bool = False,
) -> list[Detection]:
    ordered = sorted(nodes, key=lambda row: (row.frame_id, row.object_id))
    costs = np.full(len(ordered), math.inf)
    parent = np.full(len(ordered), -1, dtype=int)
    for index, node in enumerate(ordered):
        start = edge_cost(anchor, node, config, affinity, geometry)
        if not forbid_start and math.isfinite(start):
            costs[index] = start + prices.get((node.frame_id, node.object_id), 0.0)
        for predecessor in range(index):
            if not math.isfinite(costs[predecessor]):
                continue
            transition = edge_cost(
                ordered[predecessor],
                node,
                config,
                affinity,
                geometry,
            )
            candidate = (
                costs[predecessor]
                + transition
                + prices.get((node.frame_id, node.object_id), 0.0)
            )
            if math.isfinite(transition) and candidate < costs[index]:
                costs[index] = candidate
                parent[index] = predecessor
    if not ordered or not np.isfinite(costs).any():
        return [anchor]
    terminal = costs - config.miss_cost * 0.5
    end = int(np.argmin(terminal))
    path: list[Detection] = []
    while end >= 0:
        path.append(ordered[end])
        end = int(parent[end])
    return [anchor, *reversed(path)]


def _prefix_claims(
    label: int,
    path: list[Detection],
    config: MultiScanConfig,
    affinity: Affinity | None,
    geometry: Geometry | None,
) -> list[tuple[tuple[int, int], float, int, Detection]]:
    cumulative = 0.0
    claims = []
    previous = path[0]
    for node in path[1:]:
        cumulative += edge_cost(previous, node, config, affinity, geometry)
        claims.append(((node.frame_id, node.object_id), cumulative, label, node))
        previous = node
    return claims


def associate_sequence(
    proposals: list[Detection],
    seeds: list[Detection],
    config: MultiScanConfig = MultiScanConfig(),
    affinity: Affinity | None = None,
    geometry: Geometry | None = None,
) -> tuple[list[Detection], dict]:
    nodes = [proposal for proposal in proposals if proposal.frame_id > 1]
    prices: dict[tuple[int, int], float] = {}
    paths = {seed.object_id: [seed] for seed in seeds}
    iterations = 0
    for iteration in range(config.dual_iterations):
        iterations = iteration + 1
        paths = {
            seed.object_id: _best_path(
                seed,
                nodes,
                prices,
                config,
                affinity,
                geometry,
            )
            for seed in seeds
        }
        usage: dict[tuple[int, int], list[Detection]] = {}
        for path in paths.values():
            for node in path[1:]:
                usage.setdefault((node.frame_id, node.object_id), []).append(node)
        conflicts = {key: values for key, values in usage.items() if len(values) > 1}
        if not conflicts:
            break
        step = config.dual_step / math.sqrt(iteration + 1)
        for key, values in conflicts.items():
            prices[key] = prices.get(key, 0.0) + step * (len(values) - 1)

    claims: dict[tuple[int, int], tuple[float, int, Detection]] = {}
    for label, path in paths.items():
        for key, score, claim_label, node in _prefix_claims(
            label,
            path,
            config,
            affinity,
            geometry,
        ):
            if key not in claims or score < claims[key][0]:
                claims[key] = (score, claim_label, node)

    output: list[Detection] = []
    claimed: set[tuple[int, int]] = set()
    for seed in seeds:
        output.append(seed)
        for node in paths.get(seed.object_id, ())[1:]:
            key = (node.frame_id, node.object_id)
            if claims.get(key, (None, None, None))[1] == seed.object_id:
                output.append(replace(node, object_id=seed.object_id))
                claimed.add(key)

    remaining = [
        node for node in nodes if (node.frame_id, node.object_id) not in claimed
    ]
    next_id = max([seed.object_id for seed in seeds], default=0) + 1
    births = 0
    while remaining:
        anchor = min(
            remaining,
            key=lambda row: (row.frame_id, -row.confidence, row.object_id),
        )
        pseudo = replace(anchor, frame_id=max(1, anchor.frame_id - 1))
        path = _best_path(
            pseudo,
            [node for node in remaining if node is not anchor],
            prices,
            config,
            affinity,
            geometry,
        )
        chain = [anchor, *path[1:]]
        mean_confidence = float(np.mean([node.confidence for node in chain]))
        span = chain[-1].frame_id - chain[0].frame_id
        if (
            len(chain) >= config.min_birth_hits
            and span >= config.min_birth_span
            and mean_confidence >= config.min_birth_confidence
        ):
            output.extend(replace(node, object_id=next_id) for node in chain)
            next_id += 1
            births += 1
            keys = {(node.frame_id, node.object_id) for node in chain}
            remaining = [
                node
                for node in remaining
                if (node.frame_id, node.object_id) not in keys
            ]
        else:
            remaining.remove(anchor)
    diagnostics = {
        "seed_count": len(seeds),
        "proposal_count": len(proposals),
        "output_count": len(output),
        "late_birth_count": births,
        "conflict_price_count": len(prices),
        "dual_iterations_run": iterations,
        "stabilized_geometry": geometry is not None,
    }
    return output, diagnostics


def associate_directory(
    proposal_dir: Path,
    seed_dir: Path,
    output_dir: Path,
    sequence_names: list[str] | None = None,
    config: MultiScanConfig = MultiScanConfig(),
    affinity: Affinity | None = None,
) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for proposal_file in prediction_files(proposal_dir):
        name = proposal_file.stem
        if allowed is not None and name not in allowed:
            continue
        seed_file = seed_dir / proposal_file.name
        if not seed_file.is_file():
            raise FileNotFoundError(seed_file)
        rows, diagnostics = associate_sequence(
            read_rows(proposal_file),
            read_rows(seed_file),
            config,
            affinity,
        )
        write_rows(output_dir / proposal_file.name, rows)
        summary[name] = diagnostics
    return {"sequences": summary, "config": config.__dict__}


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_dir", type=Path)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument("--max-gap", type=int, default=12)
    parser.add_argument("--dual-iterations", type=int, default=16)
    parser.add_argument("--min-birth-hits", type=int, default=5)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MultiScanConfig(
        max_gap=args.max_gap,
        dual_iterations=args.dual_iterations,
        min_birth_hits=args.min_birth_hits,
    )
    result = associate_directory(
        args.proposal_dir,
        args.seed_dir,
        args.output_dir,
        _names(args.sequence_list),
        config,
    )
    if args.summary_json:
        write_json(args.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
