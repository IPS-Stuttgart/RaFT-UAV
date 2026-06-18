#!/usr/bin/env python
"""Build a repository-local MMUAD/UG2 result leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.leaderboard import (
    build_mmuad_leaderboard,
    load_leaderboard_config,
    write_leaderboard_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local MMUAD/UG2-style leaderboard from result/truth files. "
            "This uses RaFT-UAV's transparent evaluator, not the closed Codabench runtime."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank-metric", default="pose_mse_loss_m2")
    parser.add_argument("--stem", default="mmuad_local_leaderboard")
    args = parser.parse_args(argv)

    entries = load_leaderboard_config(args.config)
    result = build_mmuad_leaderboard(entries, rank_metric=args.rank_metric)
    paths = write_leaderboard_artifacts(result, output_dir=args.output_dir, stem=args.stem)
    print("mmuad_local_leaderboard=ok")
    print(f"method_count={len(result.rows)}")
    if not result.rows.empty:
        best = result.rows.iloc[0]
        print(f"best_method={best['method']}")
        if args.rank_metric in best:
            print(f"best_{args.rank_metric}={best[args.rank_metric]}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
