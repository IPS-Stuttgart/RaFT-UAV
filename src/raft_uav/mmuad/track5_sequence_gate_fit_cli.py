"""Entry point for sequence-gate fitting."""

from __future__ import annotations

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    from raft_uav.mmuad.track5_sequence_gate_fit import main as impl_main

    return impl_main(argv)
