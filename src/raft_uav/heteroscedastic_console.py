"""Console-script adapter for the heteroscedastic baseline wrapper."""

from __future__ import annotations

import sys

from raft_uav.heteroscedastic_cli import main as _main


def main() -> int:
    """Forward command-line arguments to :mod:`raft_uav.heteroscedastic_cli`."""

    return _main(sys.argv[1:])
