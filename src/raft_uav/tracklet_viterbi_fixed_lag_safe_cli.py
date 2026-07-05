"""Safe CLI shim for fixed-lag tracklet Viterbi.

This module remains the installed console-script target. The fixed-lag duration
is validated by the canonical implementation when a fixed-lag run actually reads
it, so argument-only commands such as ``--help`` are not rejected because of an
unrelated fixed-lag environment variable.
"""

from __future__ import annotations

import math
import os

from raft_uav import tracklet_viterbi_fixed_lag_cli as _impl

_FIXED_LAG_ENV = _impl._FIXED_LAG_ENV


def _validated_fixed_lag_s_from_env() -> float:
    value = os.environ.get(_FIXED_LAG_ENV)
    if value is None or value.strip() == "":
        return _impl._DEFAULT_FIXED_LAG_S
    try:
        lag_s = float(value)
    except ValueError as exc:
        raise ValueError(f"{_FIXED_LAG_ENV} must be finite and positive") from exc
    if not math.isfinite(lag_s) or lag_s <= 0.0:
        raise ValueError(f"{_FIXED_LAG_ENV} must be finite and positive")
    return lag_s


def main(argv: list[str] | None = None) -> int:
    """Run the canonical fixed-lag CLI."""

    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
