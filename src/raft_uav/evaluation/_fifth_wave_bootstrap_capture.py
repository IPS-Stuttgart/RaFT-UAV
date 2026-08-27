"""Capture the pristine fifth-wave block-bootstrap implementation."""

from __future__ import annotations

from importlib import import_module


_fifth_wave = import_module("raft_uav.evaluation.fifth_wave_diagnostics")
ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL = _fifth_wave._ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL
