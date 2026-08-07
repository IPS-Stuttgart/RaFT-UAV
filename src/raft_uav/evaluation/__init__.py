"""Evaluation utilities."""

from . import _diagnostics_sequence_scope_patch as _diagnostics_sequence_scope_patch
from . import _golden_metrics_schema_patch as _golden_metrics_schema_patch
from . import (
    _radar_oracle_endpoint_tolerance_patch as _radar_oracle_endpoint_tolerance_patch,
)
from . import _oracle_gap_sequence_scope_patch as _oracle_gap_sequence_scope_patch
from . import _oracle_gap_reused_frame_patch as _oracle_gap_reused_frame_patch
from . import _single_sample_truth_grid_patch as _single_sample_truth_grid_patch

# Oracle-coverage sequence scoping is installed by ``raft_uav.__init__`` after
# baseline initialization. Importing it here would re-enter radar association
# while that module is still being initialized.
