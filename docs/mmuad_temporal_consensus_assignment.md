# MMUAD one-to-one temporal consensus

The standard temporal-consensus scorer independently chooses the nearest
candidate in each adjacent frame. In a dense branch-preserving pool, several
raw, calibrated, or translated copies can therefore claim support from the
same observation.

`raft-uav-mmuad-temporal-consensus-assigned` adds a one-to-one assignment
option. For each adjacent frame pair it solves a gated linear assignment with
explicit unmatched slots. One neighboring candidate can support at most one
current candidate in each direction.

Use manually specified weights:

```bash
raft-uav-mmuad-temporal-consensus-assigned \
  --candidate-csv outputs/mmuad/candidates.csv \
  --output-csv outputs/mmuad/candidates_temporal_assigned.csv \
  --summary-json outputs/mmuad/temporal_assigned_summary.json \
  --assignment-mode one-to-one \
  --max-time-gap-s 2 \
  --max-speed-mps 60
```

The command can also reuse the frozen configuration written by the temporal
consensus train-CV selector:

```bash
raft-uav-mmuad-temporal-consensus-assigned \
  --candidate-csv outputs/mmuad/validation_candidates.csv \
  --config-json \
    outputs/mmuad/temporal_consensus_train_cv/mmuad_temporal_consensus_train_selected_config.json \
  --output-csv outputs/mmuad/validation_temporal_assigned.csv \
  --assignment-mode one-to-one
```

Additional diagnostics include:

- `candidate_temporal_assignment_mode`
- directional assignment-match indicators
- directional matched track IDs
- assignment-match counts in the summary JSON

`--assignment-mode nearest` preserves the original independent-nearest
behavior and is useful as a controlled ablation. The one-to-one mode is
truth-free, but it should still be compared with nearest matching through
training-sequence cross-validation before being used for validation or test
submissions.
