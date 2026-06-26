# MMUAD forward-backward candidate prior

The branch-preserving MMUAD pipeline can retain a useful raw, dynamic, merged,
or calibrated candidate while still assigning it a weak per-frame ranker score.
Hard top-1 or Viterbi selection can then commit too early, and the standard
candidate-mixture MAP prior is otherwise frame-local.

`raft-uav-mmuad-forward-backward-prior` adds a soft global path prior before the
robust candidate-mixture smoother. It runs a log-space forward-backward pass over
all candidate frames, using only inference-time information:

- candidate score and learned sigma;
- candidate position and timestamp;
- source and branch switch penalties;
- optional same-track continuation bonus;
- soft speed-gate and time-gap penalties.

The result is a per-frame posterior probability in
`candidate_forward_backward_score`. No truth is required to compute it.

## Candidate-only run

```bash
raft-uav-mmuad-forward-backward-prior \
  --candidate-csv outputs/mmuad_branch_reservoir.csv \
  --output-csv outputs/mmuad_forward_backward_candidates.csv \
  --summary-json outputs/mmuad_forward_backward_summary.json \
  --score-column candidate_reservoir_grid_score \
  --sigma-column predicted_sigma_m \
  --transition-distance-std-m 2 \
  --transition-speed-std-mps 15 \
  --max-speed-mps 80 \
  --source-switch-penalty 0.25 \
  --branch-switch-penalty 0.25
```

## Forward-backward prior plus mixture-MAP

```bash
raft-uav-mmuad-forward-backward-prior \
  --candidate-csv outputs/mmuad_branch_reservoir.csv \
  --output-csv outputs/mmuad_forward_backward_candidates.csv \
  --summary-json outputs/mmuad_forward_backward_summary.json \
  --mixture-output-dir outputs/mmuad_forward_backward_mixture \
  --mixture-top-k 20 \
  --mixture-smoothness-weight 7200 \
  --mixture-huber-delta 1 \
  --mixture-iterations 5
```

For public-validation diagnostics only, add `--mixture-truth-csv`. Do not use
validation truth to tune the prior for a leaderboard submission; choose the
transition and switch penalties with train-only cross-validation.

Useful output columns include:

- `candidate_forward_backward_score`;
- `candidate_forward_backward_rank`;
- `candidate_forward_log_probability`;
- `candidate_backward_log_probability`;
- frame entropy and effective candidate count;
- best previous/next candidate distance, speed, source, branch, and track ID.

The intended ablation compares the existing robust mixture-MAP score against the
forward-backward posterior while keeping the candidate reservoir, learned sigma,
Huber loss, and smoothness settings fixed.
