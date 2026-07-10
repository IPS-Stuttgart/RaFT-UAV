# MMUAD branch-seeded multi-start mixture MAP

The robust MMUAD candidate-mixture smoother alternates between candidate
responsibilities and a smooth trajectory update. This makes the objective
non-convex: a single score- or uncertainty-derived initialization can settle on
the wrong raw, dynamic, or source-calibrated candidate branch even when the
branch-preserving reservoir still contains a much better trajectory.

`candidate_mixture_map_multistart` addresses that failure mode without using
truth at inference time. It runs the maintained mixture-MAP smoother from:

- the core configured initialization;
- score-top1 candidates;
- the per-frame coordinate median;
- one seed for each sufficiently represented `candidate_branch`, with a global
  uncertainty-top1 fallback at frames where the branch is absent;
- an optional external initial trajectory.

The selected restart minimizes the final robust mixture negative log evidence
plus the same irregular-time acceleration penalty used by the core smoother.
When truth is supplied, pose metrics are appended to the restart table for
local diagnostics, but they are not used to select the winning restart.

## Example

```bash
python scripts/mmuad_candidate_mixture_multistart.py \
  --candidates-csv outputs/mmuad_branch_reservoir/reservoir_candidates.csv \
  --truth-csv challenge_meta/validation_ref_new_for_your_ref.csv \
  --output-dir outputs/mmuad_branch_multistart \
  --top-k 0 \
  --score-column candidate_reservoir_score \
  --sigma-column predicted_sigma_m_hgb \
  --score-weight 1 \
  --temperature 128 \
  --sigma-log-weight 3 \
  --loss huber \
  --huber-delta 1 \
  --smoothness-weight 7200 \
  --iterations 5 \
  --max-branch-starts 8
```

For a hidden-test run, omit `--truth-csv`. The restart selection remains fully
inference-safe.

## Main outputs

- `mmuad_candidate_mixture_estimates.csv`: selected trajectory;
- `mmuad_candidate_mixture_assignments.csv`: selected final responsibilities;
- `mmuad_candidate_mixture_multistart_summary.csv`: objective and optional
  diagnostic metrics for every restart;
- `mmuad_candidate_mixture_multistart_initializations.csv`: explicit restart
  trajectories;
- `mmuad_candidate_mixture_multistart_summary.json`: selected start and full
  configuration provenance.

## Recommended first comparison

Run the current frozen learned-sigma/Huber configuration once with the ordinary
single start and once with multi-start on exactly the same branch-preserving
reservoir. Compare the final Track 5 scorecard and the restart table. A useful
result is either a lower pose MSE or evidence that all starts converge to the
same objective, which would rule out initialization as the remaining assignment
bottleneck.
