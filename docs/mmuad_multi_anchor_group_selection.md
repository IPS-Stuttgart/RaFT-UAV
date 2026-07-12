# MMUAD multi-anchor physical-group selection

The anchor-conditioned posterior-mass selector uses one inference-time trajectory
to protect low-unary physical hypotheses before grouped mixture-MAP. That can be
too brittle when several plausible initial trajectories exist: one anchor may
preserve its own mode while suppressing another mode that would have been useful
to the robust smoother.

`candidate_mixture_group_multi_anchor_mass_topk.py` accepts several named anchor
trajectories and aggregates their bounded Huber costs only for the finite group
selection stage. The final grouped mixture-MAP retains the original candidate
scores, learned sigmas, multiplicity correction, Huber loss, and smoothness
objective.

## Aggregation policies

- `minimum`: preserve a candidate when it is coherent with any anchor;
- `softmin`: smoothly interpolate between best-anchor support and consensus;
- `mean`: require broader agreement across the available anchors.

Unmatched anchors are excluded from the aggregate. Frames unsupported by every
anchor remain neutral by default or can fail with `--missing-anchor-policy error`.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_multi_anchor_mass_topk.py \
  --candidates-csv /path/to/reservoir_candidates.csv \
  --anchor-csv baseline=/path/to/baseline_estimates.csv \
  --anchor-csv translated=/path/to/translated_estimates.csv \
  --anchor-csv multistart=/path/to/multistart_estimates.csv \
  --aggregation minimum \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 10 \
  --anchor-cost-cap 4 \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --hypothesis-group-column origin_row \
  --output-dir outputs/mmuad_multi_anchor_group_selection
```

A separate `--final-initial-estimates-csv` may initialize the downstream
mixture-MAP. Selection anchors are deliberately not averaged into one final
trajectory.

## Train-CV ablation

Freeze the candidate reservoir, score model, learned sigma, physical grouping,
and final Huber mixture objective. Select on train folds:

```text
aggregation:             minimum, softmin, mean
anchor_selection_weight: 0.0, 0.25, 0.5, 1.0, 2.0
anchor_scale_m:           2, 5, 10, 20
anchor_cost_cap:          1, 2, 4
softmin_temperature:      0.25, 0.5, 1.0, 2.0
```

Report pose MSE, P95, selected physical-group count, full-pool and selected-pool
oracle recall, matched-anchor coverage, and best-anchor attribution. The method
is inference-safe: validation/test truth is optional and is not used for group
selection.
