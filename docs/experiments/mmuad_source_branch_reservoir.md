# MMUAD source-by-branch reservoir

The branch-preserving candidate reservoir retains top candidates per source and
per candidate branch. Those two marginal quotas can still omit a complete
source/branch intersection. For example, a translated Livox candidate may be
neither the highest-scored Livox row nor the highest-scored translated row.

`candidate_source_branch_reservoir` adds an inference-safe quota for every
`(source, candidate_branch)` cell before the final frame cap. It uses only
candidate metadata, geometry, and configured score columns. Truth is optional
and is used only to write oracle-recall diagnostics.

When `per_source_branch_top_n > 1`, score-only quotas can spend several slots on
near-identical clusters from the same source/branch cell. The optional spatial
quota term greedily combines normalized within-cell score with distance from
already retained candidates:

```text
utility = normalized_score
        + diversity_weight * (1 - exp(-min_distance / diversity_scale_m))
```

The distance term is capped before conversion to utility. Setting
`source_branch_diversity_weight=0` preserves the previous score-only behavior.

## Suggested ablation

Keep the learned sigma model, pair prior, Huber loss, mixture settings, and
train-CV protocol frozen. Compare:

1. the current risk-adjusted branch reservoir;
2. the same reservoir with `per_source_branch_top_n=1`;
3. `per_source_branch_top_n=2` with score-only selection;
4. `per_source_branch_top_n=2` with spatial diversity weights such as
   `0.25`, `0.5`, and `1.0`.

Use a frame cap large enough to retain the intended cells; start with 40.
Select the quota and diversity settings on train CV. Inspect pose MSE, top-K
oracle recall, and `source_branch_selected_min_distance_*` diagnostics.

```bash
python scripts/mmuad_candidate_source_branch_reservoir.py \
  --candidates-csv outputs/mmuad_candidates_risk_scored.csv \
  --output-reservoir-csv outputs/mmuad_source_branch_reservoir.csv \
  --summary-json outputs/mmuad_source_branch_reservoir_summary.json \
  --score-column candidate_risk_adjusted_score \
  --fallback-score-column ranker_score \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --per-source-branch-top-n 2 \
  --source-branch-diversity-weight 0.5 \
  --source-branch-diversity-scale-m 10 \
  --source-branch-distance-cap-m 50 \
  --max-candidates-per-frame 40
```

For train/public-validation diagnostics, add the truth and oracle output paths:

```bash
  --truth-csv <truth.csv> \
  --oracle-frame-csv <oracle_frames.csv> \
  --oracle-summary-csv <oracle_summary.csv> \
  --oracle-by-sequence-csv <oracle_by_sequence.csv>
```

The primary success criterion is improved top-10/top-20 oracle recall without a
large increase in candidate count. Final pose settings must still be selected
on train only before one public-validation confirmation.
