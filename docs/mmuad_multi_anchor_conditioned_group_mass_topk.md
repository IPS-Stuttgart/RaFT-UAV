# Multi-anchor conditioned MMUAD group selection

The maintained posterior-mass group selector can condition its finite physical-
hypothesis budget on one inference-time trajectory anchor. A single anchor can
also be wrong: it may come from a bad local candidate-mixture optimum, a damaged
candidate branch, or an over-smoothed trajectory. In that case, a strong anchor
term can remove the correct deeper hypothesis before grouped Huber mixture-MAP
can recover it.

`candidate_mixture_group_multi_anchor_mass_topk` accepts several trajectory
hypotheses and combines their bounded Huber costs without using truth.

## Aggregation modes

- `min`: a candidate is supported when any anchor is close. This is the most
  recall-oriented mode.
- `softmin`: a smooth any-anchor rule controlled by
  `--anchor-softmin-temperature`.
- `mean`: favors candidate groups supported by anchor consensus.

The aggregate cost changes only the group-selection unary. The final grouped
mixture-MAP objective still uses the configured candidate score, learned sigma,
physical-group correction, robust loss, and trajectory smoothness.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_multi_anchor_mass_topk.py \
  --candidates-csv outputs/mmuad_reservoir/candidates.csv \
  --initial-estimates robust=outputs/robust/estimates.csv \
  --initial-estimates score_top1=outputs/multistart/score_top1.csv \
  --initial-estimates raw_branch=outputs/multistart/raw_branch.csv \
  --output-dir outputs/mmuad_multi_anchor_group_selection \
  --anchor-aggregation softmin \
  --anchor-softmin-temperature 0.5 \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 5 \
  --anchor-cost-cap 4 \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --final-anchor-policy none
```

`--final-anchor-policy none` keeps the anchors selection-only. `first` reuses
the first supplied trajectory for final mixture initialization. `median` builds
a framewise median trajectory from all available interpolated anchors.

## Train-CV ablation

Keep the candidate reservoir, score model, learned sigma, physical grouping,
Huber loss, smoothness, and classification features frozen. Select on train CV:

```text
anchor sets:
  robust + score-top1
  robust + score-top1 + frame-median
  robust + branch starts

aggregation:
  min, softmin, mean

anchor_selection_weight:
  0.0, 0.25, 0.5, 1.0, 2.0

anchor_scale_m:
  2, 5, 10, 20

softmin_temperature:
  0.1, 0.25, 0.5, 1.0

final_anchor_policy:
  none, first, median
```

Report pose MSE, P95, selected physical-group budget, group top-K oracle recall,
anchor-support coverage, anchor-cost disagreement, and runtime. The expected
benefit is concentrated on frames where a single initialization is confidently
wrong but another inference-safe trajectory hypothesis remains close to the
correct candidate group.
