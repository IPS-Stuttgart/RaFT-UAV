# MMUAD hypothesis-group top-K mixture selection

Raw and train-calibrated candidate branches can contain multiple coordinate
hypotheses for the same physical point-cloud cluster. A flat row-level top-K can
therefore spend several slots on raw/calibrated siblings and exclude a distinct
cluster that the learned-sigma Huber mixture smoother needs.

This experiment ranks **origin groups** first and then keeps a bounded number of
siblings from each selected group. It complements the existing grouped-mixture
log-size correction:

- group-size correction prevents duplicated branches from receiving excess
  mixture probability mass;
- group-first top-K prevents duplicated branches from consuming the finite
  candidate budget before mixture inference starts.

The selector is inference-safe. It uses score, learned sigma, and the origin-group
metadata only. Ground truth is optional and is used only by the downstream metric
reporter.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_topk.py \
  --candidates-csv outputs/mmuad_source_branch_reservoir/reservoir.csv \
  --output-dir outputs/mmuad_group_topk \
  --group-top-k 10 \
  --max-siblings-per-group 2 \
  --group-score-mode max \
  --score-column candidate_risk_adjusted_score \
  --sigma-column predicted_sigma_m \
  --temperature 128 \
  --smoothness-weight 7200 \
  --loss huber \
  --huber-delta 1
```

For a local diagnostic, add `--truth-csv`. Do not use public-validation truth to
select `group-top-k`, sibling caps, or score mode.

## Group ranking

For each candidate, the selector uses the state-independent mixture unary term:

```text
score_weight * normalized_score / temperature
- sigma_log_weight * log(predicted_sigma)
```

A group's score is either:

- `max`: its strongest sibling utility; or
- `logmeanexp`: smooth mean evidence without a group-size advantage.

The selected rows are passed to the existing origin-group-corrected Huber
mixture-MAP with the core row-level top-K disabled, so a second truncation cannot
undo the group coverage guarantee.

## Intended ablation

Freeze the candidate pool, learned-sigma model, Huber loss, temporal prior,
smoothness, classification features, and all other inference settings. Select on
train CV among:

```text
group_top_k:            5, 10, 20
max_siblings_per_group: 1, 2, 0 (all)
group_score_mode:       max, logmeanexp
```

Compare against the ordinary grouped mixture with the same nominal candidate
budget. Report:

- top-5/top-10/top-20 oracle recall;
- unique origin groups retained per frame;
- candidate rows retained per frame;
- train-CV and public-validation pose MSE;
- by-sequence deltas, especially sequences where raw and translated branches
  disagree.

The method succeeds only if it improves unique-group recall or pose without
changing the full-pool oracle ceiling. A negative result would show that sibling
crowding is not the remaining top-K bottleneck.
