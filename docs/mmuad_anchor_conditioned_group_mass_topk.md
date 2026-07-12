# Anchor-conditioned posterior-mass group selection

The MMUAD branch-preserving candidate pool can contain the correct physical
hypothesis deeper in the score ordering. The posterior-mass group selector
allocates a framewise group budget from score and learned uncertainty, but its
selection unary is state independent. It can therefore remove a low-score group
that is highly consistent with an available inference-time trajectory
initialization before robust mixture-MAP sees it.

`candidate_mixture_group_anchor_mass_topk` adds a bounded Huber cost from each
candidate to an interpolated initial trajectory **only for group selection**.
The final grouped mixture-MAP run still uses the original candidate score,
learned sigma, group multiplicity correction, Huber mixture loss, and smoothness
objective.

The selection unary is

```text
score_weight * normalized_score / temperature
- sigma_log_weight * log(predicted_sigma)
- anchor_selection_weight * huber(distance_to_anchor / anchor_scale)
```

The anchor cost is capped so a locally poor initialization cannot suppress a
physical hypothesis without bound. Frames without anchor support are neutral by
default. Ground truth is optional and is never used for selection.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_anchor_mass_topk.py \
  --candidates-csv outputs/mmuad_reservoir/candidates.csv \
  --initial-estimates-csv outputs/mmuad_baseline/estimates.csv \
  --output-dir outputs/mmuad_anchor_mass_group_topk \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --posterior-temperature 1.0 \
  --uniform-posterior-blend 0.02 \
  --max-siblings-per-group 2 \
  --diversity-weight 0.5 \
  --diversity-scale-m 5 \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 10 \
  --anchor-huber-delta 1 \
  --anchor-cost-cap 4 \
  --anchor-time-tolerance-s 0.5 \
  --score-column candidate_pair_forward_backward_score \
  --sigma-column predicted_sigma_m \
  --loss huber \
  --huber-delta 1 \
  --smoothness-weight 7200
```

## Train-CV ablation

Freeze the candidate reservoir, score model, learned sigma, physical group
definition, posterior-mass budget settings, final Huber mixture objective, and
classification features. Select only the anchor-conditioning parameters on
train CV:

```text
anchor_selection_weight: 0.00, 0.25, 0.50, 1.00, 2.00
anchor_scale_m:           2, 5, 10, 20
anchor_cost_cap:          1, 2, 4
```

Compare against the state-independent posterior-mass selector using:

- pose MSE and P95;
- top-3/top-5/top-10/top-20 physical-group oracle recall;
- selected group budget distribution;
- fraction of frames with matched anchor support;
- error stratified by anchor distance and sequence.

A useful result should improve top-K physical-hypothesis recall or pose MSE
without requiring a larger maximum group budget. If only very large anchor
weights help, inspect whether the method is merely copying the initialization
rather than improving candidate assignment.
