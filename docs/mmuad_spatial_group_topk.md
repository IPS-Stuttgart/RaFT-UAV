# Spatially diverse MMUAD group top-K

The branch-preserving MMUAD candidate pool can contain several origin groups
that are geometrically almost identical. The existing group-first top-K avoids
spending multiple slots on raw/calibrated siblings from one origin group, but a
finite group budget can still be consumed by several nearby groups.

`candidate_mixture_group_spatial_topk` adds a greedy spatial-diversity term to
the inference-safe group score:

```text
normalized_group_score
+ diversity_weight * (1 - exp(-nearest_selected_distance / diversity_scale_m))
```

The distance can be capped with `diversity_cap_m`. A diversity weight of zero
is the score-only group-top-K baseline.

## Example

```bash
raft-uav-mmuad-spatial-group-topk \
  --candidates-csv outputs/mmuad_candidates.csv \
  --output-dir outputs/mmuad_spatial_group_topk \
  --group-top-k 10 \
  --max-siblings-per-group 2 \
  --diversity-weight 0.5 \
  --diversity-scale-m 5 \
  --diversity-cap-m 30 \
  --score-column candidate_reservoir_grid_score \
  --sigma-column predicted_sigma_m \
  --loss huber \
  --huber-delta 1 \
  --smoothness-weight 7200
```

The tool writes the selected candidates, group-selection diagnostics, and the
standard grouped candidate-mixture MAP artifacts. Truth is optional and is
used only for diagnostic metrics.

## Intended train-CV ablation

Freeze the candidate pool, learned uncertainty, Huber loss, temporal prior,
smoothness, and classification features. Select on train CV among:

```text
group_top_k:      5, 10, 20
diversity_weight: 0, 0.25, 0.5, 1.0
diversity_scale_m: 2, 5, 10
```

Primary diagnostics:

- unique-group top-10/top-20 oracle recall;
- selected-group nearest-neighbor distance;
- pose MSE after grouped mixture-MAP;
- per-sequence regressions, especially sequences where useful candidates are
  deeper in the reservoir.

The public-validation row should be evaluated only after selecting the
configuration on train.
