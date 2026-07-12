# Spatial posterior-mass group top-K

The existing posterior-mass selector computes a group budget from groups sorted
by posterior probability and then applies spatial-diversity selection. With a
non-zero diversity weight, the selected spatial prefix can differ from the
score-sorted prefix and retain less posterior mass than the configured target.

`candidate_mixture_group_spatial_mass_topk` first obtains the bounded spatial
ordering and then selects the smallest prefix of that **actual ordering** whose
posterior mass reaches the target. It reports both the ideal score-order budget
and the operational spatial-order budget.

```bash
python scripts/mmuad_candidate_mixture_group_spatial_mass_topk.py \
  --candidates-csv outputs/reservoir/scored_candidates.csv \
  --output-dir outputs/spatial_mass_group_topk \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --posterior-temperature 1.0 \
  --uniform-posterior-blend 0.02 \
  --diversity-weight 0.5 \
  --diversity-scale-m 5 \
  --score-column candidate_reservoir_grid_score \
  --sigma-column predicted_sigma_m \
  --loss huber \
  --huber-delta 1 \
  --smoothness-weight 7200
```

Key diagnostics in the selected-candidate CSV and summary JSON are:

- actual and ideal selected group budgets;
- budget expansion caused by spatial diversity;
- posterior mass retained by the actual selected prefix;
- whether the target mass was reached under the maximum budget;
- posterior shortfall, entropy, and effective group count.

A train-CV ablation should freeze the reservoir, learned sigma model, Huber
loss, smoother, classification features, and hypothesis-group definition. Tune:

```text
target_posterior_mass:    0.80, 0.90, 0.95, 0.99
min_group_top_k:          1, 3, 5
max_group_top_k:          10, 20, 30
posterior_temperature:    0.5, 1.0, 2.0
uniform_posterior_blend:  0.00, 0.02, 0.05
diversity_weight:         0.25, 0.5, 1.0
```

Compare pose MSE, full-pool oracle recall, budget expansion, target-mass reach
rate, and runtime against the original posterior-mass and fixed spatial group
top-K selectors.
