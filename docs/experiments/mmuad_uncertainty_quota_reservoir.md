# MMUAD uncertainty-aware source/branch quota

The source/branch reservoir is primarily score driven. A candidate can therefore
be removed before mixture-MAP even when the train-learned uncertainty model gives
it a small predicted sigma. `candidate_uncertainty_quota` augments the reservoir
with the lowest-sigma candidates from each `(source, candidate_branch)` cell before
the final frame cap.

This is inference safe: only candidate metadata and predicted uncertainty are
used. Ground-truth columns are ignored.

## Novelty-aware quota

A plain low-sigma quota can spend its complete budget on a candidate that is
already retained by the score-driven reservoir, or on several nearly co-located
hypotheses. Set `uncertainty_novelty_radius_m` to a positive value to make the
quota additive. The selector then scans candidates by increasing predicted sigma
and adds only rows that are at least the configured distance from:

- candidates already retained by the score/source-branch reservoir; and
- earlier uncertainty-quota picks from the same frame, source, and branch.

The default radius is zero and preserves the historical lowest-sigma behavior.
The novelty radius must be selected on training folds and frozen for validation
or hidden-test inference.

## Suggested ablation

Keep the trained sigma model, Huber loss, temporal priors, smoothing, and all
other candidate-pool settings frozen. Compare:

1. score/source/branch reservoir only;
2. uncertainty quota `top_n=1`, novelty radius `0`;
3. uncertainty quota `top_n=1`, novelty radius `0.5`, `1`, and `2 m`;
4. uncertainty quota `top_n=2` with the train-selected novelty radius when the
   frame budget allows it.

Use train CV for selection. Report pose MSE together with top-10/top-20 oracle
recall, the fraction of final candidates selected by the uncertainty quota, and
the selected novelty-distance distribution. The intended benefit is recall:
preserve a reliable moderate-score hypothesis without replacing the existing
high-score and spatial-diversity quotas, while avoiding redundant quota slots.
