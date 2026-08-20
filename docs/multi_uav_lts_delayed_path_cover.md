# Delayed path-cover association milestone

## Scientific question

The permissive proposal-bank audit shows that detector proposals are already close
to an oracle ceiling, while the current tracker remains far below that ceiling. The
next falsifiable hypothesis is therefore:

> A material part of the remaining HOTA/IDF1 gap is caused by irreversible
> frame-local association decisions at crossings and short occlusions.

The current proposal graph first forms reciprocal nearest-neighbour anchor
tracklets. Later global linking can reconnect tracklets, but it cannot split an
incorrect anchor that has already joined two observations.

## Experimental change

The delayed path-cover candidate removes that irreversible first stage without
changing the detector or the benchmark protocol.

1. Canonicalize the same permissive proposal bank.
2. Keep every retained proposal atomic instead of forming local anchor tracklets.
3. Attach exact frame-one seed identities with the existing seed matcher.
4. For every adjacent-frame candidate link, compute a bounded continuation potential
   from the next two frames. The potential measures constant-residual-motion
   consistency after common image/swarm motion compensation. Only the relative
   penalty versus the best local continuation is added, so delayed evidence
   changes ambiguous ordering without globally making all links harder.
5. Retain only the best few successors per source proposal and solve all optional
   short links with the existing sparse minimum-cost path-cover assignment.
6. Collapse those paths to micro-tracklets.
7. Apply the existing velocity-aware long-gap linker, interpolation controls, and
   late-birth materializer unchanged.

With the default `--delayed-max-gap 0`, the new stage replaces only the old
frame-to-frame commitment. Long-gap reacquisition remains the existing method.
This keeps the ablation interpretable and bounds compute on the 102-sequence
proposal bank.

## First tournament candidates

Use only two new candidates initially:

```text
graph_delayed_path_cover
  --enable-delayed-path-cover
  --delayed-max-gap 0
  --delayed-lookahead-frames 2
  --delayed-successors-per-frame 3
  --delayed-continuation-weight 0.75

graph_delayed_common_motion
  [same delayed controls]
  --enable-common-motion
  --common-motion-min-pairs 4
  --common-motion-max-normalized-step 8.0
  --common-motion-max-normalized-residual 1.5
```

Do not tune a large grid before these two rows establish that delayed association
has signal.

## Acceptance criterion

Keep the existing guarded 102-sequence tournament unchanged. A delayed candidate
is scientifically interesting only if it clears all existing gates versus `raw`:

- positive held-out `CODABENCH_HOTA` gain;
- positive paired-bootstrap lower bound under the configured gate;
- no unacceptable MOTA or IDF1 regression;
- no excessive worst-scenario regression; and
- complete 102-sequence coverage and provenance.

The expected diagnostic signature is primarily higher association quality/IDF1
and fewer identity switches, rather than a gain obtained mainly by adding rows.
Report canonical `AssA`, `DetA`, ID switches, output row count, seeded paths, and
confirmed births alongside the Codabench objective.

## Synthetic regression

The focused test contains two UAVs that approach and cross. The existing local
anchor cost prefers the spatially nearest proposal at the crossing and swaps the
identities. Two-frame future evidence makes the constant-velocity continuation
unambiguous, and the delayed path-cover candidate preserves both seed identities.

This is not evidence of benchmark improvement; it is a regression proving that
the new mechanism can solve the specific failure mode it was introduced for.

## Interpretation

- **Delayed beats raw/current graph mainly in AssA/IDF1:** continue toward a
  learned edge likelihood or a larger delayed/MHT formulation.
- **Only delayed + common motion wins:** camera/swarm-motion separation is a key
  part of the association model and should be made native rather than optional.
- **HOTA improves but IDF1/MOTA fail:** the continuation potential is too
  permissive; tighten successor pruning or continuation weight before adding
  more model complexity.
- **No delayed candidate beats raw:** stop pursuing first-order geometric path
  cover. The next association milestone should then be a true multi-hypothesis
  or appearance-assisted model rather than a parameter sweep.
