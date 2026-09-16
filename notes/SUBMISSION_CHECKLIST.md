# Submission checklist

## Completed in this draft

- Fixed statistical and decision interface, with exact retained verification and actual termination-time costs.
- Full proofs for scalar boundary, regime bounds, robust certificate/compiler, prefix-safe confidence, finite termination, coupling, learning-cost separation, and unsupported tails.
- Executable implementation, 133 tests, exhaustive finite checks, and actual symbolic execution audit.
- Strong budget-conditioned and hazard predictive baselines with program-level holdout splits.
- Paired matched-acquisition capping comparison and late/noncrossing negative controls.
- Explicit reporting of the symbolic early-failure cost correction and learned-hazard baseline strength.
- Reproducible CPU commands, measured CSV/JSON outputs, manuscript source, and references.

## Required before calling it submission-ready

1. Independent proof audit, especially the active-prefix coupling, optional stopping/interface assumptions, and precise decision-complexity scope.
2. Independent novelty comparison against utilitarian algorithm configuration, including practical COUP variants and deadline-utility robustness. Decide whether the combined result is sufficiently distinct.
3. At least one stronger natural reasoning or algorithm-configuration setting. It can remain low-compute, but should not be another generator tuned to the theorem.
4. Compare against published practical capped-configuration methods, not only a matched race and round-robin. Attribute generic acquisition gains correctly.
5. Either justify the fixed-menu scope as the central contribution or add a genuine feature-transfer theorem. Current held-out critics have no distribution-free certificate.
6. Measure controller/prefix/feature overhead and sensitivity to batch size. Report rollout-work savings separately from wall-clock performance.
7. Obtain user author/affiliation decisions, format to the appropriate current venue template, and complete its reproducibility/ethics checklist. No submission has been made.

Do not strengthen claims by relabeling the stalled-failure ablation as the executable symbolic benchmark, omitting the hazard baseline, or calling an interior deadline extrapolation.
