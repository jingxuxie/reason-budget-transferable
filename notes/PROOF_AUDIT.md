# Proof audit — 16 September 2026

All proofs are written in the manuscript and have been self-checked. No independent referee or formal proof assistant has verified them.

| Claim | Proof mechanism | Implementation / check |
|---|---|---|
| Scalar boundary | Pointwise dominance; pair crossing; mixture balancing | Crossing and noncrossing controls |
| O(1/epsilon) regimes | Envelope rises by > epsilon at every switch | `lazy_selector`; 14,700 finite pair checks |
| Matching-order lower example | Alternating steps of height 2 epsilon | Five explicit epsilon tests |
| Exact robust gap | Selected lower profile, competitor upper profile | `robust_gaps`; 500 enumerated rectangles |
| Minimum interval compiler | Farthest-feasible-prefix exchange proof | Independent DP; 16,807 Boolean matrices |
| DKW validity after capping | Eligible samples are initial iid prefixes | Increasing caps rejected; no selective pooling |
| Finite termination | Chosen arm width > epsilon; all past samples eligible | Count-bound tests, batched overshoot checks |
| Matched full-horizon coupling | New tail upper bound cannot improve active-prefix bound | 40 random and 10 terminal-aware coupled tests |
| One-regime work lower bound | Swapped Bernoulli outcomes only at H; change of measure | Analytic proof; late-success control |
| Unsupported tails | Adversarial monotone extensions | Analytic deterministic-decision result |
| Hazard = ECDF | Product telescoping at common cap | 10 randomized identities; full suite max error 1.11e-16 |

Critical qualifications:

1. The complexity lower bound is for deterministic selectors. At gap exactly 2 epsilon, a 50/50 mixture is epsilon-optimal.
2. All-submenu ranking sufficiency differs from existence of a common maximizer in one full menu.
3. Confidence-band exactness is for the rectangular monotone class, not an arbitrary jointly constrained model.
4. The m*N rollout bound is on the shared coverage event. Batches overshoot by at most B-1 per candidate.
5. The coupling theorem assumes identical tie rules, latent streams, and predictable blocks. Returned schedules can differ at retired deadlines.
6. The robust formula excludes the selected candidate from its own competitors.
7. Actual work is min(termination time, cap), not min(success time, cap) when failure is detectable.
8. Feature-generalization experiments do not inherit repeated-candidate certificates.
9. Floating feasibility tolerance is 1e-12; empty-band detection uses 1e-10. Mathematical statements use exact arithmetic.
10. Numerical tests are falsification checks, not substitutes for proofs or evidence of novelty.
