# Research log — 16 September 2026

The repository began empty. Work proceeded from a fixed-menu formalization to proofs, implementations, randomized/exhaustive checks, CPU experiments, and a full manuscript.

## Decisions that changed the project

**Representation is not prediction.** Predicting survival curves is established. The draft instead studies the decision schedule induced across budgets and its learning cost. The schedule is menu-specific, not a universal compressed prefix representation.

**Regime complexity is not sample complexity.** A lazy envelope proof gives O(1/epsilon) regimes, but a swapped late-success construction still needs order H/epsilon^2 work even for a one-regime optimum.

**Censoring requires a sampling argument.** Arbitrary adaptive caps do not automatically justify empirical-CDF confidence bands. Decreasing caps make eligible observations initial prefixes of iid streams. Successful short-cap trials are not selectively reused above their cap.

**Capping is not candidate acquisition.** A matched full-horizon race has identical candidate and frontier sequences, isolating capping work. Comparisons only against round-robin would overattribute generic adaptive-sampling gains.

**Early failure changed the empirical conclusion.** The first symbolic cost calculation charged every failed proof until its cap. That corresponds to an undetectable/stalled failure model, not the actual program, which stops on missing rules. Correct termination-time accounting changes the symbolic capping gain from 72.9% to 2.7%. Both measurements are retained with explicit model labels; the executable-policy number is the main result. Training-rollout cost metadata was also corrected to account for terminal failures without changing any training labels or predictions.

**Strong baselines remain strong.** A saturated hazard estimate is the empirical CDF algebraically. In feature-generalization experiments, the standard hazard predictor is strongest. Compression approximately preserves worst-budget decisions but can increase average regret; small observed accuracy differences are not claimed significant.

## Completed checks

133 passing pytest cases; 32,007 independent finite cases; 360 completed fixed-menu certificate runs with no observed false claims; 245,760 actual symbolic rollouts; 300 held-out symbolic menus over five fitted models. No GPU or language-model API was used. The manuscript contains full proofs and honest negative controls. These checks are not independent mathematical or novelty review.
