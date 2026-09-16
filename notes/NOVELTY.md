# Novelty boundaries and closest literature

The references below were checked against primary paper pages on 16 September 2026. Bibliographic entries are in `paper/references.bib`.

## Established building blocks — not claimed as inventions

- Time-to-goal/success distributions, hazard parameterizations, and censored likelihoods: SVL, arXiv:2604.17551 (2026).
- Remaining-budget-guided reasoning search: Budget-Aware Value Tree Search, arXiv:2603.12634 (2026).
- Uncertainty about process-success probability: BetaPRM, arXiv:2605.15529, and distributional process rewards, arXiv:2605.06785 (2026).
- Runtime distributions as utility objects: Graham, Leyton-Brown, Roughgarden, ICML 2023, PMLR 202:11659–11682.
- Bounded runtime utilities, capped runs, confidence-driven algorithm configuration: Utilitarian Algorithm Configuration, arXiv:2310.20401, and Practical, Utilitarian Algorithm Configuration, arXiv:2510.14683v2.
- Empirical-CDF concentration: DKW–Massart (1990).
- Adaptive-experiment lower bounds: Kaufmann, Cappé, Garivier, JMLR 17(1), 2016.
- Best-arm identification with structured observations: Huang et al., ALT 2017.

## Candidate contribution of this draft

A single fixed-menu decision object must work uniformly over all supported deadlines. The draft combines a tight-order regime bound, exact band-based minimum-regime compilation, a decreasing-cap construction that preserves iid-prefix eligibility, and a coupling isolating capping-only work savings from acquisition. A lower bound separates a one-regime deployed policy from horizon-dependent learning work.

No priority claim is established merely by not finding identical wording. The closest risk is that algorithm-configuration theory already implies a substantial part of the learning story. A full theorem-by-theorem independent comparison and practical configuration baselines are still required. The matched full-horizon race is a controlled ablation, not a reproduction of COUP or another published implementation.

## What the empirical results do not establish

They do not show a new predictor outperforming survival modeling; hazard regression is the strongest tested predictor. Compression is modest on the held-out generator. They do not show large symbolic capping gains after correct early-failure accounting. They do not demonstrate LLM transfer, online changes of continuation policy, or inference-time gains after all controller overhead is counted.
