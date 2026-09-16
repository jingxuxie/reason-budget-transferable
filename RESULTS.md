# Recorded results — 16 September 2026

See `results/summary.json` for exact values and paired bootstrap intervals. Raw output hashes are in `results/raw_manifest.json`; run the experiment scripts to regenerate the CSVs. The downloadable research bundle also contains the full recorded CSVs.

## Capping-only work savings

The matched full-horizon race uses exactly the same candidate acquisition decisions and sample counts. Work is measured in continuation-step equivalents; accelerated law sampling is used for the main suite, with a separate actual symbolic execution audit.

| Family | BDR mean work | Full mean work | Savings | Paired 95% interval |
|---|---:|---:|---:|---|
| crossing | 304511.75 | 391961.96 | 22.31% | [19.03, 26.19]% |
| noncrossing | 114616.33 | 122157.12 | 6.17% | [4.55, 8.26]% |
| saturated | 15406.75 | 50358.71 | 69.41% | [65.64, 72.62]% |
| late | 418513.33 | 421888.00 | 0.80% | [0.77, 0.84]% |
| symbolic | 20938.67 | 21511.71 | 2.66% | [1.22, 4.42]% |

The symbolic main result charges detectable terminal failures at termination. Charging them until the cap produces a different stalled-failure model with 72.94% savings. It is retained only as an accounting ablation. All 360 main certificate runs completed; none had observed false certificates or final-band coverage failures. Confidence delta=.05 is per run, not across the full suite.

## Learned critics

300 held-out symbolic menus over five independent fits. The standard hazard predictor is strongest.

| Method | Mean worst-budget regret | SE over fits | Mean budget regret | Mean regimes |
|---|---:|---:|---:|---:|
| learned_scalar | 0.62821 | 0.00759 | 0.09705 | 1.000 |
| budget_conditioned | 0.19110 | 0.00862 | 0.01051 | 2.337 |
| budget_conditioned_compressed | 0.18676 | 0.00871 | 0.01102 | 2.127 |
| hazard | 0.06600 | 0.00946 | 0.00598 | 2.303 |
| hazard_compressed | 0.06552 | 0.00725 | 0.00638 | 2.113 |

Compression slightly increases average regret; small worst-regret improvements are not claimed significant. These learned predictors have no distribution-free unseen-program certificate.

## Validation

133 passing pytest cases; 32,007 exhaustive/finite analytic checks; 245,760 actual rule-by-rule rollouts with 937,338 rule applications. The maximum symbolic CDF discrepancy is 0.01701 versus a simultaneous 99% DKW threshold of 0.03386. No LLM or GPU experiment was performed.
