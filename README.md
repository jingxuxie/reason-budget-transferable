# Budget-Transferable Process Values

**Compact Decisions and the Cost of Capped Evidence**

A theory-first, CPU-only research draft on selecting among a fixed menu of reasoning continuations across compute deadlines. The paper source is in `paper/main.tex`; reproduce all figures and the PDF with the commands below.

## Main results

For monotone verified-success profiles, a deterministic epsilon-optimal menu selector needs O(1/epsilon) contiguous budget regimes; two candidates give a matching-order lower construction. Rectangular monotone confidence bands admit an exact regret certificate and a minimum-regime compiler. Backward Deadline Racing retires large deadlines first, making capped observations initial prefixes of independent sample streams. Standard time-uniform DKW bands then give a horizon-free rollout-count guarantee. A coupling to a matched full-horizon race preserves acquisition decisions and sample counts while weakly reducing executed work. A one-regime counterexample still requires order H/epsilon^2 learning work.

These are **menu-specific decision guarantees**, not universal per-prefix embedding compression, unrestricted tree-search guarantees, or a new survival predictor.

## Measured results

The recorded suite contains 120 menus and 360 certification runs, all completed with no observed false certificate. Tests pass 133 cases. A separate finite audit checks 32,007 cases, and actual symbolic execution checks 245,760 rollouts using 937,338 rule applications. Learned critics are evaluated on 300 held-out symbolic menus over five fits.

| Family | BDR mean work | Matched full mean work | Capping-only saving |
|---|---:|---:|---:|
| Crossing | 304,512 | 391,962 | 22.3% |
| Noncrossing | 114,616 | 122,157 | 6.2% |
| Rapidly saturated | 15,407 | 50,359 | 69.4% |
| Late success | 418,513 | 421,888 | 0.8% |
| Symbolic, terminal-aware | 20,939 | 21,512 | 2.7% |

**Accounting matters:** treating failed symbolic proofs as stalled until the cap would report 72.9% savings. The actual proof policy terminates on detectable failure, yielding only 2.7% capping savings. The alternative cost model is retained as a labeled ablation. The larger 63.6% saving against symbolic round-robin sampling mostly reflects adaptive candidate acquisition shared with the matched full-horizon race.

The standard learned hazard model is the strongest predictive baseline (mean worst-budget regret 0.0660, versus 0.1911 for budget-conditioned regression and 0.6282 for a scalar). Compression yields 0.0655 worst-budget regret and reduces intervals from 2.30 to 2.11, but slightly increases average regret; this is not a claim of significant predictive improvement.

## Reproduce

```bash
python -m pip install -e '.[test]'
make experiments
make test
make paper
```

`make paper` requires a TeX distribution with `pdflatex` and `bibtex` or `bibtex8`. On Windows without Make, run the Python commands listed in `Makefile`. No API credentials, GPU, or downloaded dataset are needed. Dependencies are in `pyproject.toml`; the recorded environment is in `results/environment.json`.

The experiment scripts regenerate raw CSV/JSON outputs, tables, and PDF/SVG figures. The main certification suite took about 24 seconds in the execution container; this is not a promised laptop runtime. Work savings count continuation steps, not controller or feature-extraction overhead.

## Layout

- `budget_values/`: profile operations, exact compiler, capped racing, symbolic environments.
- `experiments/`: exact/sampled profiles, learned critics, exhaustive checks, executable proof audit, accounting ablation, reporting.
- `tests/`: independent DP comparisons, concentration-interface tests, coupling tests, and environment checks.
- `paper/`: full manuscript, complete proofs, bibliography, generated tables.
- `results/`: measured summaries and machine-readable metadata; experiment commands regenerate all raw outputs.
- `notes/`: proof audit, novelty boundaries, research log, and submission checklist.

## Scope and status

Success must be exactly verified and retained. Menus and continuation policies are fixed. Confidence guarantees apply to repeated samples of known candidates, not to unseen programs. Capped trials must be declared predictably and nonincreasingly; outcome-selective pooling above the cap is not allowed. Early terminal failure is charged at its true termination time. No LLM performance is demonstrated.

This is a substantive reproducible research draft, **not yet independently reviewed or submission-ready**. The most important remaining work is an independent proof/novelty audit against utilitarian algorithm configuration, stronger natural reasoning or configuration experiments, and deciding whether the fixed-menu scope is sufficient without a feature-transfer theorem. See `notes/SUBMISSION_CHECKLIST.md`.
