PYTHON ?= python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=.
.PHONY: test experiments paper all
all: experiments test paper

test:
	$(PYTHON) -m pytest -q

experiments:
	$(PYTHON) experiments/run_suite.py --seeds 24
	$(PYTHON) experiments/stalled_ablation.py
	$(PYTHON) experiments/learned_critics.py
	$(PYTHON) experiments/audit_theory.py
	$(PYTHON) experiments/symbolic_rollout_audit.py
	$(PYTHON) experiments/report.py

paper:
	$(PYTHON) scripts/build_paper.py
