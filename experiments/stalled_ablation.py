"""Reproduce the explicitly alternative stalled-failure symbolic cost model.

This is NOT the executable symbolic policy, which detects terminal failure.
It is retained to make the accounting correction independently reproducible.
"""
import csv
from pathlib import Path
from budget_values.core import regret
from budget_values.environments import random_horn_program, ProfileSampler
from budget_values.racing import run_race
ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[]
    for seed in range(24):
        f,_=random_horn_program(4000+seed,6).exact_profiles(64)
        for strategy in ['backward','full_lucb','round_robin']:
            out=run_race(ProfileSampler(f,24000+seed),6,64,.1,.05,strategy=strategy,batch=32)
            rr=regret(f,out['choices'])
            rows.append(dict(family='symbolic_stalled_failure_ablation',seed=seed,strategy=strategy,
                epsilon=.1,delta=.05,batch=32,complete=out['complete'],cost=out['cost'],
                queries=out['queries'],certificate=out['certificate'],worst_regret=float(rr.max()),
                mean_regret=float(rr.mean()),segments=len(out['segments']),
                band_coverage=bool(((out['lower']<=f+1e-12)&(f<=out['upper']+1e-12)).all()),
                invalid_certificate=bool(out['complete'] and rr.max()>.1+1e-10)))
    with open(ROOT/'results/symbolic_stalled_failure_ablation.csv','w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Recorded 72 explicitly stalled-failure ablation runs.')

if __name__=='__main__':main()
