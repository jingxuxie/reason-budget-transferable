"""Independent finite enumeration of deterministic claims (not a proof substitute)."""
import itertools,json,time
from pathlib import Path
import numpy as np
from budget_values.core import *
from budget_values.environments import alternating_profiles
ROOT=Path(__file__).resolve().parents[1]


def dp_count(ok):
    h=ok.shape[1];dp=[0]+[999]*h
    for r in range(1,h+1):
        for l in range(r):
            if np.any(np.all(ok[:,l:r],axis=1)):
                dp[r]=min(dp[r],dp[l]+1)
    return dp[-1]


def main():
    start=time.perf_counter();cases=0
    cols=[np.array([(mask>>i)&1 for i in range(3)],dtype=bool) for mask in range(1,8)]
    for sequence in itertools.product(range(7),repeat=5):
        ok=np.stack([cols[i] for i in sequence],axis=1)
        assert len(compress_feasible(ok))==dp_count(ok)
        cases+=1
    grid=np.array(list(itertools.combinations_with_replacement(np.linspace(0,1,5),4)))
    complexity=0
    for i in range(len(grid)):
        for j in range(len(grid)):
            f=np.stack([grid[i],grid[j]])
            for eps in [.1,.25,.4]:
                seg=lazy_selector(f,eps)
                assert regret(f,decode(seg,4)).max()<=eps+1e-12
                assert len(seg)<=1+int(1/eps)
                complexity+=1
    # Exhaust every admissible profile on a coarse grid inside random monotone
    # rectangles and compare the analytic certificate with direct maximization.
    rng=np.random.default_rng(6001);bandcases=0
    for _ in range(500):
        base=grid[rng.integers(len(grid),size=(2,2))]
        lo=base.min(axis=1);hi=base.max(axis=1)
        candidates=[grid[np.all((grid>=lo[i])&(grid<=hi[i]),axis=1)] for i in range(2)]
        brute=np.zeros((2,4))
        for a in candidates[0]:
            for b in candidates[1]:
                brute=np.maximum(brute,np.stack([np.maximum(b-a,0),np.maximum(a-b,0)]))
        np.testing.assert_allclose(brute,robust_gaps(lo,hi),atol=1e-12)
        bandcases+=1
    summary=dict(exhaustive_feasibility_matrices=cases,monotone_pair_epsilon_checks=complexity,
                 enumerated_band_rectangles=bandcases,failures=0,runtime_seconds=time.perf_counter()-start)
    (ROOT/'results/theory_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
