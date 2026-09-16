"""Check compiled hitting-time laws against actual rule-by-rule proof executions."""
import csv,json,time
from pathlib import Path
import numpy as np
from budget_values.environments import random_horn_program
from budget_values.core import empirical_cdf
ROOT=Path(__file__).resolve().parents[1]


def main():
    start=time.perf_counter();rows=[];total_cost=0;trace_example=None
    programs=10;m=6;h=32;n=4096
    for seed in range(programs):
        p=random_horn_program(5000+seed,m)
        f,states=p.exact_profiles(h)
        for arm in range(m):
            rng=np.random.default_rng(9000+seed*m+arm)
            times=[]
            for k in range(n):
                obs=p.rollout(arm,h,rng)
                times.append(obs.success_time if obs.success_time is not None else h+1)
                total_cost+=obs.cost
            error=float(np.max(np.abs(empirical_cdf(times,h)-f[arm])))
            rows.append(dict(seed=seed,arm=arm,rollouts=n,states=states,cdf_error=error))
        if seed==0:
            rr=np.random.default_rng(18000)
            for _ in range(100):
                obs,trace=p.rollout(0,h,rr,return_trace=True)
                if obs.success_time is not None:
                    trace_example=dict(rules=p.rules,prefix=p.prefixes[0],trace=trace,success_time=obs.success_time)
                    break
    bound=float(np.sqrt(np.log(2*programs*m/.01)/(2*n)))
    summary=dict(actual_rule_by_rule_rollouts=programs*m*n,actual_rule_applications=total_cost,
                 maximum_cdf_error=max(r['cdf_error'] for r in rows),
                 simultaneous_99_percent_dkw_bound=bound,
                 bound_violations=sum(r['cdf_error']>bound for r in rows),
                 runtime_seconds=time.perf_counter()-start)
    out=ROOT/'results'
    with open(out/'symbolic_rollout_audit.csv','w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'symbolic_rollout_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'symbolic_trace_example.json').write_text(json.dumps(trace_example,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
