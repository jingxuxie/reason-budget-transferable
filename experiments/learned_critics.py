"""Held-out symbolic programs: small learned scalar, budget and hazard critics.

These are feature-based regressors, not language models. No distribution-free
certificate for unseen programs is claimed. Hyperparameters are fixed in code.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from budget_values.core import empirical_cdf,optimal_selector,decode,regret
from budget_values.environments import random_horn_program,structural_features,TerminalLawSampler
ROOT=Path(__file__).resolve().parents[1]


def dataset(offset,nmenus,h=32,m=6,rollouts=128):
    feats,truth,emp,events,risk=[],[],[],[],[]
    states=[];cost=0
    for k in range(nmenus):
        program=random_horn_program(offset+k,m)
        f,g,s=program.exact_terminal_laws(h)
        states.append(s)
        sampler=TerminalLawSampler(f,g,offset+1000000+k)
        for i,prefix in enumerate(program.prefixes):
            t,work=sampler.full_times_and_costs(i,rollouts)
            ev=np.bincount(np.minimum(t,h+1),minlength=h+2)[1:h+1]
            at=rollouts-np.r_[0,np.cumsum(ev[:-1])]
            feats.append(structural_features(program,prefix));truth.append(f[i])
            emp.append(empirical_cdf(t,h));events.append(ev);risk.append(at)
            cost+=int(work.sum())
    return (np.array(feats),np.array(truth),np.array(emp),np.array(events),
            np.array(risk),cost,max(states))


def expanded(x,h):
    repeated=np.repeat(np.log1p(x),h,axis=0)
    t=np.tile(np.log1p(np.arange(1,h+1))/np.log1p(h),len(x))
    return np.c_[repeated,t]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=5)
    ap.add_argument('--train-menus',type=int,default=160);ap.add_argument('--test-menus',type=int,default=60)
    args=ap.parse_args();h=32;m=6;rows=[];metadata=[];start=time.perf_counter()
    for seed in range(args.seeds):
        x,f,e,ev,risk,cost,maxstates=dataset(100000+seed*1000,args.train_menus,h,m)
        xt,ft,_,_,_,_,maxstatest=dataset(100000+seed*1000+500,args.test_menus,h,m)
        # Test sampled labels are never passed to fit. Only exact test laws enter
        # evaluator metrics; dataset helper computes unused test samples for a
        # consistent interface, excluded from reported training cost.
        common=dict(max_iter=100,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,
                    early_stopping=False,random_state=seed)
        scalar=HistGradientBoostingRegressor(**common).fit(np.log1p(x),e[:,-1])
        budget=HistGradientBoostingRegressor(monotonic_cst=[0]*x.shape[1]+[1],**common)
        budget.fit(expanded(x,h),e.ravel())
        hazard=HistGradientBoostingRegressor(**common)
        active=risk.ravel()>0
        hazard.fit(expanded(x,h)[active],(ev.ravel()[active]/risk.ravel()[active]),
                   sample_weight=risk.ravel()[active])
        sv=scalar.predict(np.log1p(xt))
        bv=np.clip(budget.predict(expanded(xt,h)).reshape(-1,h),0,1)
        hv=np.clip(hazard.predict(expanded(xt,h)).reshape(-1,h),0,1)
        hv=1-np.cumprod(1-hv,axis=1)
        predictions={'budget_conditioned':bv,'hazard':hv}
        for k in range(args.test_menus):
            ff=ft[k*m:(k+1)*m];ss=sv[k*m:(k+1)*m]
            choices=np.full(h,np.argmax(ss))
            rr=regret(ff,choices)
            rows.append(dict(seed=seed,problem=k,method='learned_scalar',worst_regret=float(rr.max()),
                             mean_regret=float(rr.mean()),segments=1,sup_error=float('nan'),bound=float('nan')))
            for name,pred in predictions.items():
                pp=pred[k*m:(k+1)*m];eta=float(np.max(np.abs(pp-ff)))
                for compressed in [False,True]:
                    if compressed:
                        seg=optimal_selector(pp,.03);choices=decode(seg,h);count=len(seg)
                    else:
                        choices=pp.argmax(axis=0);count=1+np.count_nonzero(np.diff(choices))
                    rr=regret(ff,choices)
                    rows.append(dict(seed=seed,problem=k,method=name+('_compressed' if compressed else ''),
                                     worst_regret=float(rr.max()),mean_regret=float(rr.mean()),segments=int(count),
                                     sup_error=eta,bound=2*eta+(.03 if compressed else 0.)))
        metadata.append(dict(seed=seed,training_rollout_steps=cost,max_training_states=maxstates,
                             max_test_states=maxstatest))
        print('learned seed',seed,'complete',flush=True)
    out=ROOT/'results'
    with open(out/'learned_critics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'learned_metadata.json').write_text(json.dumps(dict(config=vars(args),horizon=h,arms=m,
        rollouts_per_training_prefix=128,seed_details=metadata,runtime_seconds=time.perf_counter()-start,
        note='Symbolic features and histogram boosted regressors; no LLM or unseen-program certification.'),indent=2)+'\n')

if __name__=='__main__':main()
