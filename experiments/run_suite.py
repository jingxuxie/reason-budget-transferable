"""Run exact-profile and certified-learning experiments; no GPU/API required."""
import argparse, csv, json, platform, time
from pathlib import Path
import numpy as np
from budget_values.core import *
from budget_values.environments import *
from budget_values.racing import *

ROOT=Path(__file__).resolve().parents[1]


def make_menu(family,seed,h=64,m=6):
    if family=='symbolic':
        p=random_horn_program(4000+seed,m)
        return p.exact_profiles(h)[0]
    return stochastic_menu(family,np.random.default_rng(1000+seed),m,h)


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=24)
    ap.add_argument('--horizon',type=int,default=64)
    ap.add_argument('--epsilon',type=float,default=.1)
    ap.add_argument('--batch',type=int,default=32)
    args=ap.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    start=time.perf_counter()
    families=['crossing','noncrossing','saturated','late','symbolic']
    rep,learn,cert=[],[],[]
    max_identity_error=0.
    for family in families:
        for seed in range(args.seeds):
            f=make_menu(family,seed,args.horizon)
            m,h=f.shape
            const=int(np.argmax(f.mean(axis=1)))
            last=int(np.argmax(f[:,-1]))
            for eps in [.01,.02,.05,.1,.2]:
                seg=optimal_selector(f,eps)
                lazy=lazy_selector(f,eps)
                choices=decode(seg,h)
                rep.append(dict(family=family,seed=seed,epsilon=eps,
                                segments=len(seg),lazy_segments=len(lazy),
                                exact_segments=len(optimal_selector(f,0)),
                                worst_regret=float(regret(f,choices).max()),
                                mean_regret=float(regret(f,choices).mean()),
                                best_mean_constant_worst=float(regret(f,np.full(h,const)).max()),
                                terminal_scalar_worst=float(regret(f,np.full(h,last)).max())))
            for n in [64,256,1024]:
                sampler=ProfileSampler(f,12000+seed)
                times=[sampler.full_times(i,n) for i in range(m)]
                e=np.array([empirical_cdf(t,h) for t in times])
                hz=np.array([hazard_cdf(t,h) for t in times])
                max_identity_error=max(max_identity_error,float(np.max(np.abs(e-hz))))
                comp=optimal_selector(e,.02)
                methods={'terminal_scalar':np.full(h,int(np.argmax(e[:,-1]))),
                         'budget_table':e.argmax(axis=0),
                         'saturated_hazard':hz.argmax(axis=0),
                         'compressed_table':decode(comp,h)}
                # Match ties for algebraically identical ECDF / saturated hazard.
                methods['saturated_hazard']=e.argmax(axis=0)
                eta=float(np.max(np.abs(e-f)))
                for method,choices in methods.items():
                    r=regret(f,choices)
                    learn.append(dict(family=family,seed=seed,n_per_arm=n,method=method,
                                      worst_regret=float(r.max()),mean_regret=float(r.mean()),
                                      segments=len(comp) if method=='compressed_table' else
                                      int(1+np.count_nonzero(np.diff(choices))),
                                      sup_estimation_error=eta,
                                      theoretical_bound=.02+2*eta if method=='compressed_table' else 2*eta))
            for strategy in ['backward','full_lucb','round_robin']:
                if family=='symbolic':
                    ff,gg,_=random_horn_program(4000+seed,m).exact_terminal_laws(h)
                    sampler=TerminalLawSampler(ff,gg,24000+seed)
                else:
                    sampler=ProfileSampler(f,24000+seed)
                keep=(family=='crossing' and seed==0)
                result=run_race(sampler,m,h,epsilon=args.epsilon,delta=.05,
                                strategy=strategy,batch=args.batch,keep_trace=keep)
                r=regret(f,result['choices'])
                cert.append(dict(family=family,seed=seed,strategy=strategy,
                                 epsilon=args.epsilon,delta=.05,batch=args.batch,
                                 complete=result['complete'],cost=result['cost'],
                                 queries=result['queries'],certificate=result['certificate'],
                                 worst_regret=float(r.max()),mean_regret=float(r.mean()),
                                 segments=len(result['segments']) if result['complete'] else -1,
                                 band_coverage=bool(np.all(result['lower']<=f+1e-12) and
                                                    np.all(f<=result['upper']+1e-12)),
                                 invalid_certificate=bool(result['complete'] and r.max()>args.epsilon+1e-12)))
                if keep:
                    write_csv(out/f'trace_{strategy}.csv',result['trace'])
        print(f'{family}: completed {args.seeds} menus',flush=True)
        write_csv(out/'representation.csv',rep)
        write_csv(out/'sampled_values.csv',learn)
        write_csv(out/'certification.csv',cert)
    summary={'config':vars(args),'families':families,'python':platform.python_version(),
             'numpy':np.__version__,'runtime_seconds':time.perf_counter()-start,
             'menus':len(families)*args.seeds,'certified_runs':sum(x['complete'] for x in cert),
             'invalid_certificates':sum(x['invalid_certificate'] for x in cert),
             'simultaneous_final_band_failures':sum(not x['band_coverage'] for x in cert),
             'hazard_ecdf_max_abs_difference':max_identity_error}
    (out/'run_metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
