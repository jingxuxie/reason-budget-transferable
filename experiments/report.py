"""Generate paper tables, figures, and an auditable summary from measured CSVs."""
import csv,json,platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from budget_values.core import *
from budget_values.environments import alternating_profiles
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
# SVG keeps text as text: no font files are distributed.
plt.rcParams['svg.fonttype']='none'


def read(name):return list(csv.DictReader(open(ROOT/'results'/name)))
def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def main():
    cert=read('certification.csv');rep=read('representation.csv');learn=read('learned_critics.csv')
    families=['crossing','noncrossing','saturated','late','symbolic']
    labels=['Crossing','Noncrossing','Saturated','Late','Symbolic\n(terminal-aware)']
    stats=[]
    for k,family in enumerate(families):
        data={s:sorted([r for r in cert if r['family']==family and r['strategy']==s],key=lambda r:int(r['seed']))
              for s in ['backward','full_lucb','round_robin']}
        bd=np.array([float(r['cost']) for r in data['backward']]);full=np.array([float(r['cost']) for r in data['full_lucb']])
        uniform=np.array([float(r['cost']) for r in data['round_robin']])
        idx=np.random.default_rng(998+k).integers(len(bd),size=(5000,len(bd)))
        boot=1-bd[idx].mean(1)/full[idx].mean(1)
        stats.append(dict(family=family,backward_cost=float(bd.mean()),full_cost=float(full.mean()),
                          uniform_cost=float(uniform.mean()),capping_saving=1-float(bd.mean()/full.mean()),
                          capping_ci=np.quantile(boot,[.025,.975]).tolist(),
                          total_saving=1-float(bd.mean()/uniform.mean()),
                          backward_queries=mean(data['backward'],'queries'),
                          full_queries=mean(data['full_lucb'],'queries'),
                          worst_regret=mean(data['backward'],'worst_regret'),
                          regimes=mean(data['backward'],'segments')))
    learnerstats=[]
    for method in ['learned_scalar','budget_conditioned','budget_conditioned_compressed','hazard','hazard_compressed']:
        rows=[r for r in learn if r['method']==method]
        seedmeans=np.array([mean([r for r in rows if int(r['seed'])==seed],'worst_regret') for seed in range(5)])
        learnerstats.append(dict(method=method,worst_regret=mean(rows,'worst_regret'),
                                 worst_regret_se=float(seedmeans.std(ddof=1)/np.sqrt(5)),
                                 mean_regret=mean(rows,'mean_regret'),regimes=mean(rows,'segments')))
    stalled=read('symbolic_stalled_failure_ablation.csv')
    a=mean([r for r in stalled if r['strategy']=='backward'],'cost')
    b=mean([r for r in stalled if r['strategy']=='full_lucb'],'cost')
    summary=dict(certification=stats,learned_critics=learnerstats,
                 symbolic_stalled_capping_saving=1-a/b,
                 interpretation='Capping saves work but not rollout count against the matched race. '
                    'Symbolic savings are small after terminal failure is charged correctly. '
                    'Learned hazard is the strongest tested predictive baseline; compression is not a new predictor.')
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Cost effect, paired ratios and paired bootstrap CI.
    fig,ax=plt.subplots(figsize=(6.4,3.4))
    y=np.array([100*s['capping_saving'] for s in stats]);lo=np.array([100*s['capping_ci'][0] for s in stats]);hi=np.array([100*s['capping_ci'][1] for s in stats])
    ax.bar(np.arange(5),y,yerr=np.array([y-lo,hi-y]),capsize=3)
    ax.set_xticks(np.arange(5),labels,fontsize=9);ax.set_ylabel('Work saved vs. matched full-horizon race (%)')
    ax.set_ylim(0,85);ax.grid(axis='y',alpha=.25);save(fig,'capping_cost')
    # Accounting diagnostic must remain visible rather than selecting favorable accounting.
    fig,ax=plt.subplots(figsize=(5.5,3.1))
    ax.bar([0,1],[100*(1-a/b),100*stats[-1]['capping_saving']])
    ax.set_xticks([0,1],['Undetectable/stalled failure\n(ablation)','Detectable terminal failure\n(main symbolic experiment)'],fontsize=9)
    ax.set_ylabel('Capping work saved (%)');ax.set_ylim(0,85);ax.grid(axis='y',alpha=.25)
    save(fig,'cost_accounting')
    # Rate-tight adversarial alternation.
    f=alternating_profiles(.05,horizon=20)
    fig,ax=plt.subplots(figsize=(6.3,3.3))
    ax.step(np.arange(1,21),f[0],where='post',label='Prefix A')
    ax.step(np.arange(1,21),f[1],where='post',label='Prefix B')
    ax.set_xlabel('Remaining budget');ax.set_ylabel('Verified success probability')
    ax.legend(loc='lower right');ax.set_xlim(1,20);ax.grid(alpha=.2);save(fig,'alternating_profiles')
    # Compression exact menus.
    fig,ax=plt.subplots(figsize=(6.3,3.3))
    for family in ['crossing','symbolic','noncrossing']:
        xx=[.01,.02,.05,.1,.2]
        yy=[mean([r for r in rep if r['family']==family and float(r['epsilon'])==x],'segments') for x in xx]
        ax.plot(xx,yy,marker='o',label=family.capitalize())
    ax.set_xscale('log');ax.set_xlabel('Allowed worst-budget regret');ax.set_ylabel('Minimum budget intervals (mean)')
    ax.legend();ax.grid(alpha=.2);save(fig,'compression')
    # Learning trace matches number of samples but differs in simulator work.
    fig,ax=plt.subplots(figsize=(6.2,3.3))
    for strategy,label in [('backward','Backward capped'),('full_lucb','Matched full horizon'),('round_robin','Full-horizon round robin')]:
        tr=read('trace_'+strategy+'.csv')
        ax.plot([float(r['cost']) for r in tr],[float(r['certificate']) for r in tr],label=label)
    ax.axhline(.1,linestyle=':',label='Target regret 0.10')
    ax.set_xscale('log');ax.set_xlabel('Accumulated rollout work');ax.set_ylabel('Worst-budget regret certificate')
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'certification_trace')
    # Learned critics, held-out programs; SE over five separate training fits.
    fig,ax=plt.subplots(figsize=(6.4,3.3))
    short=['Scalar','Budget','Budget +\ncompression','Hazard','Hazard +\ncompression']
    ax.bar(range(5),[s['worst_regret'] for s in learnerstats],yerr=[s['worst_regret_se'] for s in learnerstats],capsize=3)
    ax.set_xticks(range(5),short,fontsize=9);ax.set_ylabel('Worst-budget regret (mean over programs)')
    ax.grid(axis='y',alpha=.25);save(fig,'learned_critics')
    # LaTeX tables.
    lines=[]
    for s in stats:
        label=s['family'].capitalize()
        lines.append(f"{label} & {s['backward_cost']/1000:.1f} & {s['full_cost']/1000:.1f} & {s['uniform_cost']/1000:.1f} & {100*s['capping_saving']:.1f} & {s['worst_regret']:.4f} \\\\")
    (ROOT/'paper/table_cost.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for s in learnerstats:
        label={'learned_scalar':'Scalar','budget_conditioned':'Budget-conditioned','budget_conditioned_compressed':'Budget-conditioned + compression','hazard':'Hazard','hazard_compressed':'Hazard + compression'}[s['method']]
        lines.append(f"{label} & {s['worst_regret']:.4f} $\\pm$ {s['worst_regret_se']:.4f} & {s['mean_regret']:.4f} & {s['regimes']:.2f} \\\\")
    (ROOT/'paper/table_learned.tex').write_text('\n'.join(lines)+'\n')
    import re
    manuscript=ROOT/'paper/main.tex'
    if manuscript.exists():
        text=manuscript.read_text()
        for name in ['cost','learned']:
            rows=(ROOT/f'paper/table_{name}.tex').read_text().strip()
            pattern=rf'(% BEGIN AUTO TABLE {name}\n).*?(\n% END AUTO TABLE {name})'
            text=re.sub(pattern,lambda m:m.group(1)+rows+m.group(2),text,flags=re.S)
        manuscript.write_text(text)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
