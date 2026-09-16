import itertools
import numpy as np
import pytest
from budget_values.core import *
from budget_values.environments import *
from budget_values.racing import *


@pytest.mark.parametrize('seed',range(40))
def test_compression_bound_and_optimality(seed):
    rng=np.random.default_rng(seed)
    f=np.cumsum(rng.dirichlet(np.ones(21),size=5)[:,:20],axis=1)
    eps=.13
    lazy=lazy_selector(f,eps)
    opt=optimal_selector(f,eps)
    assert regret(f,decode(lazy,20)).max()<=eps+1e-10
    assert len(opt)<=len(lazy)<=1+int(1/eps)
    assert regret(f,decode(opt,20)).max()<=eps+1e-10
    # Independent dynamic program for shortest feasible partition.
    ok=f.max(0)[None,:]-f<=eps+1e-12
    dp=[0]+[999]*20
    for r in range(1,21):
        for l in range(r):
            if np.any(np.all(ok[:,l:r],axis=1)):
                dp[r]=min(dp[r],dp[l]+1)
    assert len(opt)==dp[-1]


@pytest.mark.parametrize('eps',[.05,.1,.125,.2,.25])
def test_lower_bound_construction(eps):
    f=alternating_profiles(eps)
    assert len(optimal_selector(f,eps))==int(np.floor((1+1e-12)/(2*eps)))


def test_crossing_and_noncrossing():
    f=np.array([[.6]*10,[0]*9+[.95]])
    assert len(optimal_selector(f,.1))==2
    f=stochastic_menu('noncrossing',np.random.default_rng(42))
    assert len(optimal_selector(f,0))==1


def test_robust_certificate_excludes_self():
    lo=np.array([[.5],[.4]]);hi=np.array([[.5],[1.]])
    np.testing.assert_allclose(robust_gaps(lo,hi),[[.5],[.1]])
    np.testing.assert_allclose(robust_gaps([[0,0]],[[1,1]]),[[0,0]])


@pytest.mark.parametrize('seed',range(10))
def test_hazard_ecdf_identity(seed):
    times=np.random.default_rng(seed).integers(1,18,100)
    np.testing.assert_allclose(empirical_cdf(times,16),hazard_cdf(times,16),atol=1e-14)


def test_censoring_does_not_mean_failure_forever():
    r=DeadlineRacer(2,8,.1)
    for _ in range(10):
        r.update(0,CappedObservation(3,None,3))
    assert np.all(r.upper[0,3:]==1)
    assert np.all(r.n[0,3:]==0)
    with pytest.raises(ValueError):
        r.update(0,CappedObservation(4,None,4))


def test_short_success_not_selectively_pooled_above_cap():
    r=DeadlineRacer(2,8,.1)
    r.update(0,CappedObservation(3,2,2))
    assert np.all(r.n[0,3:]==0) and np.all(r.wins[0,3:]==0)


@pytest.mark.parametrize('seed',range(10))
def test_racing_certification_and_caps(seed):
    f=stochastic_menu('crossing',np.random.default_rng(seed),m=3,h=12)
    res=run_race(ProfileSampler(f,seed),3,12,epsilon=.25,delta=.1,batch=8,keep_trace=True)
    assert res['complete']
    assert regret(f,res['choices']).max()<=.25+1e-12
    assert np.all(np.diff([d['cap'] for d in res['trace']])<=0)
    assert max(res['counts'])<=sample_bound(.25,3,.1)+7


def test_symbolic_exact_and_rollout():
    p=HornProgram({0:[()],1:[],2:[(),(1,)],3:[(0,)]},[(2,),(3,)])
    f,_=p.exact_profiles(4)
    np.testing.assert_allclose(f,[[.5,.5,.5,.5],[0,1,1,1]])
    rng=np.random.default_rng(32)
    hits=[p.rollout(0,4,rng).success_time is not None for _ in range(10000)]
    assert abs(np.mean(hits)-.5)<.025


@pytest.mark.parametrize('seed',range(10))
def test_symbolic_generator(seed):
    p=random_horn_program(seed)
    f,n=p.exact_profiles(32)
    validate_profiles(f)
    assert n<20000
    assert np.any(f.max(0)-f[0]>.1) or f[0,-1]>.5


def test_bad_inputs():
    with pytest.raises(ValueError): validate_profiles([[.8,.2]])
    with pytest.raises(ValueError): validate_profiles([[np.nan]])
    with pytest.raises(ValueError): compress_feasible([[True,False]])
    with pytest.raises(ValueError): robust_gaps([[.8]],[[.2]])
    with pytest.raises(ValueError): CappedObservation(5,6,5).validate()


@pytest.mark.parametrize('seed',range(40))
def test_pathwise_uncapped_coupling(seed):
    rng=np.random.default_rng(seed+818)
    m=int(rng.integers(2,7));h=int(rng.integers(8,32))
    f=stochastic_menu('crossing',rng,m,h)
    a=run_race(ProfileSampler(f,seed),m,h,.2,.1,batch=16,keep_trace=True)
    b=run_race(ProfileSampler(f,seed),m,h,.2,.1,strategy='full_lucb',batch=16,keep_trace=True)
    assert a['complete'] and b['complete']
    assert [(x['arm'],x['frontier']) for x in a['trace']]==[(x['arm'],x['frontier']) for x in b['trace']]
    np.testing.assert_equal(a['counts'],b['counts'])
    assert a['cost']<=b['cost']


def test_terminal_failure_cost_and_labels():
    p=HornProgram({0:[()],1:[],2:[(1,)],3:[(0,)]},[(2,),(3,)])
    f,g,_=p.exact_terminal_laws(8)
    np.testing.assert_allclose(g[0],[0,1,1,1,1,1,1,1])
    sampler=TerminalLawSampler(f,g,12)
    obs=sampler(0,8)
    assert obs.terminal_failure and obs.cost==2 and obs.success_time is None
    obs.validate()
    r=DeadlineRacer(2,8,.1)
    r.update(0,CappedObservation(3,None,2,terminal_failure=True))
    assert np.all(r.n[0,3:]==0)


@pytest.mark.parametrize('seed',range(10))
def test_terminal_coupling(seed):
    f,g,_=random_horn_program(4000+seed,4).exact_terminal_laws(24)
    a=run_race(TerminalLawSampler(f,g,seed),4,24,.2,.1,batch=16,keep_trace=True)
    b=run_race(TerminalLawSampler(f,g,seed),4,24,.2,.1,strategy='full_lucb',batch=16,keep_trace=True)
    assert a['queries']==b['queries'] and a['cost']<=b['cost']
    assert regret(f,a['choices']).max()<=.2


def test_terminal_vectorized_costs_preserve_labels():
    from budget_values.environments import TerminalLawSampler, ProfileSampler
    f=np.array([[0.,.2,.2,.4]])
    g=np.array([[.3,.3,.5,.5]])
    a=TerminalLawSampler(f,g,917)
    t,c=a.full_times_and_costs(0,10000)
    b=ProfileSampler(f,917)
    assert np.array_equal(t,b.full_times(0,10000))
    assert np.all(c<=np.minimum(t,4))
    assert np.any(c<np.minimum(t,4))
