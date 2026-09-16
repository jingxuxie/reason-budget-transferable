"""Exact stochastic-completion and finite symbolic-proof environments.

The learner receives only CappedObservation, never the true CDF or an uncensored
completion time above its chosen cap. Evaluator-only CDFs permit exact regret.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from .core import validate_profiles
from .racing import CappedObservation


class ProfileSampler:
    """Law-equivalent rollout simulator with separate, reproducible arm streams."""
    def __init__(self, profiles, seed: int):
        self.f=validate_profiles(profiles)
        self.rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(self.f))]

    def __call__(self, arm: int, cap: int):
        if not 1 <= cap <= self.f.shape[1]:
            raise ValueError("invalid cap")
        u=self.rngs[arm].random()
        # Search only the observable range. No tail value is supplied to learner.
        t=int(np.searchsorted(self.f[arm,:cap],u,side="left"))+1
        success=t if t<=cap else None
        return CappedObservation(cap,success,min(t,cap))

    def full_times(self, arm: int, n: int):
        return np.searchsorted(self.f[arm],self.rngs[arm].random(n),side="left")+1


def stochastic_menu(kind: str, rng, m: int=6, h: int=64):
    t=np.arange(1,h+1,dtype=float)
    if kind=="noncrossing":
        rates=rng.uniform(0.02,0.3,m)
        return 1-np.exp(-rates[:,None]*t[None,:])
    if kind=="late":
        f=np.zeros((m,h)); f[:,-1]=np.linspace(.35,.75,m)
        return f
    if kind=="saturated":
        f=np.zeros((m,h))
        f[0]=.999*(1-np.exp(-t/2.5))
        for i in range(1,m):
            f[i]=rng.uniform(.05,.4)*(1-np.exp(-t/rng.uniform(1,5)))
        return f
    if kind=="crossing":
        f=np.zeros((m,h))
        # Diverse mixtures, plus explicit fast and delayed contenders.
        for i in range(m):
            locs=rng.integers(1,h+1,3)
            mass=rng.dirichlet(np.ones(4))[:3]
            for loc,w in zip(locs,mass):
                f[i,loc-1:]+=w
        f[0]=rng.uniform(.4,.65)*(1-np.exp(-t/rng.uniform(1,3)))
        delay=int(rng.integers(max(2,h//4),max(3,h//2)))
        f[1]=rng.uniform(.85,.99)*(1-np.exp(-np.maximum(t-delay,0)/rng.uniform(2,7)))
        return f
    raise ValueError(f"unknown family {kind}")


def alternating_profiles(epsilon: float, horizon: int|None=None):
    k=int(np.floor((1+1e-12)/(2*epsilon)))
    h=max(k,horizon or k)
    f=np.zeros((2,h))
    for j in range(k):
        f[j%2,j:]=2*epsilon*(j+1)
    return validate_profiles(f)


@dataclass
class HornProgram:
    """Acyclic backward chaining, with stochastic rule selection and no repair.

    rules[g] is a list of antecedent tuples proving g; () is an axiom rule.
    State is the ordered tuple of unproved subgoals. Choosing one rule costs one
    unit. A missing rule produces detectable terminal failure and stops execution.
    Budgets count attempted rule applications. Repeated subgoals are not memoized by the policy.
    """
    rules: dict[int,list[tuple[int,...]]]
    prefixes: list[tuple[int,...]]

    def transitions(self,state):
        if state in [(),(-1,)]:
            return [(state,1.)]
        options=self.rules.get(state[0],[])
        if not options:
            return [((-1,),1.)]
        return [(tuple(body)+state[1:],1/len(options)) for body in options]

    def exact_terminal_laws(self,horizon: int,max_states: int=20000):
        states=[(),(-1,)]
        index={s:i for i,s in enumerate(states)}
        for s in self.prefixes:
            if s not in index:
                index[s]=len(states); states.append(s)
        rows,cols,data=[],[],[]
        cursor=0
        while cursor<len(states):
            if len(states)>max_states:
                raise ValueError("symbolic state limit exceeded")
            for target,p in self.transitions(states[cursor]):
                if target not in index:
                    index[target]=len(states);states.append(target)
                rows.append(cursor);cols.append(index[target]);data.append(p)
            cursor+=1
        p=csr_matrix((data,(rows,cols)),shape=(len(states),len(states)))
        values=np.zeros((len(states),2));values[0,0]=1;values[1,1]=1
        out=np.zeros((len(self.prefixes),horizon,2))
        ids=[index[s] for s in self.prefixes]
        for b in range(horizon):
            values=p@values
            out[:,b]=values[ids]
        return validate_profiles(out[:,:,0]),validate_profiles(out[:,:,1]),len(states)

    def exact_profiles(self,horizon: int,max_states: int=20000):
        success,_,n=self.exact_terminal_laws(horizon,max_states)
        return success,n

    def rollout(self,arm: int,cap: int,rng,return_trace=False):
        state=self.prefixes[arm]
        trace=[]
        for t in range(1,cap+1):
            options=self.transitions(state)
            ix=int(rng.integers(len(options)))
            state=options[ix][0]
            if return_trace:
                trace.append(state)
            if not state:
                obs=CappedObservation(cap,t,t)
                return (obs,trace) if return_trace else obs
            if state==(-1,):
                obs=CappedObservation(cap,None,t,terminal_failure=True)
                return (obs,trace) if return_trace else obs
        obs=CappedObservation(cap,None,cap)
        return (obs,trace) if return_trace else obs


def random_horn_program(seed: int, m: int=6, depth: int=7):
    rng=np.random.default_rng(seed)
    # 0: proved axiom; 1: unprovable. Low-index antecedents ensure acyclicity.
    rules={0:[()],1:[]}
    for g in range(2,depth+2):
        opts=[]
        for _ in range(int(rng.integers(1,4))):
            size=int(rng.choice([0,1,2],p=[.12,.67,.21]))
            opts.append(tuple(int(x) for x in rng.integers(0,g,size)))
        rules[g]=opts
    prefixes=[]
    next_id=depth+2
    # Root routes correspond to already selected alternatives of a common goal.
    # Fast uncertain route with explicit failure alternatives.
    good=int(rng.integers(1,4));bad=int(rng.integers(1,3))
    rules[next_id]=[()]*good+[(1,)]*bad
    prefixes.append((next_id,)); next_id+=1
    # A reliable delayed proof: conjunctive copies of a chain, with a chance of
    # choosing an unprovable subgoal at its root. The chain is not an oracle label.
    tail=0
    for _ in range(int(rng.integers(3,9))):
        rules[next_id]=[(tail,)];tail=next_id;next_id+=1
    rules[next_id]=[(tail,)]*int(rng.integers(5,10))+[(1,)]
    prefixes.append((next_id,));next_id+=1
    for _ in range(m-2):
        g=int(rng.integers(2,depth+2))
        sub=(g,) if rng.random()<.65 else (g,int(rng.integers(0,depth+2)))
        rules[next_id]=[sub,(),(1,)] if rng.random()<.4 else [sub]
        prefixes.append((next_id,));next_id+=1
    return HornProgram(rules,prefixes)


def structural_features(program: HornProgram, prefix: tuple[int,...]):
    """Observable rule-graph features, not completion-probability labels.

    Includes shortest proof length computed on the small acyclic rule graph;
    this is privileged symbolic structure, NOT an available LLM hidden state.
    """
    from functools import lru_cache
    reached=set()
    def visit(g):
        if g in reached:return
        reached.add(g)
        for body in program.rules.get(g,[]):
            for child in body:visit(child)
    for g in prefix:visit(g)
    @lru_cache(None)
    def shortest(g):
        opts=program.rules.get(g,[])
        if not opts:return 256.
        return min(256.,min(1+sum(shortest(x) for x in body) for body in opts))
    @lru_cache(None)
    def depth(g):
        opts=program.rules.get(g,[])
        return 1+max([depth(x) for body in opts for x in body],default=0)
    options=[program.rules.get(g,[]) for g in reached]
    bodies=[b for o in options for b in o]
    branching=np.array([len(o) for o in options],dtype=float)
    return np.array([len(prefix),len(reached),len(bodies),
                     sum(len(o)==0 for o in options),sum(len(b)==0 for b in bodies),
                     sum(len(b)==2 for b in bodies),
                     branching.mean(),branching.std(),branching.max(),
                     min(256.,sum(shortest(g) for g in prefix)),
                     max(depth(g) for g in prefix),
                     np.mean([len(b) for b in bodies]) if bodies else 0.],dtype=float)


class TerminalLawSampler(ProfileSampler):
    """Correct cost when a symbolic proof can terminate unsuccessfully early.

    Success labels use the same uniform stream as ProfileSampler. Failure times
    affect only real work and the terminal flag; the certifier conservatively
    ignores their implications beyond the declared cap.
    """
    def __init__(self,success_profiles,failure_profiles,seed):
        super().__init__(success_profiles,seed)
        self.g=validate_profiles(failure_profiles)
        if self.g.shape!=self.f.shape or np.any(self.g+self.f>1+1e-10):
            raise ValueError("inconsistent terminal laws")

    def __call__(self,arm,cap):
        if not 1<=cap<=self.f.shape[1]:raise ValueError("invalid cap")
        u=self.rngs[arm].random()
        if u<=self.f[arm,-1]:
            t=int(np.searchsorted(self.f[arm],u,side="left"))+1
            return CappedObservation(cap,t if t<=cap else None,min(t,cap))
        failure_u=u-self.f[arm,-1]
        if failure_u<=self.g[arm,-1]:
            t=int(np.searchsorted(self.g[arm],failure_u,side="left"))+1
            return CappedObservation(cap,None,min(t,cap),terminal_failure=t<=cap)
        return CappedObservation(cap,None,cap)

    def full_times_and_costs(self, arm: int, n: int):
        """Return common-horizon labels and actual early-terminal rollout work."""
        if n < 0:
            raise ValueError("sample count must be nonnegative")
        u = self.rngs[arm].random(n)
        h = self.f.shape[1]
        times = np.searchsorted(self.f[arm], u, side="left") + 1
        costs = np.minimum(times, h)
        failed = (u > self.f[arm,-1]) & (u <= self.f[arm,-1]+self.g[arm,-1])
        costs[failed] = np.searchsorted(self.g[arm], u[failed]-self.f[arm,-1], side="left") + 1
        return times, costs
