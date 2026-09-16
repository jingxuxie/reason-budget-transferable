"""Backward Deadline Racing (BDR), with time-uniform, horizon-free bands.

DKW is valid here because declared caps NEVER INCREASE. At each deadline, the
eligible observations form a prefix of an arm's iid potential completion times.
Arbitrary adaptive cap orders do not inherit this argument. The API rejects them.
"""
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
from .core import validate_profiles, robust_gaps, compress_feasible, decode


def radius(n, arms: int, delta: float):
    if arms < 1 or not 0 < delta < 0.5:
        raise ValueError("require m>=1 and delta in (0,0.5)")
    n = np.asarray(n, dtype=float)
    safe = np.maximum(n, 1)
    rad = np.sqrt(np.log(np.pi**2 * arms * safe**2/(3*delta))/(2*safe))
    return np.where(n > 0, rad, 1.0)


def sample_bound(epsilon: float, arms: int, delta: float) -> int:
    """Smallest integer N with 2 r_N <= epsilon (radius decreasing for m>=2)."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1)")
    lo, hi = 0, 1
    while 2*float(radius(hi,arms,delta)) > epsilon:
        hi *= 2
    while hi-lo > 1:
        mid = (lo+hi)//2
        if 2*float(radius(mid,arms,delta)) <= epsilon:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class CappedObservation:
    cap: int
    success_time: int | None
    cost: int
    terminal_failure: bool = False

    def validate(self):
        if self.cap < 1 or self.cost < 1 or self.cost > self.cap:
            raise ValueError("invalid cap or cost")
        if self.success_time is not None:
            if self.terminal_failure or not 1 <= self.success_time <= self.cap or self.cost != self.success_time:
                raise ValueError("invalid observed success")
        elif self.cost != self.cap and not self.terminal_failure:
            raise ValueError("early unsuccessful termination must be flagged")


class DeadlineRacer:
    def __init__(self, arms: int, horizon: int, epsilon: float, delta: float=0.05):
        if arms < 1 or horizon < 1 or not 0 < epsilon < 1:
            raise ValueError("invalid dimensions or epsilon")
        radius(1,arms,delta)
        self.m, self.h, self.eps, self.delta = arms,horizon,epsilon,delta
        self.n = np.zeros((arms,horizon),dtype=np.int64)
        self.wins = np.zeros_like(self.n)
        self.lower = np.zeros_like(self.n,dtype=float)
        self.upper = np.ones_like(self.n,dtype=float)
        self.last_cap = horizon
        self.queries = self.cost = 0
        self.counts = np.zeros(arms,dtype=int)
        self.history: list[dict] = []

    def gaps(self):
        return robust_gaps(self.lower,self.upper)

    def certificate(self):
        gap = self.gaps()
        return gap.min(axis=0)

    def suggest(self):
        gap = self.gaps()
        unresolved = np.flatnonzero(gap.min(axis=0) > self.eps+1e-12)
        if len(unresolved)==0:
            return None
        b = int(unresolved[-1])
        leader = int(np.argmax(self.lower[:,b]))
        u = self.upper[:,b].copy()
        u[leader] = -np.inf
        challenger = int(np.argmax(u))
        # Wider of the lower leader and upper challenger; width > epsilon.
        pool = [leader,challenger]
        width = self.upper[pool,b]-self.lower[pool,b]
        arm = pool[int(np.argmax(width))]
        if b+1 > self.last_cap:
            raise RuntimeError("internal error: cap increased")
        return arm,b+1

    def update(self, arm: int, observation: CappedObservation):
        self.update_batch(arm,[observation])

    def update_batch(self, arm: int, observations: list[CappedObservation]):
        """A predictable same-arm, same-cap block; inspect bands at block end."""
        if not observations:
            raise ValueError("empty block")
        cap=observations[0].cap
        if not 0 <= arm < self.m or cap > self.h or cap > self.last_cap:
            raise ValueError("caps must be globally nonincreasing; invalid arm/cap")
        for obs in observations:
            obs.validate()
            if obs.cap != cap:
                raise ValueError("one declared cap per block")
        self.last_cap=cap
        self.n[arm,:cap]+=len(observations)
        events=np.array([obs.success_time for obs in observations if obs.success_time is not None],dtype=int)
        self.wins[arm,:cap]+=np.cumsum(np.bincount(events,minlength=cap+1)[1:cap+1])
        # Do NOT count successful short-cap runs at deadlines above their cap.
        mean = np.divide(self.wins[arm],self.n[arm],out=np.zeros(self.h),where=self.n[arm]>0)
        rad = radius(self.n[arm],self.m,self.delta)
        lo = np.maximum(0,mean-rad)
        hi = np.minimum(1,mean+rad)
        lo[self.n[arm]==0],hi[self.n[arm]==0] = 0,1
        self.lower[arm] = np.maximum.accumulate(np.maximum(self.lower[arm],lo))
        self.upper[arm] = np.minimum.accumulate(np.minimum(self.upper[arm],hi)[::-1])[::-1]
        if np.any(self.lower[arm] > self.upper[arm]+1e-10):
            raise RuntimeError("confidence bands inconsistent; no certificate issued")
        self.queries+=len(observations)
        self.cost+=sum(obs.cost for obs in observations)
        self.counts[arm]+=len(observations)

    def result(self):
        gap = self.gaps()
        complete = bool(np.all(gap.min(axis=0)<=self.eps+1e-12))
        choices = gap.argmin(axis=0)
        segments = None
        if complete:
            segments = compress_feasible(gap<=self.eps+1e-12)
            choices = decode(segments,self.h)
        return {"complete":complete,"choices":choices,"segments":segments,
                "certificate":float(gap[choices,np.arange(self.h)].max()),
                "cost":self.cost,"queries":self.queries,"counts":self.counts.copy()}


def run_race(sampler: Callable[[int,int],CappedObservation], arms: int, horizon: int,
             epsilon: float=0.1, delta: float=0.05, strategy: str="backward",
             max_queries: int=200000, batch: int=1, keep_trace: bool=False):
    """Strategies: backward; full_lucb (same arm decision, cap H); round_robin.

    Batching fixes an action for a predictable block. Only batch=1 has the exact
    m*N statement; predictable batches overshoot by at most batch-1 per arm.
    """
    if strategy not in {"backward","full_lucb","round_robin"} or batch<1:
        raise ValueError("invalid strategy or batch")
    racer=DeadlineRacer(arms,horizon,epsilon,delta)
    trace=[]
    while racer.queries < max_queries:
        action = racer.suggest()
        if action is None:
            break
        arm,cap=action
        if strategy=="full_lucb":
            cap=horizon
        elif strategy=="round_robin":
            arm=(racer.queries//batch)%arms
            cap=horizon
        observations=[sampler(arm,cap) for _ in range(min(batch,max_queries-racer.queries))]
        racer.update_batch(arm,observations)
        if keep_trace:
            trace.append({"query":racer.queries,"arm":arm,"cap":cap,"frontier":action[1],
                          "cost":racer.cost,"certificate":float(racer.certificate().max())})
    result=racer.result()
    result["trace"]=trace
    result["lower"]=racer.lower.copy()
    result["upper"]=racer.upper.copy()
    return result
