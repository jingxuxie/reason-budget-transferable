"""Deterministic decision compression and distributionally robust certificates.

Budgets are integers 1,...,H; array column b-1 represents budget b.
All guarantees concern a fixed finite menu and a fixed continuation procedure.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_profiles(values: ArrayLike) -> NDArray[np.float64]:
    f = np.asarray(values, dtype=float)
    if f.ndim != 2 or min(f.shape) < 1:
        raise ValueError("profiles must have shape (m,H), with m,H positive")
    if not np.all(np.isfinite(f)) or np.min(f) < -1e-12 or np.max(f) > 1+1e-12:
        raise ValueError("profiles must be finite and in [0,1]")
    if np.any(np.diff(f, axis=1) < -1e-12):
        raise ValueError("profiles must be nondecreasing in budget")
    return np.clip(f, 0, 1)


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    arm: int


def decode(segments: list[Segment], horizon: int) -> NDArray[np.int64]:
    out = np.full(horizon, -1, dtype=int)
    next_start = 1
    for seg in segments:
        if seg.start != next_start or not seg.start <= seg.end <= horizon:
            raise ValueError("segments must partition 1,...,H")
        out[seg.start-1:seg.end] = seg.arm
        next_start = seg.end+1
    if next_start != horizon+1:
        raise ValueError("incomplete segmentation")
    return out


def regret(f: ArrayLike, choices: ArrayLike) -> NDArray[np.float64]:
    f = validate_profiles(f)
    choices = np.asarray(choices, dtype=int)
    if choices.shape != (f.shape[1],) or np.any(choices < 0) or np.any(choices >= f.shape[0]):
        raise ValueError("one valid arm index is needed per budget")
    return f.max(axis=0)-f[choices, np.arange(f.shape[1])]


def compress_feasible(feasible: ArrayLike) -> list[Segment]:
    """Minimum-cardinality contiguous partition with a feasible label per segment.

    Farthest-extension greedy; exact for an arbitrary Boolean feasibility matrix.
    It is not a heuristic and does not require monotone feasibility sets.
    """
    ok = np.asarray(feasible, dtype=bool)
    if ok.ndim != 2 or min(ok.shape) == 0 or not np.all(ok.any(axis=0)):
        raise ValueError("every budget needs at least one feasible arm")
    m, h = ok.shape
    start, alive = 0, np.ones(m, dtype=bool)
    result = []
    for b in range(h):
        updated = alive & ok[:, b]
        if not np.any(updated):
            result.append(Segment(start+1, b, int(np.flatnonzero(alive)[0])))
            start, alive = b, ok[:, b].copy()
        else:
            alive = updated
    result.append(Segment(start+1, h, int(np.flatnonzero(alive)[0])))
    return result


def optimal_selector(f: ArrayLike, epsilon: float) -> list[Segment]:
    f = validate_profiles(f)
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    return compress_feasible(f.max(axis=0)[None, :] - f <= epsilon+1e-12)


def lazy_selector(f: ArrayLike, epsilon: float) -> list[Segment]:
    """Envelope-triggered selector used in the O(1/epsilon) existence proof."""
    f = validate_profiles(f)
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0,1]")
    envelope = f.max(axis=0)
    arm, start, result = int(np.argmax(f[:, 0])), 1, []
    for b in range(1, f.shape[1]):
        if envelope[b] - f[arm, b] > epsilon+1e-12:
            result.append(Segment(start, b, arm))
            arm, start = int(np.argmax(f[:, b])), b+1
    return result+[Segment(start, f.shape[1], arm)]


def robust_gaps(lower: ArrayLike, upper: ArrayLike) -> NDArray[np.float64]:
    """Exact worst-case regret under independent monotone CDF bands.

    The selected arm MUST be excluded from its own competitors. Including it
    creates an unnecessarily conservative certificate, even when m=1.
    """
    lo, hi = validate_profiles(lower), validate_profiles(upper)
    if lo.shape != hi.shape or np.any(lo > hi+1e-10):
        raise ValueError("inconsistent bands")
    if lo.shape[0] == 1:
        return np.zeros_like(lo)
    top = np.argmax(hi, axis=0)
    largest = hi[top, np.arange(hi.shape[1])]
    other = hi.copy()
    other[top, np.arange(hi.shape[1])] = -np.inf
    second = other.max(axis=0)
    competitors = np.broadcast_to(largest, lo.shape).copy()
    competitors[top, np.arange(hi.shape[1])] = second
    return np.maximum(0, competitors-lo)


def empirical_cdf(times: ArrayLike, horizon: int) -> NDArray[np.float64]:
    """Full-cap empirical CDF; H+1 denotes right-censoring, NOT eternal failure."""
    t = np.asarray(times, dtype=int)
    if t.ndim != 1 or len(t) == 0 or np.any(t < 1):
        raise ValueError("positive completion/censoring times are needed")
    return np.cumsum(np.bincount(np.minimum(t,horizon+1), minlength=horizon+2)[1:horizon+1])/len(t)


def hazard_cdf(times: ArrayLike, horizon: int) -> NDArray[np.float64]:
    """Saturated discrete hazard MLE at a common cap; equals empirical CDF."""
    t = np.asarray(times, dtype=int)
    events = np.bincount(np.minimum(t,horizon+1), minlength=horizon+2)[1:horizon+1]
    risk = len(t)-np.r_[0, np.cumsum(events[:-1])]
    hazards = np.divide(events, risk, out=np.zeros(horizon), where=risk > 0)
    return 1-np.cumprod(1-hazards)
