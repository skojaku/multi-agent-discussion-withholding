"""Where the measured team sits on the phase diagram, and which knob is the brake.

``c*`` is the concealment at which a wrong consensus stops being something the
team can talk its way out of. It is not a universal constant and it is not a
property of a model: it is a combination of things measured on one transcript,
in the way a Reynolds number is a combination of measured quantities. What it is
good for is reading which regime the measurement lands in.

    margin = c - c*      > 0  trapped: a wrong consensus, once formed, holds
                         < 0  recoverable: dissent still propagates

Two cautions carried on every result here.

**The sign is only meaningful away from the boundary.** On 120 tasks x 5 agents
x 3 rounds the sign of the margin is recovered 100% of the time when the true
margin exceeds 0.4 and 60% of the time -- a coin flip -- when it is zero. The
bootstrap interval on the margin is therefore not decoration; ``verdict``
returns INCONCLUSIVE when it straddles zero, and callers should not override it.

**c* moves with the trigger convention.** ``gamma`` is estimated by dividing
flips by summed dissent (``fraction``) or by the count of dissenting rows
(``any``), and the two differ by a factor that the transcript does not pin down.
Both are computed, and the honest reading of c* is the range they bracket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import c_star  # noqa: F401  (re-exported: it is a phase quantity)
from .model import Params, basin_boundary, mf_trajectory


@dataclass(frozen=True)
class Lever:
    """One parameter, and how far it would have to move to reach the boundary.

    ``target`` is the value that would put the team exactly on ``margin = 0``
    with the other two held where they were measured; ``change`` is that move as
    a fraction of the measured value. The smallest ``change`` is the cheapest
    intervention, which is the question a team designer actually has.
    """

    name: str
    measured: float
    target: float
    change: float
    feasible: bool

    def __repr__(self):
        if not self.feasible:
            return f"{self.name}: no value of this parameter alone reaches the boundary"
        return (f"{self.name}: {self.measured:.3f} -> {self.target:.3f} "
                f"({self.change * 100:+.0f}%)")


def levers(c: float, a: float, gamma: float) -> list[Lever]:
    """What each of c, a, gamma would have to become to zero out the margin.

    From ``c = gamma / (gamma + a)`` at the boundary:

        c      -> c* itself
        gamma  -> a c / (1 - c)      (raise repair until it outruns concealment)
        a      -> gamma (1 - c) / c  (cut internalisation until nothing holds)
    """
    out = []
    g = max(float(gamma), 0.0) if np.isfinite(gamma) else np.nan
    if np.isfinite(g) and np.isfinite(a) and (g + a) > 0:
        cs = g / (g + a)
        out.append(Lever("c (concealment)", c, cs,
                         (cs - c) / c if c else np.nan, True))
    else:
        out.append(Lever("c (concealment)", c, np.nan, np.nan, False))

    if np.isfinite(c) and c < 1 and np.isfinite(a):
        tgt = a * c / (1 - c)
        out.append(Lever("gamma (net repair)", g, tgt,
                         (tgt - g) / g if g > 0 else np.inf, True))
    else:
        out.append(Lever("gamma (net repair)", g, np.nan, np.nan, False))

    if np.isfinite(c) and c > 0 and np.isfinite(g):
        tgt = g * (1 - c) / c
        out.append(Lever("a (internalisation)", a, tgt,
                         (tgt - a) / a if a else np.inf, True))
    else:
        out.append(Lever("a (internalisation)", a, np.nan, np.nan, False))
    return out


def cheapest_lever(levs) -> str:
    """The knob that needs the smallest fractional move. 'unknown' if none does."""
    ok = [v for v in levs if v.feasible and np.isfinite(v.change)]
    if not ok:
        return "unknown"
    return min(ok, key=lambda v: abs(v.change)).name


def verdict(margin: float, lo: float, hi: float, min_n_ok: bool = True) -> str:
    """TRAPPED / RECOVERABLE / INCONCLUSIVE, and never a bare sign."""
    if not min_n_ok or not np.isfinite(margin):
        return "UNMEASURABLE"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "TRAPPED (no interval)" if margin > 0 else "RECOVERABLE (no interval)"
    if lo > 0:
        return "TRAPPED"
    if hi < 0:
        return "RECOVERABLE"
    return "INCONCLUSIVE"


def as_params(c, a, rho, r, q=5, trigger="fraction") -> Params:
    """Measured rates -> the model's parameter object, for forward prediction."""
    return Params(phi=float(c), lam=float(a), rho=float(min(rho, 1.0)),
                  r=float(r), q=int(q), trigger=trigger)


def p_star(c, a, rho, r, q=5, trigger="fraction") -> float:
    """Initial competence a team needs to recover at this operating point.

    The phase boundary read as "how many agents must already be right". 0.0 means
    any correct minority grows; 1.0 means none does. Costs one vectorised
    mean-field sweep, so it is cheap enough to put on every report.
    """
    if not all(np.isfinite(x) for x in (c, a, rho, r)):
        return np.nan
    return float(basin_boundary(as_params(c, a, rho, r, q, trigger)))


def predict_rounds(p0, c, a, rho, r, q=5, trigger="fraction", n_rounds=10):
    """Mean-field forecast of (private, expressed) accuracy, round by round.

    Run at the MEASURED rates and the run's own starting competence, this is a
    prediction with nothing left to fit -- which is what makes it a test rather
    than a fit. It is a mean field over an annealed panel of q, so it is a
    statement about the model, not about any particular five agents.
    """
    if not all(np.isfinite(x) for x in (p0, c, a, rho, r)):
        return None
    return mf_trajectory(float(p0), as_params(c, a, rho, r, q, trigger),
                         n_rounds=n_rounds)
