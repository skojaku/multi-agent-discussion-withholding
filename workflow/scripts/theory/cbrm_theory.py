"""Concealed-Belief Repair Model (CBRM): two-layer opinion dynamics with a ground truth.

The model couples the private/expressed split of the concealed-voter literature
to a task that has a correct answer, so the observable is *collective accuracy*
rather than consensus time.

Each agent carries two bits: a private belief ``c`` (its actual answer) and an
expressed token ``x`` (what neighbours see). Only ``x`` is observable. Ground
truth is ``c = 1``.

Per synchronous round agent *i* reads the expressed tokens of its neighbours and

1. **expresses.** If the local expressed majority ``M`` contradicts ``c``, the
   agent conceals with probability ``phi`` (``x = M``) and discloses otherwise
   (``x = c``). If ``M`` agrees, ``x = c``.
2. **updates its belief** through exactly one of two channels:

   * *repair* -- an agent that did NOT conceal and can see expressed
     disagreement with its own belief re-derives its answer, landing on the
     truth with probability ``r``. The engagement rate is ``rho`` scaled by how
     much dissent is visible (see ``trigger``).
   * *internalisation* -- an agent that DID conceal never airs the objection
     and, with probability ``lam``, comes to believe what it said (``c <- x``).

**Disclosure buys repair; concealment buys internalisation.** ``phi`` trades one
for the other. Repair is the only route out of a wrong majority and it is fed by
visible disagreement, which concealment destroys -- the positive feedback that
makes the falsified consensus stable above a critical ``phi``.

Why the panel size decides whether a transition exists at all
------------------------------------------------------------
Near the wrong consensus (``E`` = density of correct expressions -> 0) the
expression map is ``E' = P(1-phi) + phi B_q(E)``:

* ``q = 1`` (copy one neighbour, the voter limit) has ``B_1(E) = E``, so
  ``E' = P(1-phi) + phi E`` relaxes to ``E = P``: concealment does not attenuate
  the expressed dissent at all, only slows it. The growth factor at the wrong
  consensus works out to ``1 + (1-phi) rho (2r-1) > 1`` for every ``phi < 1``
  and every ``lam``: the falsified consensus is never stable, so there is no
  transition. This reproduces the known result that the concealed voter model
  sits in the voter universality class.
* ``q >= 3`` has ``B_q(E) = O(E^2)``, so ``E ~ P(1-phi)``: concealment
  multiplicatively suppresses visible dissent, the repair current it feeds
  shrinks with it, and above ``phi_c`` the wrong consensus becomes stable.

So the nonlinearity of *local majority aggregation* is what creates the trapped
phase. ``q`` is a genuine control parameter, not a modelling detail.

Parameters
----------
p        initial competence, P(c_i = 1) at t = 0, iid across agents.
phi      concealment probability when the local majority contradicts the belief.
lam      internalisation rate for an agent that concealed.
rho      deliberation rate when disagreement is visible and nothing was concealed.
r        accuracy of a re-derivation; r > 1/2 makes deliberation truth-seeking.
q        panel size in the mean field / annealed panel topology (odd).
trigger  how visible dissent gates deliberation:
         ``"fraction"`` -- engagement is proportional to the fraction of
         neighbours expressing something else (social-impact convention, degree
         independent; the default);
         ``"any"``      -- a single dissenting voice is enough (Asch ally).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import comb

import numpy as np

TRIGGERS = ("fraction", "any")
VARIANTS = ("full", "no_repair", "no_internalize", "voter")


@dataclass(frozen=True)
class Params:
    phi: float = 0.5
    lam: float = 0.5
    rho: float = 1.0
    r: float = 0.8
    q: int = 3
    trigger: str = "fraction"
    variant: str = "full"

    def effective(self) -> "Params":
        """Variants are parameter restrictions, not separate code paths."""
        if self.variant == "no_repair":
            return replace(self, rho=0.0, variant="full")
        if self.variant == "no_internalize":
            return replace(self, lam=0.0, variant="full")
        if self.variant == "voter":
            return replace(self, q=1, variant="full")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- mean field
def _panel_weights(E, q: int):
    """w[k] = P(k of q iid Bernoulli(E) tokens are 1), shape (q+1,) + E.shape."""
    E = np.asarray(E, dtype=float)
    ks = np.arange(q + 1).reshape((q + 1,) + (1,) * E.ndim)
    coef = np.array([comb(q, k) for k in range(q + 1)]).reshape(ks.shape)
    return coef * E**ks * (1.0 - E) ** (q - ks)


def majority_prob(E, q: int):
    """P(majority of q iid Bernoulli(E) tokens is 1). q odd, so no ties."""
    if q % 2 == 0:
        raise ValueError("q must be odd so the panel majority is well defined")
    w = _panel_weights(E, q)
    return w[(q // 2) + 1:].sum(axis=0)


def mf_step(P, E, prm: Params):
    """One synchronous round of the mean-field map. Vectorised over (P, E).

    The panel is q tokens drawn iid Bernoulli(E) -- the annealed limit of the
    agent-based model. The sum over panel compositions k is exact, so the
    ``fraction`` and ``any`` triggers need no approximation.
    """
    prm = prm.effective()
    P = np.asarray(P, dtype=float)
    E = np.asarray(E, dtype=float)
    q, phi, lam, rho, r = prm.q, prm.phi, prm.lam, prm.rho, prm.r

    w = _panel_weights(E, q)  # (q+1, ...)
    k = np.arange(q + 1).reshape((q + 1,) + (1,) * np.ndim(E))
    maj_is_1 = (2 * k > q).astype(float)  # q odd: no ties

    if prm.trigger == "fraction":
        trig1 = rho * (q - k) / q  # a c=1 agent is dissented against by the 0s
        trig0 = rho * k / q
    elif prm.trigger == "any":
        trig1 = rho * (k < q).astype(float)
        trig0 = rho * (k > 0).astype(float)
    else:
        raise ValueError(f"unknown trigger {prm.trigger!r}")

    keep1 = 1.0 - trig1 + trig1 * r  # belief survives a re-derivation
    gain1 = trig0 * r  # a wrong agent re-derives its way to the truth

    # c = 1: contradicted when the panel majority is 0.
    stay1 = np.where(
        maj_is_1 > 0,
        keep1,
        phi * (1.0 - lam) + (1.0 - phi) * keep1,
    )
    # c = 0: contradicted when the panel majority is 1. Conceding to a correct
    # majority is the one case where internalisation helps.
    gain0 = np.where(
        maj_is_1 > 0,
        phi * lam + (1.0 - phi) * gain1,
        gain1,
    )

    B = (w * maj_is_1).sum(axis=0)
    E_next = P * (1.0 - phi) + phi * B
    P_next = P * (w * stay1).sum(axis=0) + (1.0 - P) * (w * gain0).sum(axis=0)
    return P_next, E_next


def mf_run(p0, prm: Params, n_rounds: int = 5000, tol: float = 1e-13):
    """Iterate from (P, E) = (p0, p0) to the fixed point. p0 may be an array."""
    P = np.asarray(p0, dtype=float).copy()
    E = P.copy()
    for _ in range(n_rounds):
        P_new, E_new = mf_step(P, E, prm)
        done = (np.max(np.abs(P_new - P)) < tol) and (np.max(np.abs(E_new - E)) < tol)
        P, E = P_new, E_new
        if done:
            break
    return P, E


def mf_trajectory(p0: float, prm: Params, n_rounds: int = 60):
    """Round-by-round (P, E) from a single initial condition."""
    P, E = float(p0), float(p0)
    out = [(P, E)]
    for _ in range(n_rounds):
        P, E = mf_step(np.array(P), np.array(E), prm)
        out.append((float(P), float(E)))
    return np.array(out)


# Largest concealment the fixed-point solver is asked about. At phi = 1 the
# expression map no longer determines P, so the root finder there is solving a
# degenerate problem; 0.999 is inside the regime and far past any phi a team
# reaches.
PHI_MAX = 0.999


def basin_boundary(prm: Params, grid: int = 257, refine: int = 2,
                   n_rounds: int = 2000) -> float:
    """Smallest initial competence p whose orbit ends on the correct side.

    The phase boundary: 0.0 when every p > 0 recovers, 1.0 when none does.

    ``mf_run`` is vectorised over p, so a whole grid of initial conditions costs
    one run. Bracketing on a grid and then refining inside the bracket is far
    cheaper than scalar bisection, which pays the map's critical slowing-down
    near the separatrix once per bisection step.
    """
    # p = 0 is itself a fixed point, so the scan starts just above it: the
    # question is whether an arbitrarily small correct minority can grow.
    lo, hi = 1e-9, 1.0
    for _ in range(refine + 1):
        ps = np.linspace(lo, hi, grid)
        _, E = mf_run(ps, prm, n_rounds=n_rounds)
        above = np.flatnonzero(E > 0.5)
        if above.size == 0:
            return 1.0
        i = int(above[0])
        if i == 0:
            return 0.0 if lo <= 1e-9 else float(lo)
        lo, hi = float(ps[i - 1]), float(ps[i])
    return 0.5 * (lo + hi)


def phi_absorbing(prm: Params) -> float:
    """Concealment at which the *fully* falsified consensus becomes stable.

    Linearise the map about (P, E) = (0, 0). For q >= 3, B_q(E) = O(E^2) so the
    expression map relaxes to E = P(1 - phi), and the growth factor of P is

        phi (1 - lam) + (1 - phi) S,     S = 1 - rho(1 - r) + rho r * kappa,

    with kappa = q for the ``any`` trigger (a wrong agent repairs if any of its
    q slots carries a disclosed correct token, probability q E + O(E^2)) and
    kappa = 1 for the ``fraction`` trigger (engagement rate rho E). Setting the
    factor to 1,

        phi_c = (S - 1) / (S - 1 + lam).

    Two edge cases fall out of the same algebra. lam = 0 gives phi_c = 1:
    without internalisation nothing stabilises the falsified consensus. q = 1
    also gives phi_c = 1, for a different reason -- B_1(E) = E makes the
    expression map marginal, E relaxes to P rather than P(1 - phi), and the
    growth factor is 1 + (1 - phi) rho (2r - 1) > 1 for every phi < 1.

    This is where the wrong consensus becomes *absorbing*. The collective
    accuracy collapses earlier, at the saddle-node where a low-accuracy
    attractor first appears -- see ``basin_boundary``.
    """
    prm = prm.effective()
    if prm.q == 1:
        return 1.0
    kappa = prm.q if prm.trigger == "any" else 1.0
    S = 1.0 - prm.rho * (1.0 - prm.r) + prm.rho * prm.r * kappa
    if S <= 1.0:
        return 0.0
    if prm.lam <= 0.0:
        return 1.0
    return (S - 1.0) / (S - 1.0 + prm.lam)


def fixed_points(prm: Params, grid: int = 20001, refine: int = 60):
    """All fixed points of the map, with their stability.

    At a fixed point the expression equation ``E = P(1-phi) + phi B_q(E)`` can be
    solved for P, so the 2D problem collapses to one root-finding problem in E:

        P = (E - phi B_q(E)) / (1 - phi),   h(E) = P_next(P, E) - P = 0.

    Returns a list of ``(E, P, stable)`` sorted by E. ``phi = 1`` is excluded --
    there the expression map no longer determines P.
    """
    prm = prm.effective()
    if prm.phi >= 1.0 - 1e-12:
        return []
    Es = np.linspace(0.0, 1.0, grid)
    Ps = (Es - prm.phi * majority_prob(Es, prm.q)) / (1.0 - prm.phi)
    ok = (Ps >= -1e-12) & (Ps <= 1.0 + 1e-12)
    Ps = np.clip(Ps, 0.0, 1.0)
    h = mf_step(Ps, Es, prm)[0] - Ps
    h = np.where(ok, h, np.nan)

    def h_at(E):
        P = np.clip((E - prm.phi * majority_prob(np.array(E), prm.q)) / (1.0 - prm.phi),
                    0.0, 1.0)
        return float(mf_step(np.array(P), np.array(E), prm)[0] - P), float(P)

    # (0, 0) and (1, 1) are fixed points of every parameterisation and sit on the
    # boundary of the scan, where a sign change cannot be seen.
    roots = [0.0, 1.0]
    for i in range(grid - 1):
        a, b = h[i], h[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a * b < 0:
            lo, hi = Es[i], Es[i + 1]
            for _ in range(refine):
                mid = 0.5 * (lo + hi)
                if h_at(mid)[0] * a > 0:
                    lo = mid
                else:
                    hi = mid
            roots.append(0.5 * (lo + hi))

    out = []
    for E in sorted(set(np.round(roots, 9))):
        _, P = h_at(E)
        # 2x2 Jacobian by finite differences, stepped inward at the boundaries
        # so that (0, 0) and (1, 1) are not evaluated outside [0, 1].
        eps = 1e-6
        J = np.zeros((2, 2))
        for c, var in enumerate("PE"):
            v = P if var == "P" else E
            lo, hi = (v, v + eps) if v < eps else (v - eps, min(v, 1.0))
            args = [(np.array(lo if var == "P" else P), np.array(lo if var == "E" else E)),
                    (np.array(hi if var == "P" else P), np.array(hi if var == "E" else E))]
            f0 = mf_step(*args[0], prm)
            f1 = mf_step(*args[1], prm)
            J[0, c] = float(f1[0] - f0[0]) / (hi - lo)
            J[1, c] = float(f1[1] - f0[1]) / (hi - lo)
        stable = bool(np.max(np.abs(np.linalg.eigvals(J))) < 1.0)
        out.append((float(E), float(P), stable))
    return sorted(out)


def phi_saddle(prm: Params, iters: int = 40) -> float:
    """Concealment at which a low-accuracy attractor first appears.

    This is the *physical* transition: the smallest phi for which some initial
    competence fails to recover, i.e. the saddle-node that opens the trapped
    basin. It sits at or just below ``phi_absorbing``, which is where the wrong
    consensus additionally becomes fully absorbing (E = 0 exactly).

    Whether a trap exists *at all* is settled on the fixed points and not on a
    round budget. Asked with ``basin_boundary`` alone, the voter limit q = 1
    reported a saddle-node at 0.899 that is not there: the growth factor at
    (0, 0) is 1 + (1 - phi) rho (2 r - 1), which is 1.003 at phi = 0.95, so a
    seed of 1e-9 needs ~7000 rounds to climb across 1/2 and a group that does
    recover reads as trapped against the 2000-round budget. The map for q = 1
    has no stable fixed point below E = 1/2 at any concealment, and that is the
    statement "the voter limit has no transition". lambda = 0 is the opposite
    case and keeps its transition at 0.856: the origin is unstable there too,
    but the map still has an interior stable fixed point at E = 0.11.

    The existence scan stops at ``PHI_MAX``. At phi = 1 nothing anyone says
    depends on what they believe, the expression map stops determining P, and
    the root finder is handed a degenerate problem.
    """
    prm = prm.effective()
    if not any(any(stable and E < 0.5
                   for E, _, stable in fixed_points(replace(prm, phi=float(phi))))
               for phi in np.linspace(0.0, PHI_MAX, 60)):
        return 1.0
    if basin_boundary(replace(prm, phi=0.0)) > 0:
        return 0.0
    if basin_boundary(replace(prm, phi=1.0 - 1e-9)) == 0:
        return 1.0
    a, b = 0.0, 1.0 - 1e-9
    for _ in range(iters):
        m = 0.5 * (a + b)
        if basin_boundary(replace(prm, phi=m)) > 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def phi_star(p: float, prm: Params, iters: int = 40) -> float:
    """Critical concealment for a group that starts at competence p.

    The phase boundary read the other way round: bisection on phi holding p
    fixed. Returns 1.0 if the group recovers at every phi.
    """

    def recovers(phi: float) -> bool:
        _, E = mf_run(np.array([p]), replace(prm, phi=phi))
        return bool(E[0] > 0.5)

    if not recovers(0.0):
        return 0.0
    if recovers(1.0):
        return 1.0
    a, b = 0.0, 1.0
    for _ in range(iters):
        m = 0.5 * (a + b)
        if recovers(m):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# ------------------------------------------------------------------- topologies
TOPOLOGIES = ("panel", "complete", "er", "ring", "ws", "star", "chain")


def adjacency(topology: str, N: int, rng: np.random.Generator, k: int = 4,
              rewire: float = 0.1) -> np.ndarray:
    """0/1 adjacency; A[i, j] = 1 means i observes j. ``chain`` is directed."""
    A = np.zeros((N, N), dtype=np.int8)
    if topology in ("complete", "panel"):
        A[:] = 1
        np.fill_diagonal(A, 0)
    elif topology == "star":
        A[0, 1:] = 1
        A[1:, 0] = 1
    elif topology == "ring":
        idx = np.arange(N)
        for d in range(1, k // 2 + 1):
            A[idx, (idx + d) % N] = 1
            A[idx, (idx - d) % N] = 1
    elif topology == "chain":  # the pipeline: agent i reads only agent i-1
        for i in range(1, N):
            A[i, i - 1] = 1
    elif topology == "ws":
        A = adjacency("ring", N, rng, k=k)
        for i in range(N):
            for jj in np.flatnonzero(A[i]):
                if rng.random() < rewire:
                    cand = int(rng.integers(N))
                    if cand != i and A[i, cand] == 0:
                        A[i, jj] = 0
                        A[i, cand] = 1
    elif topology == "er":
        prob = min(1.0, k / max(N - 1, 1))
        M = np.triu((rng.random((N, N)) < prob).astype(np.int8), 1)
        A = M + M.T
    else:
        raise ValueError(f"unknown topology {topology!r}")
    return A


# ------------------------------------------------------------- agent-based model
def simulate(N: int, p: float, prm: Params, topology: str = "panel",
             n_rounds: int = 80, n_reps: int = 200, seed: int = 0, k: int = 4,
             rewire: float = 0.1, record_traj: bool = False):
    """Run ``n_reps`` independent groups; return summary statistics.

    ``accuracy`` -- the fraction of replicates whose final expressed majority is
    correct -- is the collective observable. ``gap`` is the pluralistic-ignorance
    signature: mean private accuracy minus mean expressed accuracy.
    """
    prm = prm.effective()
    rng = np.random.default_rng(seed)
    phi, lam, rho, r, q = prm.phi, prm.lam, prm.rho, prm.r, prm.q
    frac_trigger = prm.trigger == "fraction"

    c = (rng.random((n_reps, N)) < p).astype(np.int8)
    x = c.copy()

    if topology != "panel":
        # One graph per replicate: for er/ws the graph disorder is part of the model.
        A = np.stack([adjacency(topology, N, rng, k=k, rewire=rewire)
                      for _ in range(n_reps)]).astype(np.int16)
        deg = A.sum(axis=2)
        isolated = deg == 0  # the pipeline head has nobody to look at
        deg_safe = np.maximum(deg, 1)

    traj = []
    for _ in range(n_rounds):
        if topology == "panel":
            # q neighbours sampled without self-loops, resampled every round.
            idx = rng.integers(0, N - 1, size=(n_reps, N, q))
            idx = idx + (idx >= np.arange(N)[None, :, None])
            # Gather straight out of x: broadcasting x to (n_reps, N, N) first
            # would allocate N times more memory than the sample needs.
            n1 = x[np.arange(n_reps)[:, None, None], idx].sum(axis=2)
            dd = np.full((n_reps, N), q, dtype=np.int16)
            iso = np.zeros((n_reps, N), dtype=bool)
        else:
            n1 = np.einsum("rij,rj->ri", A, x.astype(np.int16))
            dd, iso = deg_safe, isolated

        M = np.where(2 * n1 > dd, 1, np.where(2 * n1 < dd, 0, c))
        M = np.where(iso, c, M).astype(np.int8)
        n_dissent = np.where(c == 1, dd - n1, n1)  # neighbours expressing != c
        if frac_trigger:
            trig = rho * n_dissent / dd
        else:
            trig = rho * (n_dissent > 0)
        trig = np.where(iso, 0.0, trig)

        contradicted = M != c
        conceal = contradicted & (rng.random((n_reps, N)) < phi)
        x = np.where(conceal, M, c).astype(np.int8)

        # Exactly one belief channel fires per agent per round.
        internalise = conceal & (rng.random((n_reps, N)) < lam)
        deliberate = (~conceal) & (rng.random((n_reps, N)) < trig)
        redraw = (rng.random((n_reps, N)) < r).astype(np.int8)
        c = np.where(internalise, x, np.where(deliberate, redraw, c)).astype(np.int8)

        if record_traj:
            traj.append((float(c.mean()), float(x.mean())))

    m_priv = c.mean(axis=1)
    m_expr = x.mean(axis=1)
    out = {
        "m_priv": float(m_priv.mean()),
        "m_expr": float(m_expr.mean()),
        "gap": float((m_priv - m_expr).mean()),
        "accuracy": float((m_expr > 0.5).mean()),
        "n_reps": int(n_reps),
    }
    if record_traj:
        out["traj"] = np.array(traj)
    return out


def abm_phase_boundary(N: int, prm: Params, topology: str = "panel",
                       n_rounds: int = 80, n_reps: int = 200, seed: int = 0,
                       grid: int = 41) -> float:
    """Initial competence at which the finite-N recovery probability crosses 1/2."""
    ps = np.linspace(0.0, 1.0, grid)
    acc = np.array([simulate(N, p, prm, topology, n_rounds, n_reps, seed)["accuracy"]
                    for p in ps])
    above = np.flatnonzero(acc >= 0.5)
    if above.size == 0:
        return 1.0
    i = int(above[0])
    if i == 0:
        return 0.0
    y0, y1 = acc[i - 1], acc[i]
    if y1 == y0:
        return float(ps[i])
    return float(ps[i - 1] + (0.5 - y0) * (ps[i] - ps[i - 1]) / (y1 - y0))
