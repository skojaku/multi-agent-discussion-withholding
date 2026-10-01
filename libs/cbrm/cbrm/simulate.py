"""Generate transcripts from known parameters, so the estimators can be checked.

The estimators in this package are conditional frequencies over a transcript.
Nothing about that guarantees they recover the rates that generated it -- the
alignment between "the belief that faced the majority" and "the row before this
one" has to be right, the projection onto the model's state space has to be
right, and the moment estimator for repair has to divide by the right thing.
The only way to know is to run them on logs whose parameters are known.

``cbrm.model.simulate`` returns summary statistics and throws the trajectory
away, so it cannot be used for this. ``simulate_events`` runs the same dynamics
and emits the per-agent rows instead.

Alignment with the agent-based model, which is what makes the recovery test
meaningful: inside the model's round loop ``c`` is the belief *entering* the
round -- the one that faces the majority and that the estimator reads off the
previous row -- and ``x`` on entry is the previous round's expression, which is
exactly what a neighbour could have read. So

    round 0     branch "r0",  private = public = c_0,  no neighbours
    round t     branch cond,  private = c_t,  public = x_t,
                nb_public = the neighbours' x_{t-1}

is not an approximation of the model; it is a faithful serialisation of it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .model import Params as _Params
from .model import adjacency

TRUTH_LABEL = "A"


class Params(_Params):
    pass


def params_from_rates(c: float, a: float, rho: float, r: float, q: int = 5,
                      trigger: str = "fraction") -> _Params:
    """Paper notation (c, a, rho, r) -> the model's (phi, lam, rho, r)."""
    return _Params(phi=c, lam=a, rho=rho, r=r, q=q, trigger=trigger)


_Params.from_rates = staticmethod(params_from_rates)


def _labeller(n_wrong_labels: int):
    """Map the model's bit onto an answer label.

    ``n_wrong_labels > 1`` spreads the c = 0 state over several distinct wrong
    answers, which is what a real multiple-choice item does and what breaks the
    letter-level projection. The label a wrong agent uses is a deterministic
    function of (task, agent), so an agent keeps its wrong answer across rounds
    instead of jittering between distractors.
    """
    wrong = [chr(ord("B") + i) for i in range(max(int(n_wrong_labels), 1))]

    def lab(bit, task, agent):
        if bit == 1:
            return TRUTH_LABEL
        return wrong[(task * 7919 + agent * 104729) % len(wrong)]

    return lab


def simulate_events(N: int, p: float, prm: _Params, n_tasks: int = 200,
                    n_rounds: int = 3, seed: int = 0, topology: str = "complete",
                    condition: str = "C0", k: int = 4, n_wrong_labels: int = 1,
                    model_name: str = "cbrm-sim") -> list[dict]:
    """One 'task' is one independent group. Returns rows in the events schema."""
    prm = prm.effective()
    rng = np.random.default_rng(seed)
    phi, lam, rho, r, q = prm.phi, prm.lam, prm.rho, prm.r, prm.q
    frac = prm.trigger == "fraction"
    R = int(n_tasks)
    lab = _labeller(n_wrong_labels)

    c = (rng.random((R, N)) < p).astype(np.int8)
    x = c.copy()

    common = dict(answer=TRUTH_LABEL, topology=topology, model=model_name)
    rows = [dict(task_id=f"t{t:05d}", agent=int(i), round=0, branch="r0",
                 private=lab(c[t, i], t, i), public=lab(c[t, i], t, i),
                 nb_public=None, **common)
            for t in range(R) for i in range(N)]

    if topology != "panel":
        A = np.stack([adjacency(topology, N, rng, k=k) for _ in range(R)])
        A = A.astype(np.int16)
        deg = A.sum(axis=2)
        isolated = deg == 0
        deg_safe = np.maximum(deg, 1)

    for rnd in range(1, n_rounds + 1):
        x_seen = x.copy()                 # what neighbours could read this round
        if topology == "panel":
            idx = rng.integers(0, N - 1, size=(R, N, q))
            idx = idx + (idx >= np.arange(N)[None, :, None])
            n1 = x[np.arange(R)[:, None, None], idx].sum(axis=2)
            dd = np.full((R, N), q, dtype=np.int16)
            iso = np.zeros((R, N), dtype=bool)
            nb_of = lambda t, i: [int(j) for j in idx[t, i]]          # noqa: E731
        else:
            n1 = np.einsum("rij,rj->ri", A, x.astype(np.int16))
            dd, iso = deg_safe, isolated
            nb_of = lambda t, i: [int(j) for j in np.flatnonzero(A[t, i])]  # noqa: E731

        M = np.where(2 * n1 > dd, 1, np.where(2 * n1 < dd, 0, c))
        M = np.where(iso, c, M).astype(np.int8)
        n_dissent = np.where(c == 1, dd - n1, n1)
        trig = rho * n_dissent / dd if frac else rho * (n_dissent > 0)
        trig = np.where(iso, 0.0, trig)

        contradicted = M != c
        conceal = contradicted & (rng.random((R, N)) < phi)
        x_new = np.where(conceal, M, c).astype(np.int8)

        # Exactly one belief channel fires per agent per round.
        internalise = conceal & (rng.random((R, N)) < lam)
        deliberate = (~conceal) & (rng.random((R, N)) < trig)
        redraw = (rng.random((R, N)) < r).astype(np.int8)
        c_new = np.where(internalise, x_new,
                         np.where(deliberate, redraw, c)).astype(np.int8)

        for t in range(R):
            for i in range(N):
                nb = nb_of(t, i)
                rows.append(dict(
                    task_id=f"t{t:05d}", agent=int(i), round=rnd, branch=condition,
                    private=lab(c_new[t, i], t, i), public=lab(x_new[t, i], t, i),
                    nb_public=[lab(x_seen[t, j], t, j) for j in nb],
                    neighbours=nb, **common))
        c, x = c_new, x_new
    return rows


def write_events(rows, path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(rounds=int(max(r["round"] for r in rows)), probe="joint",
                model=rows[0].get("model", "cbrm-sim"),
                topology=rows[0].get("topology", "complete"), synthetic=True)
    p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    return str(p)


def transcript(**kw):
    """Convenience: simulate straight into a Transcript, no file on disk."""
    from .io import Transcript
    rows = simulate_events(**kw)
    return Transcript(rows=rows, source="synthetic", meta=dict(
        rounds=int(max(r["round"] for r in rows)), probe="joint", synthetic=True))


# --------------------------------------------------------------- K-way choice
def simulate_events_k(N: int, p: float, prm: _Params, K: int = 3,
                      concentration: float = 1.0, n_tasks: int = 200,
                      n_rounds: int = 3, seed: int = 0, topology: str = "complete",
                      condition: str = "C0", k: int = 4,
                      model_name: str = "cbrm-sim-k") -> list[dict]:
    """The same rules on K options instead of two. Only two things change.

    **The majority becomes a plurality.** ``m_i`` is the most common answer among
    the expressions an agent can see, with no majority declared on a tie. Nothing
    else about concealment or internalisation moves: an agent whose plurality
    contradicts its belief conceals with probability ``phi`` and, having
    concealed, comes to believe what it said with probability ``lam``. The
    reconsideration trigger is unchanged too -- ``d`` is the fraction of visible
    expressions that differ from the agent's own belief, and a neighbour holding
    a *different wrong answer* is dissent just as much as one holding the truth.

    **Re-derivation needs somewhere to land.** With one wrong answer there was
    nothing to decide; with K - 1 there is. The rule here is persuasion: a
    re-derivation lands on the truth with probability ``r`` and otherwise on one
    of the wrong answers it can currently see, uniformly. That is what the
    transcripts show -- on MMLU-Pro the measured ``r`` runs 0.15 to 0.37 and the
    losers flow to whichever distractor the room is already holding.

    ``concentration`` is the second initial condition K > 2 introduces, and it is
    the one that decides whether any of this behaves like the two-choice model.
    At 1.0 every wrong agent starts on the same distractor; at 0.0 the wrong side
    is spread uniformly over all K - 1 of them. A concentrated wrong side is the
    two-choice case and the trap closes at the same c; a split one recovers far
    above it, because agents disagreeing about *which* wrong answer is right
    still trigger each other's re-derivations, and each of those lands on the
    truth with probability r.

    So the two-choice model is the worst case of the K-choice one, and the force
    that concentrates the wrong side is conformity itself. LLM teams start
    concentrated -- correlated priors put them on the same distractor -- which is
    why the reduction describes the regime that actually applies rather than
    approximating it away.
    """
    prm = prm.effective()
    if K < 2:
        raise ValueError("K must be at least 2")
    rng = np.random.default_rng(seed)
    phi, lam, rho, r = prm.phi, prm.lam, prm.rho, prm.r
    q = prm.q
    frac = prm.trigger == "fraction"
    R = int(n_tasks)
    labels = [TRUTH_LABEL] + [chr(ord("B") + i) for i in range(K - 1)]
    TRUTH = 0                                   # index of the correct option

    # Initial state: p correct, the rest on the wrong options. ``concentration``
    # interpolates between all of them on option 1 and a uniform spread.
    b = np.zeros((R, N), dtype=np.int16)
    correct = rng.random((R, N)) < p
    spread = rng.integers(1, K, size=(R, N)) if K > 2 else np.ones((R, N), int)
    clumped = np.ones((R, N), dtype=np.int16)
    pick_clump = rng.random((R, N)) < concentration
    b = np.where(correct, TRUTH, np.where(pick_clump, clumped, spread)).astype(np.int16)
    x = b.copy()

    common = dict(answer=TRUTH_LABEL, topology=topology, model=model_name)
    rows = [dict(task_id=f"t{t:05d}", agent=int(i), round=0, branch="r0",
                 private=labels[b[t, i]], public=labels[b[t, i]],
                 nb_public=None, **common)
            for t in range(R) for i in range(N)]

    if topology != "panel":
        A = np.stack([adjacency(topology, N, rng, k=k) for _ in range(R)])
        A = A.astype(np.int16)
        deg = A.sum(axis=2)
        isolated = deg == 0

    for rnd in range(1, n_rounds + 1):
        x_seen = x.copy()
        if topology == "panel":
            idx = rng.integers(0, N - 1, size=(R, N, q))
            idx = idx + (idx >= np.arange(N)[None, :, None])
            panel = x[np.arange(R)[:, None, None], idx]
            iso = np.zeros((R, N), dtype=bool)
            nb_of = lambda t, i: [int(j) for j in idx[t, i]]            # noqa: E731
        else:
            iso = isolated
            nb_of = lambda t, i: [int(j) for j in np.flatnonzero(A[t, i])]  # noqa: E731
            panel = None

        # Counts of each option among the expressions this agent can see.
        cnt = np.zeros((R, N, K), dtype=np.int16)
        if panel is not None:
            for kk in range(K):
                cnt[:, :, kk] = (panel == kk).sum(axis=2)
        else:
            onehot = np.zeros((R, N, K), dtype=np.int16)
            np.put_along_axis(onehot, x[:, :, None].astype(np.int64), 1, axis=2)
            cnt = np.einsum("rij,rjk->rik", A, onehot)
        seen_n = cnt.sum(axis=2)
        top = cnt.max(axis=2)
        # A tie leaves the plurality undefined; the agent then faces no majority.
        tied = (cnt == top[:, :, None]).sum(axis=2) > 1
        M = np.where(tied | iso | (seen_n == 0), b, cnt.argmax(axis=2)).astype(np.int16)
        own_seen = np.take_along_axis(cnt, b[:, :, None].astype(np.int64), axis=2)[:, :, 0]
        n_dissent = seen_n - own_seen
        with np.errstate(invalid="ignore", divide="ignore"):
            trig = (rho * n_dissent / np.maximum(seen_n, 1) if frac
                    else rho * (n_dissent > 0))
        trig = np.where(iso | (seen_n == 0), 0.0, trig)

        contradicted = (M != b) & ~tied & ~iso & (seen_n > 0)
        conceal = contradicted & (rng.random((R, N)) < phi)
        x_new = np.where(conceal, M, b).astype(np.int16)

        internalise = conceal & (rng.random((R, N)) < lam)
        deliberate = (~conceal) & (rng.random((R, N)) < trig)
        hits_truth = rng.random((R, N)) < r
        # A failed re-derivation is persuaded onto a wrong answer it can see.
        wrong_cnt = cnt.copy()
        wrong_cnt[:, :, TRUTH] = 0
        wrong_tot = wrong_cnt.sum(axis=2)
        u = rng.random((R, N)) * np.maximum(wrong_tot, 1)
        pick = (wrong_cnt.cumsum(axis=2) > u[:, :, None]).argmax(axis=2)
        landed = np.where(hits_truth, TRUTH,
                          np.where(wrong_tot > 0, pick, b)).astype(np.int16)
        b_new = np.where(internalise, x_new,
                         np.where(deliberate, landed, b)).astype(np.int16)

        for t in range(R):
            for i in range(N):
                nb = nb_of(t, i)
                rows.append(dict(
                    task_id=f"t{t:05d}", agent=int(i), round=rnd, branch=condition,
                    private=labels[b_new[t, i]], public=labels[x_new[t, i]],
                    nb_public=[labels[x_seen[t, j]] for j in nb],
                    neighbours=nb, **common))
        b, x = b_new, x_new
    return rows


def transcript_k(**kw):
    """simulate_events_k straight into a Transcript."""
    from .io import Transcript
    rows = simulate_events_k(**kw)
    return Transcript(rows=rows, source="synthetic-k", meta=dict(
        rounds=int(max(r["round"] for r in rows)), probe="joint", synthetic=True,
        K=kw.get("K", 3), concentration=kw.get("concentration", 1.0)))
