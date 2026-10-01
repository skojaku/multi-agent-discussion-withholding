"""The four rates, as frequencies in the decision table. Nothing here is fit.

Two design decisions are worth stating up front, because both were forced by
measurement, not taste.

**Why the legacy concealment estimator is the primary one.** Two definitions
have been used in this codebase:

    legacy   P(expressed = M | M contradicts the belief held ENTERING the round)
    strict   P(expressed = M and expressed != belief held NOW | belief now != M)

``strict`` was introduced on the worry that ``legacy`` counts a genuinely
persuaded agent -- one whose new private answer already IS the majority -- as
having concealed. Run both against logs generated from known parameters and the
worry inverts: ``legacy`` is unbiased at every internalisation rate, while
``strict`` falls away in proportion to it, because an agent that concealed and
then internalised is dropped from ``strict``'s denominator entirely. Measured on
synthetic logs with a true c = 0.5:

    a = 0.0   legacy 0.503   strict 0.594
    a = 0.5   legacy 0.521   strict 0.485
    a = 1.0   legacy 0.515   strict 0.265

The "persuasion" the strict estimator removes is not contamination. It IS the
internalisation channel, and the fraction of legacy hits that show a real
private/public gap comes out at exactly ``1 - a``. So both are reported:
``legacy`` as the estimate, ``strict`` as a lower bound, and the gap between
them as a consistency check on ``a``.

**Why rho and r come with a trigger assumption and gamma comes as a range.**
Repair is a moment estimator, not a frequency: an agent that sees dissent
re-derives only sometimes, and if it does not it keeps the belief it had. Under
the ``fraction`` convention the engagement rate is ``rho * d``, so dividing
flips by the summed ``d`` recovers ``rho``; under ``any`` a single dissenter
suffices, so dividing by the count of rows with ``d > 0`` is what recovers it.
Get the convention wrong and rho saturates: on synthetic logs generated with the
``any`` rule, the ``fraction`` estimator returns rho = 1.0 against a true 0.6
and pushes c* from 0.545 to 0.679.

The ratio ``r`` survives the choice -- both readings scale numerator and
denominator by the same factor -- so ``r`` is reported once and ``gamma`` twice.
``c*`` is therefore a range, and its width is a real statement about what the
transcript can and cannot pin down.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import majority

TRIGGERS = ("fraction", "any")


@dataclass(frozen=True)
class Estimate:
    """A rate, its interval, and the denominator that produced it.

    ``n`` is not decoration. Every rate here is a conditional probability and
    the conditioning event does not always occur; ``n = 0`` is the honest answer
    for a team that was never internally split, and callers must be able to see
    the difference between that and a rate of zero.
    """

    value: float
    lo: float = np.nan
    hi: float = np.nan
    n: int = 0
    note: str = ""

    def __repr__(self):
        v = "  nan" if not np.isfinite(self.value) else f"{self.value:5.3f}"
        ci = ("" if not np.isfinite(self.lo)
              else f" [{self.lo:.3f}, {self.hi:.3f}]")
        return f"{v}{ci} n={self.n}{(' ' + self.note) if self.note else ''}"


def _rate(hits, trials):
    return float(hits) / float(trials) if trials else np.nan


# ------------------------------------------------------------------ raw counts
def counts(dec: pd.DataFrame, repair_requires_majority: bool = False) -> dict:
    """Sufficient statistics. Every estimator below is a ratio of these.

    Splitting counting from dividing is what makes the cluster bootstrap cheap:
    a resample sums per-task count vectors instead of re-walking the transcript.

    ``repair_requires_majority`` reproduces the estimator the 0831 analysis
    scripts used, which skipped a row whenever the panel it saw was tied and
    so had no well-defined majority. That skip is correct for concealment --
    with no majority there is nothing to concede to -- and wrong for repair,
    which the model gates on visible dissent alone. A tied panel is the most
    dissenting panel there is, so dropping those rows removes the strongest
    repair opportunities from the denominator. On five-agent teams, where
    every agent sees four neighbours and 2-2 splits are common, this moved
    rho by up to 0.24. Default False; set True only to reproduce old numbers.
    """
    if dec.empty:
        return dict.fromkeys(
            ["n_contra", "n_conceal", "n_gap", "n_strict_denom", "n_strict_hit",
             "n_conc_pairs", "n_internalise", "sum_d_wrong", "flip_up",
             "n_wrong_rows", "sum_d_right", "flip_down", "n_right_rows",
             "n_rows"], 0)

    M_known = dec["M"].notna()
    contradicted = M_known & (dec["M"] != dec["c_prev"])
    concealed = contradicted & (dec["pub"] == dec["M"])
    # An agent that concealed can only move its belief by internalising, so the
    # concealed rows are exactly the internalisation denominator.
    internalised = concealed & (dec["priv"] == dec["pub"])

    # strict: conditioned on the answer the agent holds NOW, and requiring a
    # real private/public gap.
    s_denom = M_known & (dec["M"] != dec["priv"])
    s_hit = s_denom & (dec["pub"] == dec["M"]) & (dec["pub"] != dec["priv"])

    # Repair runs on the rows that did NOT conceal and saw some dissent.
    rep = (~concealed) & (dec["d"] > 0) & dec["was_correct"].notna()
    if repair_requires_majority:
        rep = rep & M_known
    wrong = rep & (~dec["was_correct"].astype("boolean").fillna(False))
    right = rep & (dec["was_correct"].astype("boolean").fillna(False))
    now_ok = dec["now_correct"].astype("boolean").fillna(False)

    return dict(
        n_contra=int(contradicted.sum()),
        n_conceal=int(concealed.sum()),
        n_gap=int((concealed & (dec["pub"] != dec["priv"])).sum()),
        n_strict_denom=int(s_denom.sum()),
        n_strict_hit=int(s_hit.sum()),
        n_conc_pairs=int(concealed.sum()),
        n_internalise=int(internalised.sum()),
        sum_d_wrong=float(dec.loc[wrong, "d"].sum()),
        flip_up=int((wrong & now_ok).sum()),
        n_wrong_rows=int(wrong.sum()),
        sum_d_right=float(dec.loc[right, "d"].sum()),
        flip_down=int((right & ~now_ok).sum()),
        n_right_rows=int(right.sum()),
        n_rows=int(len(dec)),
    )


# ------------------------------------------------------------------- the rates
def rates(cnt: dict, trigger: str = "fraction", legacy_round: bool = False) -> dict:
    """Counts -> rates. Pure arithmetic, so the bootstrap can call it per draw.

    ``legacy_round`` rounds the summed-dissent denominators to whole numbers,
    which is what the 0831 scripts did because a Wilson interval needs an
    integer n. This package intervals by cluster bootstrap instead, so it has
    no reason to round -- but rounding a denominator that can be as small as a
    dozen moves rho by several points, so the flag exists to reproduce the
    published numbers exactly rather than approximately.
    """
    if trigger not in TRIGGERS:
        raise ValueError(f"unknown trigger {trigger!r}; expected one of {TRIGGERS}")
    if legacy_round:
        cnt = dict(cnt, sum_d_wrong=float(round(cnt["sum_d_wrong"])),
                   sum_d_right=float(round(cnt["sum_d_right"])))

    c_legacy = _rate(cnt["n_conceal"], cnt["n_contra"])
    c_strict = _rate(cnt["n_strict_hit"], cnt["n_strict_denom"])
    real_gap = _rate(cnt["n_gap"], cnt["n_conceal"])
    a = _rate(cnt["n_internalise"], cnt["n_conc_pairs"])

    # Two readings of the same flips, one per trigger convention.
    up_frac = _rate(cnt["flip_up"], cnt["sum_d_wrong"])
    dn_frac = _rate(cnt["flip_down"], cnt["sum_d_right"])
    up_any = _rate(cnt["flip_up"], cnt["n_wrong_rows"])
    dn_any = _rate(cnt["flip_down"], cnt["n_right_rows"])

    def pair(up, dn):
        if not np.isfinite(up):
            return np.nan, np.nan
        d = 0.0 if not np.isfinite(dn) else dn
        return min(up + d, 1.0), up - d          # rho, gamma

    rho_frac, gamma_frac = pair(up_frac, dn_frac)
    rho_any, gamma_any = pair(up_any, dn_any)
    # r is the share of re-derivations that landed on the truth. Numerator and
    # denominator scale together under either convention, so it is trigger-free.
    tot = (up_frac + (dn_frac if np.isfinite(dn_frac) else 0.0)
           if np.isfinite(up_frac) else np.nan)
    r = up_frac / tot if (np.isfinite(tot) and tot > 0) else np.nan

    rho = rho_frac if trigger == "fraction" else rho_any
    gamma = gamma_frac if trigger == "fraction" else gamma_any
    return dict(
        c=c_legacy, c_strict=c_strict, real_gap_share=real_gap, a=a,
        rho=rho, r=r, gamma=gamma,
        rho_fraction=rho_frac, rho_any=rho_any,
        gamma_fraction=gamma_frac, gamma_any=gamma_any,
        c_star=c_star(gamma, a),
        c_star_fraction=c_star(gamma_frac, a), c_star_any=c_star(gamma_any, a),
        margin=c_legacy - c_star(gamma, a),
        **{k: cnt[k] for k in cnt},
    )


def c_star(gamma, a):
    """Concealment at which a wrong consensus becomes self-sustaining.

    Linearising the mean-field map about the fully falsified consensus gives
    ``c* = gamma / (gamma + a)`` with ``gamma = rho(2r - 1)``. Two boundary cases
    matter in practice and both fall out of the same expression:

    ``gamma <= 0`` (r <= 1/2)  deliberation is not truth-seeking, there is no
                               current out of the wrong consensus, and c* = 0:
                               any concealment at all is above critical.
    ``a = 0``                  nothing holds the falsified consensus together
                               and c* = 1: the team recovers at every c.
    """
    if not np.isfinite(gamma) or not np.isfinite(a):
        return np.nan
    g = max(float(gamma), 0.0)
    if g + a <= 0:
        return np.nan if a != 0 else 1.0
    return g / (g + a)


# --------------------------------------------------------------- uncertainty
def cluster_bootstrap(dec: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                      trigger: str = "fraction",
                      repair_requires_majority: bool = False) -> pd.DataFrame:
    """Resample TASKS, not agent-rounds, and recompute every rate per draw.

    Agent-rounds are not independent draws: the same item is answered by N
    agents over T rounds, and items differ in difficulty, so the effective
    sample size is closer to the number of tasks. Resampling tasks also gives
    the derived quantities -- gamma, c*, and above all the margin c - c* -- a
    *joint* interval, which is the only kind that can answer "is this team above
    the boundary" without pretending c and c* are independent.
    """
    if dec.empty:
        return pd.DataFrame()
    tasks = sorted(dec["task_id"].unique())
    per_task = {t: counts(g, repair_requires_majority=repair_requires_majority)
                for t, g in dec.groupby("task_id")}
    keys = list(next(iter(per_task.values())))
    mat = np.array([[per_task[t][k] for k in keys] for t in tasks], dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tasks), size=(n_boot, len(tasks)))
    sums = mat[idx].sum(axis=1)                      # (n_boot, n_keys)
    draws = [rates(dict(zip(keys, row)), trigger=trigger) for row in sums]
    return pd.DataFrame(draws)


def interval(draws: pd.DataFrame, name: str, alpha: float = 0.05):
    if draws.empty or name not in draws:
        return (np.nan, np.nan)
    v = draws[name].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))


# ------------------------------------------------------------ conditional cuts
def by_dissent(dec: pd.DataFrame, bins=(1, 2, 3, 4, 100)) -> pd.DataFrame:
    """Concealment conditioned on how many visible neighbours disagreed.

    Two reasons this is not optional. The theory's engagement rate is a function
    of exactly this quantity, and concealment is known to saturate at three or
    four dissenters (the Asch group-size effect), so a single pooled ``c`` from a
    degree-9 graph is not comparable to one from a degree-3 graph. Anything
    comparing runs with different topologies has to compare these rows.
    """
    if dec.empty:
        return pd.DataFrame()
    d = dec[dec["M"].notna() & (dec["M"] != dec["c_prev"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["bin"] = pd.cut(d["n_dissent"], bins=list(bins), right=False)
    out = []
    for b, g in d.groupby("bin", observed=True):
        n = len(g)
        out.append(dict(n_dissenters=str(b), n=n,
                        c=_rate(int((g["pub"] == g["M"]).sum()), n)))
    return pd.DataFrame(out)


def by_round(dec: pd.DataFrame) -> pd.DataFrame:
    """Concealment round by round -- does the team harden as it talks?"""
    if dec.empty:
        return pd.DataFrame()
    d = dec[dec["M"].notna() & (dec["M"] != dec["c_prev"])]
    if d.empty:
        return pd.DataFrame()
    return (d.assign(hit=(d["pub"] == d["M"]).astype(int))
             .groupby("round")
             .agg(n=("hit", "size"), c=("hit", "mean"))
             .reset_index())


def gap_series(t, condition: str) -> pd.DataFrame:
    """Mean private-minus-public accuracy per round: the susceptibility.

    The private/public gap is the pluralistic-ignorance signature and it grows
    as the team approaches the boundary, so its trajectory is a cheap proximity
    diagnostic that needs neither ground-truth-free modelling nor a fit.
    """
    truth = t.truth()
    rows = [r for r in t.rows if r["branch"] in (condition, "r0")]
    out = []
    for rnd in sorted({r["round"] for r in rows}):
        priv, pub = [], []
        for r in rows:
            if r["round"] != rnd:
                continue
            tr = truth.get(r["task_id"])
            if tr is None:
                continue
            if r.get("private"):
                priv.append(r["private"] == tr)
            if r.get("public"):
                pub.append(r["public"] == tr)
        if priv and pub:
            out.append(dict(round=int(rnd), private_acc=float(np.mean(priv)),
                            public_acc=float(np.mean(pub)),
                            gap=float(np.mean(priv)) - float(np.mean(pub))))
    return pd.DataFrame(out)


def trigger_check(dec: pd.DataFrame) -> dict:
    """Is visible dissent alone what gates re-derivation, as the model says?

    The model's engagement rate is ``rho * d`` and makes no reference to whether
    a majority has formed. If that holds, ``rho`` estimated from the rows whose
    panel was tied must equal ``rho`` estimated from the rows whose panel was
    decided. It is a strong assumption and it is testable on any transcript,
    because ties happen: on a five-agent team every agent sees four neighbours
    and 2-2 splits are common.

    ``rho`` is recomputed within each group rather than a raw flip rate being
    compared, because the two groups differ in composition -- a tied panel never
    contradicts anyone, so nobody in it conceals, while the decided group has its
    contradicted agents thinned out by concealment. Splitting the flips into the
    upward and downward channels and recombining them the way ``rates`` does
    controls for that; on transcripts generated by the model the ratio comes back
    at 1.00 +- 0.05, which is what makes a departure on real data a finding
    rather than an artefact.

    Measured across the (run, condition) cells in this repo the ratio spans an
    order of magnitude, and its direction is a property of the model rather than
    noise: gpt-4o-mini re-derives *less* under a tie than the fraction rule
    predicts, as if it needs a majority against it before it rechecks, while the
    reasoning models and the 25-agent panels re-derive *more*, as if an evenly
    split room is itself the prompt to think again.

    So ``d`` is not a sufficient statistic for engagement, ``rho`` and ``gamma``
    carry a specification-dependent component, and a run whose ratio is far from
    1 should not have its ``rho`` compared against one whose ratio is near it.
    Reporting the number is the only honest option; the alternative is to pick a
    trigger and not say that a choice was made.
    """
    empty = dict(n_tied=0, n_decided=0, rho_tied=np.nan, rho_decided=np.nan,
                 mean_d_tied=np.nan, mean_d_decided=np.nan, ratio=np.nan)
    if dec.empty:
        return empty
    out = {}
    for tag, sel in (("tied", dec["M"].isna()), ("decided", dec["M"].notna())):
        sub = dec[sel]
        c = counts(sub)
        r_ = rates(c)
        out[f"n_{tag}"] = c["n_wrong_rows"] + c["n_right_rows"]
        out[f"mean_d_{tag}"] = (float(sub.loc[sub["d"] > 0, "d"].mean())
                                if (sub["d"] > 0).any() else np.nan)
        out[f"rho_{tag}"] = r_["rho"]
    a, b = out["rho_tied"], out["rho_decided"]
    out["ratio"] = (a / b if (np.isfinite(a) and np.isfinite(b) and b > 0)
                    else np.nan)
    return out


def base_noise(t, condition: str, baseline_branch: str = "r0") -> dict:
    """How much happens that the model has no rule for. Two channels, same rounds.

    The CBRM moves a belief only through repair or internalisation, and both need
    something visible to react to. An agent whose visible neighbours all express
    what it already believed has nothing to respond to, so anything that moves
    there is off-model. Those *quiet slots* are the natural place to measure the
    base noise rate, and this estimator measures it there.

    Two things can be off-model in a quiet slot, and they are separate channels:

    ``eps_belief``  the belief changed. Reported as twice the flip rate, because
                    a redrawn belief lands on the other label half the time, so
                    the redraw *rate* is twice the flip rate.
    ``eps_express`` the agent said something other than what it believed, with no
                    majority to concede to. The model assigns this probability
                    zero outright, and it is what makes a latent-variable fit
                    collapse without a noise term (see ``cbrm.latent``).

    ``strict`` additionally drops slots where a neighbour disclosed a finding on
    the round just past: a quiet neighbourhood is not a silent one, and a belief
    that moved because someone put a decisive fact on the board moved on
    information rather than on noise. The strict figure is the estimate and the
    loose one an upper bound; the gap between them is the pooling channel.

    Measured against the HMM's fitted noise rate on seven runs the two agree at
    Spearman 0.64 -- two estimators of different channels, built on different
    machinery, ordering the models the same way.
    """
    truth = t.truth()
    by = {(r["task_id"], r["agent"], r["round"], r["branch"]): r for r in t.rows}
    shared_at = defaultdict(set)
    for r in t.rows:
        if r.get("share_text") or r.get("share"):
            shared_at[(r["task_id"], r["branch"], r["round"])].add(r["agent"])

    n_quiet = n_quiet_flip = n_quiet_said = 0
    n_strict = n_strict_flip = n_strict_said = 0
    for r in t.rows:
        if r["branch"] != condition or r["round"] == 0:
            continue
        prev = (by.get((r["task_id"], r["agent"], r["round"] - 1, condition))
                or by.get((r["task_id"], r["agent"], r["round"] - 1,
                           baseline_branch)))
        if prev is None or prev.get("private") is None or r.get("private") is None:
            continue
        nb = [a for a in (r.get("nb_public") or []) if a]
        if not nb:
            continue                       # nothing visible: says nothing here
        c_prev = prev["private"]
        M = majority(nb)
        concealed = M is not None and M != c_prev and r.get("public") == M
        if sum(a != c_prev for a in nb) != 0 or concealed:
            continue                       # not a quiet slot
        flipped = r["private"] != c_prev
        said_other = r.get("public") is not None and r["public"] != c_prev
        n_quiet += 1
        n_quiet_flip += int(flipped)
        n_quiet_said += int(said_other)
        fresh = shared_at.get((r["task_id"], r["branch"], r["round"] - 1), set())
        if not (set(r.get("neighbours") or []) & fresh):
            n_strict += 1
            n_strict_flip += int(flipped)
            n_strict_said += int(said_other)

    def two(k, n):
        return 2.0 * k / n if n else np.nan

    return dict(
        n_quiet=n_quiet, n_strict=n_strict,
        eps_belief_loose=two(n_quiet_flip, n_quiet),
        eps_belief=two(n_strict_flip, n_strict),
        eps_express_loose=_rate(n_quiet_said, n_quiet),
        eps_express=_rate(n_strict_said, n_strict),
        has_truth=bool(truth))
