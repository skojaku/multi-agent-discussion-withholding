"""Posteriors instead of point estimates, with the task clustering kept.

Two things the frequency estimator cannot do, and one trap in doing them.

**The r = 1 boundary.** When no correct agent was ever talked out of its answer
-- 52 of 144 real cells -- the downward flip count is zero, the maximum-likelihood
r sits exactly on 1, gamma collapses onto rho, and c* is an artefact printed as a
number. Worse, the cluster bootstrap returns [1.000, 1.000]: every resample has
the same empty numerator, so the interval has zero width and is not an interval.
A posterior puts mass below 1 and reports what the data actually support. The
(u, v) parametrisation of ``_uv_posterior`` removes the boundary outright: an
empty downward count is just a small-count posterior on v.

**P(trapped).** The margin c - c* is a nonlinear function of three rates. What a
caller wants is not a point and an interval but the probability the team is above
the boundary. Push joint draws through the formula and it falls out.

**The trap: agent-rounds are not independent draws.** The same item is answered
by N agents over T rounds and items differ in difficulty, so a plain
Beta-Binomial posterior is too narrow -- measured against the cluster bootstrap
on a 25-agent panel it is 3.4x too tight. That would be a step backwards from
what this package already does. So c and a get a hierarchical Beta-Binomial with
a per-task rate and a concentration parameter, marginalised exactly; the pooled
limit is the concentration going to infinity, and the data decide.

Everything is a grid. The models are two- and three-dimensional, so exact
evaluation is faster than any sampler, has no convergence question, and adds no
dependency beyond scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, logsumexp

from . import estimators as est

# Concentration grid: 0.5 is wildly overdispersed (each task its own rate),
# 1e4 is effectively pooled. Uniform on log phi is the usual weak default.
PHI_GRID = np.exp(np.linspace(np.log(0.5), np.log(1e4), 60))
MU_GRID = np.linspace(1e-3, 1 - 1e-3, 401)


def _betabinom_posterior(per_task, prior=(1.0, 1.0)):
    """Posterior over the population rate mu, tasks clustered.

    ``per_task`` is a list of (successes, trials). The per-task rate is
    integrated out analytically, so the only numerical work is a 2-D grid over
    (mu, phi) and a marginalisation of phi.
    """
    k = np.array([p[0] for p in per_task], dtype=float)
    n = np.array([p[1] for p in per_task], dtype=float)
    keep = n > 0
    k, n = k[keep], n[keep]
    if n.sum() == 0:
        return None
    mu = MU_GRID[:, None, None]
    phi = PHI_GRID[None, :, None]
    a1, b1 = mu * phi, (1.0 - mu) * phi
    ll = (betaln(k[None, None, :] + a1, n[None, None, :] - k[None, None, :] + b1)
          - betaln(a1, b1)).sum(axis=2)
    # Beta(prior) on mu, uniform on log phi
    ll += ((prior[0] - 1) * np.log(MU_GRID) +
           (prior[1] - 1) * np.log1p(-MU_GRID))[:, None]
    post = np.exp(ll - ll.max())
    post /= post.sum()
    return MU_GRID, post.sum(axis=1) / post.sum()


def _sample(grid, weights, n, rng):
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        return np.full(n, np.nan)
    w = w / w.sum()
    idx = rng.choice(len(grid), size=n, p=w)
    step = grid[1] - grid[0] if len(grid) > 1 else 0.0
    return np.clip(grid[idx] + rng.uniform(-0.5, 0.5, n) * step, 0.0, 1.0)


def _repair_rows(dec):
    M_known = dec["M"].notna()
    concealed = M_known & (dec["M"] != dec["c_prev"]) & (dec["pub"] == dec["M"])
    rep = (~concealed) & (dec["d"] > 0) & dec["was_correct"].notna()
    d = dec[rep]
    if d.empty:
        return None
    was = d["was_correct"].astype(bool).to_numpy()
    now = d["now_correct"].astype(bool).to_numpy()
    return d["d"].to_numpy(), was, (now != was), d["task_id"].to_numpy()


def _uv_posterior(dec, grid=401, temper=1.0, constrain=True):
    """Exact 2-D posterior over the two directed flows, u = rho*r and v = rho*(1-r).

    The natural parameters here are not rho and r but the two flows themselves.
    An agent that enters wrong crosses to the truth with probability ``u * d``;
    one that enters right crosses away with ``v * d``. Those are *disjoint* row
    sets, so the likelihood factorises, u and v are independent a posteriori,
    and gamma = u - v is the convolution of the two. Three things follow.

    **gamma keeps its sign.** In (rho, r) the net flow is rho(2r - 1), which is
    negative exactly when r < 1/2 -- a region the grid has to reach through a
    product. Here it is a difference, so a team that loses more than it gains
    reports a negative gamma with an interval, instead of a point mass at the
    floor.

    **gamma is not widened by the ridge.** rho and r are only weakly identified
    apart: many (rho, r) give the same net flow, and marginalising along that
    ridge leaks into gamma. u and v are each identified by their own rows, so
    nothing leaks. Measured against the (rho, r) grid at the same temperature,
    the gamma interval is 30-50% narrower with the same centre.

    **It is the counting estimator's own posterior.** The moment estimator of
    the main text is S_W/D_W - S_R/D_R, which is exactly u-hat - v-hat. The two
    are now one construction rather than two.

    ``constrain`` keeps u + v = rho inside [0, 1]. It binds only where the
    engagement the data asks for exceeds one re-derivation per round, which is
    itself a sign that the proportional trigger cannot carry the observed flow.
    """
    got = _repair_rows(dec)
    if got is None:
        return None
    dd, was, flipped, _ = got
    g = np.linspace(1e-4, 1 - 1e-4, grid)

    def loglik(sel):
        if not sel.any():
            return np.zeros(grid)
        p = np.clip(g[:, None] * dd[sel][None, :], 1e-12, 1 - 1e-12)
        f = flipped[sel][None, :]
        return np.sum(np.where(f, np.log(p), np.log1p(-p)), axis=1)

    ll = (loglik(was == False)[:, None] + loglik(was == True)[None, :])  # noqa: E712
    ll = ll * float(temper)
    if constrain:
        U, V = np.meshgrid(g, g, indexing="ij")
        ll = np.where(U + V <= 1.0, ll, -np.inf)
    post = np.exp(ll - np.nanmax(ll))
    tot = post.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None
    return g, post / tot


def _design_effect(dec, key, naive_sd, n_boot=600, seed=0):
    """How much wider the cluster bootstrap is than the independent-draw sd.

    Used only to temper the (rho, r) likelihood; c and a get the clustering
    modelled properly instead.
    """
    if not np.isfinite(naive_sd) or naive_sd <= 0:
        return 1.0
    draws = est.cluster_bootstrap(dec, n_boot=n_boot, seed=seed)
    if draws.empty or key not in draws:
        return 1.0
    v = draws[key].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 20 or v.std() <= 0:
        return 1.0
    return float(max(1.0, (v.std() / naive_sd) ** 2))


def posterior(dec, n_draw: int = 40000, seed: int = 0, prior=(1.0, 1.0),
              hierarchical: bool = True, temper_rho_r: bool = True) -> dict:
    """Joint posterior draws for c, a, rho, r and everything derived from them."""
    rng = np.random.default_rng(seed)
    out = {"counts": est.counts(dec)}
    if dec.empty:
        return out

    M_known = dec["M"].notna()
    contradicted = M_known & (dec["M"] != dec["c_prev"])
    concealed = contradicted & (dec["pub"] == dec["M"])
    internalised = concealed & (dec["priv"] == dec["pub"])
    tasks = dec["task_id"].to_numpy()

    def draw_rate(num_mask, den_mask):
        if hierarchical:
            per_task = []
            for tk in np.unique(tasks):
                sel = tasks == tk
                per_task.append((int((num_mask & sel).sum()),
                                 int((den_mask & sel).sum())))
            got = _betabinom_posterior(per_task, prior=prior)
            if got is None:
                return np.full(n_draw, np.nan)
            return _sample(got[0], got[1], n_draw, rng)
        k, n = int(num_mask.sum()), int(den_mask.sum())
        return rng.beta(prior[0] + k, prior[1] + n - k, n_draw)

    c = draw_rate(concealed.to_numpy(), contradicted.to_numpy())
    a = draw_rate(internalised.to_numpy(), concealed.to_numpy())

    def _draw_uv(temper):
        got = _uv_posterior(dec, temper=temper)
        if got is None:
            return None
        g, p2 = got
        flat = p2.ravel()
        idx = rng.choice(flat.size, size=n_draw, p=flat / flat.sum())
        step = g[1] - g[0]
        u = np.clip(g[idx // len(g)] + rng.uniform(-0.5, 0.5, n_draw) * step, 0, 1)
        v = np.clip(g[idx % len(g)] + rng.uniform(-0.5, 0.5, n_draw) * step, 0, 1)
        return u, v

    # The temperature is calibrated on gamma, because gamma is what is reported.
    temper = 1.0
    if temper_rho_r:
        first = _draw_uv(1.0)
        if first is not None:
            sd = float(np.std(first[0] - first[1]))
            temper = 1.0 / _design_effect(dec, "gamma", sd, seed=seed)

    got = _draw_uv(temper)
    if got is None:
        rho = np.full(n_draw, np.nan)
        r = np.full(n_draw, np.nan)
        gamma = np.full(n_draw, np.nan)
    else:
        u, v = got
        gamma = u - v
        rho = np.clip(u + v, 0, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(u + v > 0, u / (u + v), np.nan)
    # gamma <= 0 means no net correction at all: every withholding rate is above
    # the threshold, so c* is 0 rather than undefined.
    with np.errstate(invalid="ignore", divide="ignore"):
        c_star = np.where(gamma <= 0, 0.0,
                          np.where(gamma + a > 0, gamma / (gamma + a), np.nan))
    out.update(c=c, a=a, rho=rho, r=r, gamma=gamma, c_star=c_star,
               margin=c - c_star, temper=temper)
    return out


def summary(post: dict, alpha: float = 0.05) -> dict:
    """Posterior mean, credible interval, and P(trapped), per quantity."""
    out = {}
    for k in ("c", "a", "rho", "r", "gamma", "c_star", "margin"):
        v = post.get(k)
        if v is None:
            continue
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[k] = (np.nan, np.nan, np.nan)
            continue
        out[k] = (float(v.mean()), float(np.quantile(v, alpha / 2)),
                  float(np.quantile(v, 1 - alpha / 2)))
    m = np.asarray(post.get("margin", []), dtype=float)
    m = m[np.isfinite(m)]
    out["p_trapped"] = float((m > 0).mean()) if m.size else np.nan
    return out
