#!/usr/bin/env python3
"""Split the measured withholding rate into a model part and an instruction part.

    logit c_cell = mu + alpha[model] + kappa[condition] + e_cell

One cell is one (run, condition) row of ``data/sweep.csv``. The observation is
``y = logit(c_post)`` with the hierarchical posterior interval of ``c_post``
giving the measurement standard error on the logit scale, so the task clustering
that the posterior already models is carried into this fit. ``e_cell`` absorbs
what is left (replicate runs, task-design variants, condition-specific
deviations). ``alpha`` and ``e`` get mean-zero Gaussian priors with their own
variances (inverse-gamma hyperpriors); ``kappa`` is a fixed effect with C0 as the
reference. Everything is conditionally Gaussian / inverse-gamma, so the posterior
is sampled by Gibbs with no tuning.

**Two fits, one sampler.** The default is confined to the hidden profile
benchmark, where a benchmark main effect is a constant absorbed into ``mu`` and a
model x benchmark interaction is a constant per model absorbed into ``alpha``.
``--crossed`` fits every benchmark at once and puts both back,

    logit c_cell = mu + alpha[model] + beta[benchmark] + kappa[condition]
                   + delta[model x benchmark] + e_cell

which is the fit the paper's Figure 3(c) draws: the claim it makes is that the
benchmark moves the withholding rate at least as far as the model does, and that
claim needs the benchmark on the same logit scale as the model and the
instruction, not only in the per-setting rates. The crossed fit was dropped on
9/12, when nothing in the paper quoted it, and is back on 9/14 because 4.2 does.
Its outputs carry the ``_crossed`` tag so the two never overwrite each other.

What crosses benchmarks is therefore reported, not fitted: ``decomposition_fitted.csv``
holds the per-setting posterior rate of each (model, benchmark) at C0, pooled by
inverse-variance weighting on the logit scale when a pair has more than one run.
That file keeps the column names the fitted version had, so the paper figure that
reads it is unchanged; what changed is where the numbers come from.

**One stage (9/24).** ``--one-stage`` fits the crossed model to the per-task
counts of ``task_counts.py`` instead of to logit(c_post) with a known error: the
binomial likelihood, a task-level term with one variance per benchmark, and the
same effects and priors, sampled by Gibbs with Polya-Gamma augmentation. It is
the fit the paper's Figure 3(d) draws; F2 and F3(a-c) keep the per-setting rates.

Outputs (under the experiment directory, ``<tag>`` empty or ``_crossed``):
    data/decomposition_cells<tag>.csv     the cells that entered the fit
    data/decomposition_effects<tag>.csv   posterior summaries of every effect
    data/decomposition_shares<tag>.csv    variance shares of the fitted terms
    data/decomposition_fitted<tag>.csv    measured C0 rate per (model, benchmark)
    figures/fig_decomposition<tag>.png    benchmark x model, effects and shares

Cross-check: a binomial mixed GLM on the raw counts (statsmodels, variational
Bayes) with the same model random effect; printed, not saved as a figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

# Filled from the workflow config by `configure()` (see main). Nothing about
# which models, benchmarks or instruction arms exist is written in this file.
FAMILY: dict[str, str] = {}        # run-directory prefix -> benchmark
INTERVENTIONS = ("devils_advocate", "disclose_first", "disclose_forced", "evidence_only")
EXCLUDED_RUNS: tuple[str, ...] = ()
# Reasoning ablations carry the same model id as the main arm and would be read
# as extra cells of that model. They are recognised by the suffix of the run
# name and excluded, unless --reasoning admits the reasoning-off arms
# (OFF_SUFFIX), which are then pooled into their model.
ABLATION_SUFFIX: tuple[str, ...] = ()
OFF_SUFFIX = "_nothink"
MAIN_MODELS: tuple[str, ...] = ()  # models the fit is over, by transcript id
CONDITIONS: tuple[str, ...] = ()   # raw instruction keys the fit is over
REFERENCE = "C0"                   # the instruction every kappa is relative to
FIT_FAMILY = ""                    # the one benchmark the single fit is on
MIN_CONTRA = 30
# Which elicitation the main text reports (the separate-call probe). Mixing
# elicitations moves a-hat by about 0.5 and c-hat by about 0.1, so only cells
# from runs made the same way enter.
MAIN_PROBE = "separate"
PROBE_KEEP = {"joint": ("joint", "unknown"), "separate": ("separate",)}


def configure(cfg: dict) -> None:
    """Set the module constants from the workflow config (config.yaml)."""
    global FAMILY, ABLATION_SUFFIX, OFF_SUFFIX, MAIN_MODELS, CONDITIONS, FIT_FAMILY
    global MIN_CONTRA, MAIN_PROBE, REFERENCE
    FAMILY = {p: key for key, b in cfg["benchmarks"].items()
              for p in b["decomposition_prefixes"]}
    ABLATION_SUFFIX = tuple(cfg["ablation_suffix"])
    OFF_SUFFIX = cfg.get("reasoning_off_suffix", OFF_SUFFIX)
    MAIN_MODELS = tuple(m["name"] for m in cfg["models"] if m.get("decomposition"))
    CONDITIONS = tuple(cfg["arms"])
    REFERENCE = cfg.get("reference_arm", REFERENCE)
    FIT_FAMILY = cfg["decomposition_fit_family"]
    MIN_CONTRA = int(cfg.get("decomposition_min_contra", 30))
    MAIN_PROBE = cfg["probe"]


def load_cells(path: str, models: tuple[str, ...] | None,
               family: str | None = None, reasoning: bool = False) -> pd.DataFrame:
    """Cells of the standard protocol, on the logit scale with their own error.

    ``family`` restricts to one benchmark, which is what the fit wants; left at
    ``None`` the frame spans every benchmark, which is what the cross-benchmark
    table wants. ``reasoning`` also admits the ``_nothink`` arms, pooled into
    their base model: its alpha is then the model over both reasoning
    settings (9/24, F3(d)). Split as ``<model>@off`` levels the two arms sat
    on top of each other for every model, so the paper draws the pool.
    """
    df = pd.read_csv(path)
    df["model_short"] = df["model"].fillna("").str.split("/").str[-1]
    df["family_task"] = df["run"].str.split("/").str[0].map(FAMILY)
    leaf = df["run"].str.rsplit("/", n=1).str[-1]
    ablation = leaf.str.endswith(ABLATION_SUFFIX)
    if reasoning:
        ablation &= ~leaf.str.endswith(OFF_SUFFIX)
    keep = (
        (df["topology"] == "complete")
        & (df["T"] == 3)
        & df["n_agents"].isin([4, 5])
        & df["probe"].fillna("unknown").isin(PROBE_KEEP[MAIN_PROBE])
        & df["condition"].isin(CONDITIONS)
        & df["family_task"].notna()
        & df["c_post"].notna()
        & (df["n_contra"] >= MIN_CONTRA)
        & ~df["run"].isin(EXCLUDED_RUNS)
        & ~ablation
    )
    for w in INTERVENTIONS:
        keep &= ~df["run"].str.contains(w)
    if models is not None:
        keep &= df["model_short"].isin(models)
    if family is not None:
        keep &= df["family_task"] == family
    d = df.loc[keep].copy()
    lo = d["c_post_lo"].clip(0.01, 0.99).to_numpy()
    hi = d["c_post_hi"].clip(0.01, 0.99).to_numpy()
    d["y"] = logit(d["c_post"].clip(0.01, 0.99).to_numpy())
    d["se"] = np.maximum((logit(hi) - logit(lo)) / (2 * 1.96), 0.05)
    return d.reset_index(drop=True)


def measured_by_benchmark(path: str, models: tuple[str, ...] | None) -> pd.DataFrame:
    """C0 withholding rate per (model, benchmark), straight from the posteriors.

    No decomposition is involved. Where a pair ran more than once -- the repeat
    runs -- the logit estimates are combined by inverse-variance weighting, which
    is the same weighting the fit gives them, and the interval is the weighted
    mean plus or minus 1.96 standard errors carried back through the logistic.
    Column names match the file the fitted version wrote, so the figure that
    reads it does not change.
    """
    d = load_cells(path, models)
    d = d[d["condition"] == REFERENCE]
    rows = []
    for (mdl, fam), g in d.groupby(["model_short", "family_task"]):
        w = 1.0 / g["se"].to_numpy() ** 2
        eta = float(np.average(g["y"].to_numpy(), weights=w))
        se = float(np.sqrt(1.0 / w.sum()))
        rows.append(dict(model=mdl, family=fam, c_fit_mean=float(expit(eta)),
                         c_fit_lo=float(expit(eta - 1.96 * se)),
                         c_fit_hi=float(expit(eta + 1.96 * se)),
                         c_raw_C0_mean=float(g["c_post"].mean()),
                         eta=eta, eta_se=se, n_cells=int(len(g))))
    return pd.DataFrame(rows).sort_values(["family", "model"]).reset_index(drop=True)


# The random-effect blocks each fit puts in: the one-benchmark fit has only the
# model, the crossed fit adds the benchmark and the model x benchmark cell. The
# sampler below is written over this list, so a block costs one line here.
BLOCKS = {"single": (("model", "model_short"),),
          "crossed": (("model", "model_short"), ("family", "family_task"),
                      ("mf", "mf"))}
TERM_NAME = {"model": "alpha", "family": "beta", "mf": "delta"}
SHARE_NAME = {"model": "model", "family": "task", "mf": "interaction",
              "kappa": "instruction", "e": "residual"}


# 9/23: 30000 sweeps from one seed was not enough for the crossed fit once
# C1-C3 left it. The intercept and the four benchmark levels trade off along a
# ridge the sampler crawls along, and the benchmark share read 35% from seed 0
# against 24-26% from seeds 1-3. Four chains of 210000 sweeps agree to within
# about two points on every share, and the fit pools them.
N_ITER, BURN, THIN, CHAINS = 210000, 10000, 50, 4


def gibbs(d: pd.DataFrame, n_iter=N_ITER, burn=BURN, thin=THIN, a0=1.0, b0=0.5, seed=0,
          blocks=BLOCKS["single"]):
    rng = np.random.default_rng(seed)
    y = d["y"].to_numpy()
    w = 1.0 / d["se"].to_numpy() ** 2
    n = len(y)
    groups = {}
    for name, col in blocks:
        levels = sorted(d[col].unique())
        groups[name] = (d[col].map({l: i for i, l in enumerate(levels)}).to_numpy(), levels)
    conds = [c for c in CONDITIONS if c != REFERENCE and (d["condition"] == c).any()]
    X = np.stack([(d["condition"] == c).to_numpy(float) for c in conds], axis=1) if conds else np.zeros((n, 0))
    u = {g: np.zeros(len(lv)) for g, (_, lv) in groups.items()}
    sig2 = {g: 1.0 for g in groups}
    kappa = np.zeros(X.shape[1])
    e = np.zeros(n)
    sig2_e = 0.5
    mu = float(np.average(y, weights=w))
    out = []

    def contrib(exclude=None):
        tot = np.zeros(n)
        for g, (idx, _) in groups.items():
            if g != exclude:
                tot += u[g][idx]
        return tot

    for it in range(n_iter):
        # intercept (flat prior)
        r = y - e - contrib() - X @ kappa
        prec = w.sum()
        mu = rng.normal((w * r).sum() / prec, prec ** -0.5)
        # fixed condition effects, prior N(0, 3^2)
        if X.shape[1]:
            r = y - mu - e - contrib()
            A = (X * w[:, None]).T @ X + np.eye(X.shape[1]) / 9.0
            b = (X * w[:, None]).T @ r
            L = np.linalg.cholesky(A)
            mean = np.linalg.solve(A, b)
            kappa = mean + np.linalg.solve(L.T, rng.standard_normal(X.shape[1]))
        # random effects
        for g, (idx, lv) in groups.items():
            r = y - mu - e - X @ kappa - contrib(exclude=g)
            sw = np.bincount(idx, weights=w, minlength=len(lv))
            swr = np.bincount(idx, weights=w * r, minlength=len(lv))
            prec = sw + 1.0 / sig2[g]
            u[g] = rng.normal(swr / prec, prec ** -0.5)
            sig2[g] = 1.0 / rng.gamma(a0 + len(lv) / 2, 1.0 / (b0 + (u[g] ** 2).sum() / 2))
        # residual
        r = y - mu - X @ kappa - contrib()
        prec = w + 1.0 / sig2_e
        e = rng.normal(w * r / prec, prec ** -0.5)
        sig2_e = 1.0 / rng.gamma(a0 + n / 2, 1.0 / (b0 + (e ** 2).sum() / 2))
        if it >= burn and (it - burn) % thin == 0:
            out.append(dict(mu=mu, kappa=kappa.copy(), sig2_e=sig2_e,
                            **{f"u_{g}": u[g].copy() for g in groups},
                            **{f"sig2_{g}": sig2[g] for g in groups}))
    return out, groups, conds


def load_task_rows(d: pd.DataFrame, path: str) -> pd.DataFrame:
    """The per-task counts of the settings in ``d``, from ``task_counts.py``.

    A task with no contradicted row carries no likelihood and is dropped.
    """
    t = pd.read_csv(path)
    keep = ["run", "condition", "model_short", "family_task", "mf"]
    rows = t.merge(d[keep], on=["run", "condition"], how="inner")
    rows = rows[rows["n_contra"] > 0].reset_index(drop=True)
    rows["cell"] = rows["run"] + " | " + rows["condition"]
    got = rows.groupby("cell")[["n_contra", "n_conceal"]].sum()
    ref = (d.assign(cell=d["run"] + " | " + d["condition"])
           .set_index("cell")[["n_contra", "n_conceal"]].loc[got.index])
    if len(got) != len(d) or (got != ref).any().any():
        raise SystemExit(f"{path} does not match the sweep's counts on the fitted settings")
    return rows


def gibbs_one_stage(rows: pd.DataFrame, n_iter=N_ITER, burn=BURN, thin=THIN,
                    a0=1.0, b0=0.5, seed=0, blocks=BLOCKS["crossed"]):
    """The crossed decomposition fitted to the per-task counts themselves.

        k_l ~ Binomial(n_l, theta_l),
        logit theta_l = mu + alpha[m] + beta[b] + kappa[k] + delta[m, b]
                        + e[setting] + u_l,          u_l ~ N(0, tau_b^2)

    ``u_l`` is the task-to-task spread inside a setting, the part the Beta over
    tasks carries in the per-setting estimate, with one variance per benchmark.
    Polya-Gamma augmentation (Polson, Scott and Windle 2013) makes the binomial
    likelihood conditionally Gaussian: given omega_l ~ PG(n_l, psi_l), the
    pseudo-observation (k_l - n_l / 2) / omega_l is normal around psi_l with
    precision omega_l, so every other block is the weighted Gaussian update the
    two-stage sampler already made, with omega in place of 1 / s^2. The setting
    residual e is a random-effect block here, since a setting holds many rows.
    """
    from polyagamma import random_polyagamma
    rng = np.random.default_rng(seed)
    k = rows["n_conceal"].to_numpy(float)
    nn = rows["n_contra"].to_numpy(float)
    kappa_obs = k - nn / 2
    n = len(rows)
    groups = {}
    for name, col in tuple(blocks) + (("e", "cell"),):
        levels = sorted(rows[col].unique())
        groups[name] = (rows[col].map({l: i for i, l in enumerate(levels)}).to_numpy(), levels)
    fams = sorted(rows["family_task"].unique())
    fidx = rows["family_task"].map({f: i for i, f in enumerate(fams)}).to_numpy()
    conds = [c for c in CONDITIONS if c != REFERENCE and (rows["condition"] == c).any()]
    X = np.stack([(rows["condition"] == c).to_numpy(float) for c in conds], axis=1)
    u = {g: np.zeros(len(lv)) for g, (_, lv) in groups.items()}
    sig2 = {g: 0.5 for g in groups}
    kappa = np.zeros(X.shape[1])
    tl = np.zeros(n)                       # task-level u_l
    tau2 = np.full(len(fams), 0.5)
    p0 = np.clip((k.sum() + 0.5) / (nn.sum() + 1), 0.01, 0.99)
    mu = float(logit(p0))
    out = []

    def contrib(exclude=None):
        tot = np.zeros(n)
        for g, (idx, _) in groups.items():
            if g != exclude:
                tot += u[g][idx]
        return tot

    for it in range(n_iter):
        psi = mu + X @ kappa + contrib() + tl
        w = random_polyagamma(nn, psi, random_state=rng)
        y = kappa_obs / w
        # intercept (flat prior)
        r = y - X @ kappa - contrib() - tl
        prec = w.sum()
        mu = rng.normal((w * r).sum() / prec, prec ** -0.5)
        # fixed condition effects, prior N(0, 3^2)
        r = y - mu - contrib() - tl
        A = (X * w[:, None]).T @ X + np.eye(X.shape[1]) / 9.0
        b = (X * w[:, None]).T @ r
        L = np.linalg.cholesky(A)
        kappa = np.linalg.solve(A, b) + np.linalg.solve(L.T, rng.standard_normal(X.shape[1]))
        # random effects, the setting residual among them
        for g, (idx, lv) in groups.items():
            r = y - mu - X @ kappa - contrib(exclude=g) - tl
            sw = np.bincount(idx, weights=w, minlength=len(lv))
            swr = np.bincount(idx, weights=w * r, minlength=len(lv))
            prec = sw + 1.0 / sig2[g]
            u[g] = rng.normal(swr / prec, prec ** -0.5)
            sig2[g] = 1.0 / rng.gamma(a0 + len(lv) / 2, 1.0 / (b0 + (u[g] ** 2).sum() / 2))
        # task-level spread, one variance per benchmark
        r = y - mu - X @ kappa - contrib()
        prec = w + 1.0 / tau2[fidx]
        tl = rng.normal(w * r / prec, prec ** -0.5)
        cnt = np.bincount(fidx, minlength=len(fams))
        ss = np.bincount(fidx, weights=tl ** 2, minlength=len(fams))
        tau2 = 1.0 / rng.gamma(a0 + cnt / 2, 1.0 / (b0 + ss / 2))
        if it >= burn and (it - burn) % thin == 0:
            out.append(dict(mu=mu, kappa=kappa.copy(), sig2_e=sig2["e"], tau2=tau2.copy(),
                            **{f"u_{g}": u[g].copy() for g in groups if g != "e"},
                            **{f"sig2_{g}": sig2[g] for g in groups if g != "e"}))
    return out, {g: v for g, v in groups.items() if g != "e"}, conds, fams


def check_chains(chains, groups):
    """Split R-hat of every effect across chains; prints the worst few."""
    def rhat(x):                       # x: chains x draws
        h = x.shape[1] // 2
        x = np.concatenate([x[:, :h], x[:, h:2 * h]])
        m, n = x.shape
        W = x.var(axis=1, ddof=1).mean()
        B = n * x.mean(axis=1).var(ddof=1)
        return float(np.sqrt(((n - 1) / n * W + B / n) / W))
    res = {"mu": rhat(np.array([[dr["mu"] for dr in c] for c in chains]))}
    for g, (_, lv) in groups.items():
        for j, name in enumerate(lv):
            res[f"{g}:{name}"] = rhat(np.array([[dr[f"u_{g}"][j] for dr in c] for c in chains]))
    K = np.array([[dr["kappa"] for dr in c] for c in chains])
    for j in range(K.shape[2]):
        res[f"kappa:{j}"] = rhat(K[:, :, j])
    worst = sorted(res.items(), key=lambda kv: -kv[1])[:5]
    print("split R-hat, worst five:", ", ".join(f"{k} {v:.3f}" for k, v in worst))


def summarise(draws, groups, conds):
    q = lambda a: (np.mean(a), np.percentile(a, 2.5), np.percentile(a, 50), np.percentile(a, 97.5))
    rows = []
    for g, (_, lv) in groups.items():
        U = np.stack([dr[f"u_{g}"] for dr in draws])
        for j, name in enumerate(lv):
            m, lo, med, hi = q(U[:, j])
            rows.append(dict(term=TERM_NAME[g], level=name,
                             mean=m, lo=lo, median=med, hi=hi))
    K = np.stack([dr["kappa"] for dr in draws])
    for j, c in enumerate(conds):
        m, lo, med, hi = q(K[:, j])
        rows.append(dict(term="kappa", level=c, mean=m, lo=lo, median=med, hi=hi))
    m, lo, med, hi = q(np.array([dr["mu"] for dr in draws]))
    rows.append(dict(term="mu", level="", mean=m, lo=lo, median=med, hi=hi))
    for g in list(groups) + ["e"]:
        s = np.sqrt(np.array([dr[f"sig2_{g}"] for dr in draws]))
        m, lo, med, hi = q(s)
        rows.append(dict(term="sigma", level=g, mean=m, lo=lo, median=med, hi=hi))
    effects = pd.DataFrame(rows)

    # variance shares from the realised effects (finite-population), per draw
    # The instruction now enters the denominator. kappa is a fixed effect, so it
    # has no population variance to quote, but its realized levels have a mean
    # square exactly as alpha's do, and the reference condition contributes a
    # zero to that mean. Leaving it out was defensible while the split was
    # between the model and the benchmark; with the benchmark terms gone the only
    # comparison left is the model against the instruction, so both are measured
    # the same way, over the levels that actually ran.
    n_kappa = len(conds) + 1        # the conditions fitted, plus the C0 reference
    shares = []
    for dr in draws:
        v = {g: float(np.mean(dr[f"u_{g}"] ** 2)) for g in groups}
        v["kappa"] = float((dr["kappa"] ** 2).sum() / n_kappa)
        v["e"] = float(dr["sig2_e"])
        tot = sum(v.values())
        shares.append({k: val / tot for k, val in v.items()})
    S = pd.DataFrame(shares).rename(columns=SHARE_NAME)
    share_rows = []
    for k in S.columns:
        m, lo, med, hi = q(S[k].to_numpy())
        share_rows.append(dict(component=k, mean=m, lo=lo, median=med, hi=hi))
    shares_df = pd.DataFrame(share_rows)

    return effects, shares_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep.csv")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--config", required=True, help="workflow config (yaml)")
    ap.add_argument("--crossed", action="store_true",
                    help="fit across every benchmark with a benchmark and a "
                         "model x benchmark term (paper Figure 3c)")
    ap.add_argument("--reasoning", action="store_true",
                    help="admit the _nothink arms, pooled into their base "
                         "model (paper Figure 3d, 9/24)")
    ap.add_argument("--one-stage", action="store_true",
                    help="fit the crossed model to the per-task counts "
                         "(implies --crossed; paper Figure 3d, 9/24)")
    ap.add_argument("--task-counts", default="data/task_counts.csv")
    ap.add_argument("--iter", type=int, default=N_ITER)
    ap.add_argument("--burn", type=int, default=BURN)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first chain")
    ap.add_argument("--chains", type=int, default=CHAINS)
    ap.add_argument("--b0", type=float, default=0.5, help="inverse-gamma scale of the variance hyperpriors")
    args = ap.parse_args()
    import yaml
    with open(args.config) as fh:
        configure(yaml.safe_load(fh))
    models = MAIN_MODELS
    args.crossed |= args.one_stage
    d = load_cells(args.sweep, models, family=None if args.crossed else FIT_FAMILY,
                   reasoning=args.reasoning)
    measured = measured_by_benchmark(args.sweep, models)
    tag = args.tag or (("_crossed" if args.crossed else "")
                       + ("_reasoning" if args.reasoning else ""))
    blocks = BLOCKS["crossed" if args.crossed else "single"]
    if args.crossed:
        d["mf"] = d["model_short"] + " x " + d["family_task"]
    where = "every benchmark" if args.crossed else FIT_FAMILY
    print(f"fit on {where}: {len(d)} cells  models: {sorted(d['model_short'].unique())}")
    print(d.groupby(["model_short", "condition"]).size().unstack(fill_value=0))
    if args.crossed:
        print(d.groupby(["family_task", "model_short"]).size().unstack(fill_value=0))
    draws = []
    if args.one_stage:
        from concurrent.futures import ProcessPoolExecutor
        rows = load_task_rows(d, args.task_counts)
        print(f"one stage: {len(rows)} tasks with a contradicted row")
        kw = dict(n_iter=args.iter, burn=args.burn, b0=args.b0, blocks=blocks)
        with ProcessPoolExecutor(args.chains) as ex:
            futs = [ex.submit(gibbs_one_stage, rows, seed=args.seed + k, **kw)
                    for k in range(args.chains)]
            chains = [f.result() for f in futs]
        for chain, groups, conds, fams in chains:
            draws += chain
        check_chains([c[0] for c in chains], groups)
    else:
        for k in range(args.chains):
            chain, groups, conds = gibbs(d, seed=args.seed + k, b0=args.b0, blocks=blocks)
            draws += chain
    effects, shares_df = summarise(draws, groups, conds)
    if args.one_stage:
        tau = np.sqrt(np.stack([dr["tau2"] for dr in draws]))
        effects = pd.concat([effects, pd.DataFrame(
            [dict(term="tau", level=f, mean=tau[:, j].mean(),
                  lo=np.percentile(tau[:, j], 2.5), median=np.median(tau[:, j]),
                  hi=np.percentile(tau[:, j], 97.5)) for j, f in enumerate(fams)])],
            ignore_index=True)
    os.makedirs(args.outdir, exist_ok=True)
    d.to_csv(os.path.join(args.outdir, f"decomposition_cells{tag}.csv"), index=False)
    effects.to_csv(os.path.join(args.outdir, f"decomposition_effects{tag}.csv"), index=False)
    shares_df.to_csv(os.path.join(args.outdir, f"decomposition_shares{tag}.csv"), index=False)
    measured.to_csv(os.path.join(args.outdir, f"decomposition_fitted{tag}.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n== effects (posterior median [95%])"); print(effects.round(3).to_string(index=False))
    print("\n== variance shares"); print(shares_df.round(3).to_string(index=False))
    print("\n== measured C0 rate per (model, benchmark)")
    print(measured.drop(columns=["eta", "eta_se"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
