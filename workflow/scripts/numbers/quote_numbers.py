"""Every number the paper quotes, computed from the estimates, beside the paper's value.

Reads the list in config/paper_numbers.yaml, computes each `key` from the
estimates the figure context points at (released or recomputed), and writes

    results/numbers.json   one record per number: key, where, text, paper,
                           value, match, source
    results/numbers.csv    the same, flat

A number is a match when the computed value, rounded half-up to the precision
the paper prints, equals the paper's value (`kind: upper_bound`: the computed
value is below the bound; `kind: bool`: the claim holds). Nothing here fixes a
mismatch: it is reported.

Run through the workflow (rule `numbers`), which sets FIG_CONTEXT.
"""
from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import figstyle as fs  # noqa: E402
import evidence as ev  # noqa: E402
import fig_B3 as b3  # noqa: E402
import fig_F3 as f3  # noqa: E402

EST = fs.EST
DECOMP = fs.DECOMP_DIR


# ------------------------------------------------------------------ helpers
FIT = "_crossed_onestage"      # the fit Fig. 3 (d) and B.4 report


def round_half_up(x: float, digits: int) -> float:
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def spearman_ci(x, y, n_boot=5000, seed=0):
    """Spearman rho with a 95 % percentile bootstrap interval over settings."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        k = rng.integers(0, len(x), len(x))
        if len(set(x[k])) > 1 and len(set(y[k])) > 1:
            draws.append(spearmanr(x[k], y[k]).statistic)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


def f3_cells(benchmark: str) -> pd.DataFrame:
    """Every setting F3 (a-c) draws for one benchmark, reasoning on and off."""
    name = fs.BENCHMARKS[benchmark]["figure_name"]
    d = pd.concat([f3.load_settings(), f3.load_settings(off=True)], ignore_index=True)
    return d[d.benchmark == name]


def hp_params() -> pd.DataFrame:
    """cbrm_params rows of every main-text model at the paper's instructions."""
    rows = []
    for m in fs.MODELS:
        d = fs.arms_only(pd.read_csv(fs.MODEL_DIR[m] / "cbrm_params.csv"))
        rows.append(d.assign(model_key=m))
    return pd.concat(rows, ignore_index=True)


def hp_sweep() -> pd.DataFrame:
    """Sweep cells of every hidden-profile main and reasoning-off run."""
    runs = set(fs.MODEL_HP_RUN.values()) | {ab["run"] for ab in fs.MODEL_ABLATION.values()
                                           if ab["kind"] == "switch"}
    d = fs.arms_only(pd.read_csv(fs.SWEEP))
    d = d[(d.run.isin(runs)) & (d.probe.fillna("unknown") == fs.PROBE)]
    return d.dropna(subset=["margin", "interaction_gain"])


def switch_data() -> dict:
    return b3.load()


def effects() -> pd.DataFrame:
    return pd.read_csv(DECOMP / f"decomposition_effects{FIT}.csv")


def shares() -> pd.DataFrame:
    return pd.read_csv(DECOMP / f"decomposition_shares{FIT}.csv").set_index("component")


def cells() -> pd.DataFrame:
    return pd.read_csv(DECOMP / f"decomposition_cells{FIT}.csv")


def overlap(lo1, hi1, lo2, hi2) -> bool:
    return max(lo1, lo2) <= min(hi1, hi2)


# ------------------------------------------------------------------ the claims
def compute(key: str, p: dict):
    """Return (value, source) for one key."""
    if key == "withholding_rises_with_conformity":
        m = hp_params().groupby("condition")["c_post"].mean()
        v = [m[a] for a in p["order"]]
        return all(x < y for x, y in zip(v, v[1:])), \
            "cbrm_params.csv c_post, mean over models: " + ", ".join(
                f"{a} {m[a]:.3f}" for a in p["order"])
    if key == "p_above_threshold_rises_with_conformity":
        m = hp_params().groupby("condition")["p_trapped"].mean()
        return bool(m[p["low"]] < m[p["high"]]), \
            "cbrm_params.csv p_trapped = P(c > c*), mean over models: " + ", ".join(
                f"{a} {m[a]:.3f}" for a in fs.ARMS)
    if key == "reasoning_models_above_threshold":
        d = hp_params()
        d = d[d.model_key.map(fs.REASONS)]
        below = d[d.c_post <= d.c_star_post]
        return below.empty, "cbrm_params.csv c_post vs c_star_post; below: " + (
            ", ".join(f"{fs.MODEL_SHORT[r.model_key]} {r.condition}" for r in below.itertuples())
            or "none")
    if key == "reasoning_models_withhold_less_internalize_more":
        d = hp_params()
        d["reasons"] = d.model_key.map(fs.REASONS)
        g = d.groupby("reasons")[["c_post", "a_post"]].mean()
        ok = g.loc[True, "c_post"] < g.loc[False, "c_post"] and g.loc[True, "a_post"] > g.loc[False, "a_post"]
        return bool(ok), ("mean over models and instructions, reasoning / not: c-hat "
                          f"{g.loc[True, 'c_post']:.3f} / {g.loc[False, 'c_post']:.3f}, a-hat "
                          f"{g.loc[True, 'a_post']:.3f} / {g.loc[False, 'a_post']:.3f}")
    if key == "instruction_c_at_high_withholding_end":
        m = hp_sweep().groupby("condition")["margin"].mean()
        return bool(all(m[p["high"]] > m[o] for o in p["others"])), \
            "mean margin c-hat - c* by instruction: " + ", ".join(f"{a} {m[a]:+.3f}" for a in fs.ARMS)
    if key == "gain_levels_off_at_high_margin":
        d = hp_sweep()
        top = d[d.margin >= d.margin.quantile(2 / 3)]
        v = float(top.interaction_gain.mean())
        return abs(v) <= p["tolerance"], f"top third of margins (n={len(top)}, margin >= {top.margin.min():.2f}): mean gain {v:+.3f}"
    if key == "gain_falls_with_margin_hidden_profile":
        d = hp_sweep()
        rho, lo, hi = spearman_ci(d.margin, d.interaction_gain)
        return bool(hi < 0), f"sweep.csv hidden-profile cells (n={len(d)}): Spearman {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    if key.startswith("reasoning_") or key in ("n_reasoning_switch_models",
                                               "gemini_settings_nearly_coincide"):
        data = switch_data()
        models = [m for m in b3.PAIRS]
        sh = {q: {m: data[m]["on"][q]["value"] - data[m]["off"][q]["value"] for m in models}
              for q in ("c", "a", "gamma", "gain")}
        txt = lambda q: ", ".join(f"{fs.MODEL_SHORT[m]} {v:+.3f}" for m, v in sh[q].items())
        if key == "n_reasoning_switch_models":
            return len(models), "config ablation entries of kind switch, drawn by B3"
        if key == "reasoning_off_withholding_no_consistent_shift":
            signs = {np.sign(v) for v in sh["c"].values()}
            return len(signs) > 1, "on - off, withholding: " + txt("c")
        if key == "reasoning_on_internalization_rises_or_stays":
            return all(v >= -p["tolerance"] for v in sh["a"].values()), \
                "on - off, internalization: " + txt("a")
        if key == "reasoning_on_lowers_correction_and_gain":
            ok = all(v < 0 for v in sh["gamma"].values()) and all(v < 0 for v in sh["gain"].values())
            return ok, "on - off, net correction: " + txt("gamma") + "; gain: " + txt("gain")
        if key == "gemini_settings_nearly_coincide":
            m0 = p["model"]
            if m0 not in sh["gain"]:
                return False, f"{m0} has no reasoning switch in the config"
            smallest = min(sh["gain"], key=lambda m: abs(sh["gain"][m]))
            return smallest == m0, "on - off, gain: " + txt("gain")

    if key.startswith("gain_falls_with_margin_") or key == "musique_relation_unclear":
        d = f3_cells(p["benchmark"])
        rho, lo, hi = spearman_ci(d.margin, d.interaction_gain)
        src = f"sweep.csv cells F3 draws (n={len(d)}): Spearman {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        if key == "musique_relation_unclear":
            return bool(lo <= 0 <= hi), src
        return bool(hi < 0), src
    if key == "gains_large_hiddenbench_small_others":
        mx = {b: float(f3_cells(b).interaction_gain.max()) for b in [p["large"]] + p["small"]}
        ok = all(mx[p["large"]] > 2 * mx[b] for b in p["small"])
        return ok, "largest gain per benchmark: " + ", ".join(f"{b} {v:.2f}" for b, v in mx.items())
    if key == "musique_floor_cells_are_mixtral":
        name = fs.BENCHMARKS[p["benchmark"]]["figure_name"]
        f3.NOTES.clear()
        f3.load_settings(); f3.load_settings(off=True)
        floor = [n for n in f3.NOTES if "floor cells" in n]
        runs = " ".join(floor)
        ok = bool(floor) and p["benchmark"] in runs and all(
            (p["benchmark"] + "/") not in part or "mixtral" in part
            for part in runs.replace(",", " ").split())
        return ok, (floor[0][:200] if floor else "no floor cells dropped")
    if key == "musique_cstar_always_zero":
        d = f3_cells(p["benchmark"])
        return bool((d["c_star"] == 0).all()), f"sweep.csv c_star over {len(d)} cells"
    if key == "benchmark_order_withholding":
        b = effects().query("term == 'beta'").set_index("level")["mean"]
        v = [b[k] for k in p["order"]]
        return all(x < y for x, y in zip(v, v[1:])), \
            "benchmark effects (beta means): " + ", ".join(f"{k} {b[k]:+.2f}" for k in p["order"])
    if key == "comparable_shares":
        s = shares().loc[p["components"]]
        ok = all(overlap(s.lo.iloc[i], s.hi.iloc[i], s.lo.iloc[j], s.hi.iloc[j])
                 for i in range(len(s)) for j in range(i + 1, len(s)))
        return ok, "variance shares, mean [95%]: " + ", ".join(
            f"{k} {r['mean']:.2f} [{r.lo:.2f}, {r.hi:.2f}]" for k, r in s.iterrows())
    if key == "instruction_direction":
        k = effects().query("term == 'kappa'").set_index("level")
        ok = k.loc[p["negative"], "hi"] < 0 and k.loc[p["positive"], "lo"] > 0
        return bool(ok), "kappa: " + ", ".join(
            f"{l} {r['mean']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]" for l, r in k.iterrows())
    if key == "most_llm_intervals_cover_zero":
        a = effects().query("term == 'alpha'")
        cover = int(((a.lo <= 0) & (a.hi >= 0)).sum())
        return cover > len(a) / 2, f"{cover} of {len(a)} LLM effects cover zero"
    if key == "most_withholding_llm_varies_by_benchmark":
        fit = pd.read_csv(DECOMP / "decomposition_fitted.csv")
        top = fit.loc[fit.groupby("family")["c_fit_mean"].idxmax()]
        return top["model"].nunique() > 1, "highest measured rate per benchmark: " + ", ".join(
            f"{r.family} {r.model}" for r in top.itertuples())
    if key == "most_intervals_cover_zero_f3d":
        e = effects()
        e = e[e.term.isin(["alpha", "beta"]) | ((e.term == "kappa")
                                                & e.level.isin(fs.FIGCTX["f3_arms"]))]
        cover = int(((e.lo <= 0) & (e.hi >= 0)).sum())
        return cover > len(e) / 2, f"{cover} of {len(e)} drawn intervals cover zero"

    if key == "reasoning_models_share_own_side":
        order, _, share = ev.tabled_summary()
        rs = [m for m in order if fs.REASONS[m]]
        ok = all(share[m]["own"][0] > share[m]["truth"][0] for m in rs)
        return ok, "own vs correct share: " + ", ".join(
            f"{fs.MODEL_SHORT[m]} {share[m]['own'][0]:.2f}/{share[m]['truth'][0]:.2f}" for m in rs)
    if key == "pooling_utilization_tradeoff":
        raw = {v: k for k, v in fs.ARM_NAME.items()}[p["arm"]]
        du = {m: ev.du_estimate(m, raw) for m in fs.MODELS}
        m0 = p["model"]
        rs = [m for m in fs.MODELS if fs.REASONS[m]]
        ok = all(du[m0][0] > du[m][0] and du[m0][2] < du[m][2] for m in rs)
        return ok, f"pooling D / utilization U at {p['arm']}: " + ", ".join(
            f"{fs.MODEL_SHORT[m]} {du[m][0]:.2f}/{du[m][2]:.2f}" for m in fs.MODELS)
    if key == "acc_without_sufficient_evidence_max":
        vals = []
        for m in fs.MODELS:
            a = fs.arms_only(pd.read_csv(fs.MODEL_DIR[m] / "hp_accuracy.csv"))
            a = a[a["k_informed"].astype(str) == "all"]
            vals += list(a["acc_if_insufficient"].dropna())
        return float(max(vals)), "hp_accuracy.csv acc_if_insufficient, max over models x instructions"

    if key == "decomposition_n_benchmarks":
        return int(cells()["family_task"].nunique()), f"decomposition_cells{FIT}.csv"
    if key == "decomposition_n_reasoning_off_models":
        c = cells()
        off = c[c.run.str.rsplit("/", n=1).str[-1].str.endswith(fs.FIGCTX.get("reasoning_off_suffix", "_nothink"))]
        return int(off["model_short"].nunique()), "models with reasoning-off cells: " + ", ".join(
            sorted(off["model_short"].unique()))
    if key == "decomposition_excludes":
        c = cells()
        ok = (p["absent_model"] not in set(c["model_short"])
              and not c.run.str.endswith(p["absent_suffix"]).any()
              and bool((c["n_contra"] >= p["min_contra"]).all()))
        return bool(ok), f"decomposition_cells{FIT}.csv: {len(c)} settings, min n_contra {int(c.n_contra.min())}"
    if key == "decomposition_rhat_max":
        log = DECOMP / "decomposition_onestage.log"
        if not log.exists():
            return None, "sampler log not in this estimates tree (run the estimates entry point)"
        import re
        line = next((l for l in log.read_text().splitlines() if "split R-hat" in l), "")
        vals = [float(v) for v in re.findall(r" (\d+\.\d+)", line)]
        return (max(vals) if vals else None), line.strip()
    raise KeyError(key)


def judge(spec: dict, value) -> tuple[bool, object]:
    kind = spec.get("kind", "value")
    if value is None:
        return None, None
    if kind == "bool":
        return bool(value) == bool(spec["paper"]), bool(value)
    if kind == "upper_bound":          # "below 5 %", "at most 1.03"
        return float(value) <= float(spec["paper"]), float(value)
    digits = int(spec["digits"])
    r = round_half_up(float(value), digits)
    return abs(r - float(spec["paper"])) < 1e-9, r


def main(spec_path: str, out_json: str, out_csv: str) -> int:
    specs = yaml.safe_load(Path(spec_path).read_text())["numbers"]
    records = []
    for s in specs:
        value, source = compute(s["key"], s.get("params", {}))
        ok, shown = judge(s, value)
        records.append(dict(key=s["key"], where=s["where"], text=s["text"],
                            kind=s.get("kind", "value"), paper=s["paper"],
                            value=value if not isinstance(value, (np.floating, np.integer))
                            else value.item(),
                            rounded=shown, match=None if ok is None else bool(ok), source=source))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(dict(estimates=str(EST), numbers=records),
                                         indent=1, default=float) + "\n")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    checked = [r for r in records if r["match"] is not None]
    bad = [r for r in checked if not r["match"]]
    for r in records:
        tag = "n/a     " if r["match"] is None else ("ok      " if r["match"] else "MISMATCH")
        print(f"{tag} {r['key']:44s} paper {r['paper']!s:>6}  here {r['rounded']!s:>8}"
              f"   ({r['where']})")
    print(f"\n{len(checked) - len(bad)} of {len(checked)} checked claims hold"
          f" ({len(records) - len(checked)} not checkable from this estimates tree).")
    return 0


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        main(sm.input.spec, sm.output.json, sm.output.csv)
    else:
        sys.exit(main(*sys.argv[1:4]))
