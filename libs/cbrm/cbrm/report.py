"""One transcript, one condition, in: a diagnosis with its own reliability out.

Every rate this package produces is a conditional probability, and the
conditioning event does not always occur. A team that was never internally split
about the answer has no concealment to measure; a team in which no correct agent
was ever talked out of its answer has no downward flips, so ``r`` pins at 1.0 and
``gamma`` silently collapses onto ``rho``. Both look like ordinary numbers in a
CSV. Across the transcripts in this repo the first case covers 39 of 144 (run,
condition) cells and the second 52 of 144, so this is the common case, not the
edge case.

So the primary object is not a number but a verdict about a number: value,
interval, denominator, and whether the denominator was large enough to mean
anything. ``NaN`` with a stated reason is a correct output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import bayes as bay
from . import estimators as est
from . import phase as ph
from .evidence import evidence_rates, limiting_factor
from .io import Transcript, decisions, error_concentration, p_hat

MIN_N = 50          # below this a rate is reported but flagged WEAK
MIN_N_HARD = 10     # below this it is not worth reporting at all


def _flag(n, min_n=MIN_N):
    if not n:
        return "UNMEASURABLE"
    if n < MIN_N_HARD:
        return "UNMEASURABLE"
    return "WEAK" if n < min_n else "OK"


@dataclass
class Diagnosis:
    """The measured operating point of one team on one arm."""

    run: str
    condition: str
    projection: str
    trigger: str
    rates: dict
    intervals: dict
    meta: dict
    notes: list[str] = field(default_factory=list)
    evidence: dict | None = None
    alt_rates: dict | None = None       # the other projection, for comparison
    post: dict | None = None            # posterior summary, when asked for

    # ------------------------------------------------------------- accessors
    def est(self, name, n_key=None) -> est.Estimate:
        lo, hi = self.intervals.get(name, (np.nan, np.nan))
        n = int(self.rates.get(n_key, 0)) if n_key else 0
        return est.Estimate(self.rates.get(name, np.nan), lo, hi, n,
                            _flag(n) if n_key else "")

    @property
    def measurable(self) -> bool:
        return _flag(self.rates.get("n_contra", 0)) != "UNMEASURABLE"

    @property
    def margin_interval(self):
        return self.intervals.get("margin", (np.nan, np.nan))

    @property
    def p_trapped(self) -> float:
        """Posterior probability the team is above the boundary.

        The three-valued verdict is a decision rule; this is the quantity the
        rule is thresholding. Available only when ``bayes=True``, because it
        needs a joint posterior rather than an interval per parameter.
        """
        return self.post.get("p_trapped", np.nan) if self.post else np.nan

    @property
    def verdict(self) -> str:
        lo, hi = self.margin_interval
        return ph.verdict(self.rates.get("margin", np.nan), lo, hi,
                          min_n_ok=self.measurable)

    @property
    def levers(self):
        return ph.levers(self.rates.get("c", np.nan), self.rates.get("a", np.nan),
                         self.rates.get("gamma", np.nan))

    # ------------------------------------------------------------- rendering
    def to_frame(self) -> pd.DataFrame:
        row = dict(run=self.run, condition=self.condition,
                   projection=self.projection, trigger=self.trigger,
                   **{k: v for k, v in self.meta.items()})
        for k in ("c", "c_strict", "real_gap_share", "a", "rho", "r", "gamma",
                  "c_star", "c_star_fraction", "c_star_any", "margin"):
            row[k] = self.rates.get(k, np.nan)
            lo, hi = self.intervals.get(k, (np.nan, np.nan))
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        row["trigger_ratio"] = self.meta.get("trigger_ratio", np.nan)
        row["eps_belief"] = self.meta.get("eps_belief", np.nan)
        row["eps_express"] = self.meta.get("eps_express", np.nan)
        for k in ("n_contra", "n_conceal", "n_conc_pairs", "n_wrong_rows",
                  "n_right_rows", "n_rows"):
            row[k] = self.rates.get(k, 0)
        row["verdict"] = self.verdict
        if self.post:
            row["p_trapped"] = self.post.get("p_trapped", np.nan)
            for k in ("c", "a", "rho", "r", "gamma", "c_star", "margin"):
                if k in self.post:
                    m, lo, hi = self.post[k]
                    row[f"{k}_post"] = m
                    row[f"{k}_post_lo"], row[f"{k}_post_hi"] = lo, hi
        row["cheapest_lever"] = ph.cheapest_lever(self.levers)
        row["measurable"] = self.measurable
        row["notes"] = " | ".join(self.notes)
        if self.alt_rates:
            row["c_alt_projection"] = self.alt_rates.get("c", np.nan)
            row["c_star_alt_projection"] = self.alt_rates.get("c_star", np.nan)
        if self.evidence:
            for k in ("c_ev", "n_c_ev", "c_ev_opinion", "n_c_ev_opinion",
                      "c_ev_any", "n_c_ev_any", "share_rate", "share_truth_rate",
                      "self_serving_rate", "D", "U", "pooled_fraction"):
                row[f"ev_{k}"] = self.evidence.get(k, np.nan)
            row["limiting_factor"] = limiting_factor(self.evidence.get("D", np.nan),
                                                     self.evidence.get("U", np.nan))
        return pd.DataFrame([row])

    def __str__(self):
        m = self.meta
        L = [f"Diagnosis(run={self.run!r}, condition={self.condition!r})",
             f"  {m.get('n_tasks', '?')} tasks x N={m.get('n_agents', '?')} "
             f"x T={m.get('T', '?')}   probe={m.get('probe', '?')}  "
             f"topology={m.get('topology', '?')}",
             f"  projection={self.projection}  trigger={self.trigger}  "
             f"labels={m.get('n_labels', '?')}  "
             f"error concentration={m.get('error_concentration', float('nan')):.2f}  "
             f"base noise={m.get('eps_belief', float('nan')):.2f}  "
             f"model={m.get('model', '?')}",
             ""]
        for label, key, nkey in (("c   concealment  ", "c", "n_contra"),
                                 ("a   internalise  ", "a", "n_conc_pairs"),
                                 ("r   re-derive ok ", "r", "n_wrong_rows"),
                                 ("gam net repair   ", "gamma", "n_wrong_rows")):
            L.append(f"  {label} {self.est(key, nkey)!r}")
        cs = self.est("c_star")   # a derived quantity: no denominator of its own
        L.append(f"  c*  critical      {cs.value:5.3f} "
                 f"[{cs.lo:.3f}, {cs.hi:.3f}]        "
                 f"[trigger range {self.rates.get('c_star_fraction', np.nan):.3f}"
                 f"-{self.rates.get('c_star_any', np.nan):.3f}]")
        lo, hi = self.margin_interval
        L.append(f"  margin c - c*    {self.rates.get('margin', np.nan):+.3f} "
                 f"[{lo:+.3f}, {hi:+.3f}]  -> {self.verdict}")
        if self.post:
            L.append(f"  P(trapped) = {self.post.get('p_trapped', float('nan')):.3f}"
                     f"      posterior, tasks clustered")
            for k, lab in (("r", "r   "), ("c_star", "c*  ")):
                if k in self.post:
                    m, lo, hi = self.post[k]
                    L.append(f"    {lab} posterior {m:5.3f} [{lo:.3f}, {hi:.3f}]")
        L.append("")
        L.append(f"  cheapest lever to the boundary: {ph.cheapest_lever(self.levers)}")
        for lev in self.levers:
            L.append(f"    {lev!r}")
        if self.evidence and self.evidence.get("n_slots"):
            e = self.evidence
            L.append("")
            L.append(f"  evidence level (T={e['T']}):")
            L.append(f"    c_ev  held decisive evidence, board pointed the "
                     f"other way, stayed silent   {e['c_ev']:.3f}  n={e['n_c_ev']}")
            L.append(f"          conditioned on opinions instead: "
                     f"{e.get('c_ev_opinion', float('nan')):.3f}  "
                     f"n={e.get('n_c_ev_opinion', 0)}")
            L.append(f"    share rate {e['share_rate']:.3f}   of shares, "
                     f"{e['share_truth_rate']:.3f} favour the truth, "
                     f"{e['self_serving_rate']:.3f} favour own answer")
            L.append(f"    D disclosure {e['D']:.3f}   U usage {e['U']:.3f}   "
                     f"-> limiting factor: {limiting_factor(e['D'], e['U'])}")
        if self.notes:
            L.append("")
            L.append("  notes:")
            L += [f"    - {n}" for n in self.notes]
        return "\n".join(L)

    __repr__ = __str__


# --------------------------------------------------------------------- driver
def diagnose(t: Transcript, condition: str, projection: str = "plurality",
             trigger: str = "fraction", n_boot: int = 2000, seed: int = 0,
             evidence=None, q: int | None = None, run: str = "",
             bayes: bool = False, n_draw: int = 20000) -> Diagnosis:
    """Measure one arm of one transcript and say how far to trust the result."""
    dec = decisions(t, condition, projection=projection)
    cnt = est.counts(dec)
    R = est.rates(cnt, trigger=trigger)
    draws = est.cluster_bootstrap(dec, n_boot=n_boot, seed=seed, trigger=trigger)
    keys = ("c", "c_strict", "real_gap_share", "a", "rho", "r", "gamma",
            "c_star", "c_star_fraction", "c_star_any", "margin")
    intervals = {k: est.interval(draws, k) for k in keys}

    notes = []
    labels = t.labels
    alt = None
    kappa = error_concentration(t)
    kappa_mean = float(kappa.mean()) if len(kappa) else np.nan
    if len(labels) > 2:
        other = "binary" if projection != "binary" else "plurality"
        alt = est.rates(est.counts(decisions(t, condition, projection=other)),
                        trigger=trigger)
        gap = abs((alt.get("c") or np.nan) - (R.get("c") or np.nan))
        notes.append(
            f"{len(labels)} answer labels; under the {other} projection c would "
            f"read {alt.get('c', float('nan')):.3f} against "
            f"{R.get('c', float('nan')):.3f} here (gap {gap:.3f}).")
        if np.isfinite(kappa_mean):
            notes.append(
                f"the wrong side is {kappa_mean:.2f} concentrated on one answer "
                + ("-- the team is in the two-choice regime, where the c* formula "
                   "applies and the two projections agree."
                   if kappa_mean > 0.8 else
                   "-- the errors are split, so the two-choice reduction does not "
                   "apply: disagreement among wrong answers still triggers "
                   "re-derivation, and the binary projection would score a "
                   "contradiction where the model sees none."))
    # A plurality that is almost never defined means the answer set is
    # effectively unbounded and this is not a K-choice task at all.
    if projection != "binary" and cnt["n_rows"]:
        undefined = float(dec["M"].isna().mean())
        if undefined > 0.5:
            notes.append(
                f"the visible plurality is undefined on {undefined:.0%} of rows: "
                f"with {len(labels)} distinct answers this is a free-response "
                f"task, not a K-choice one. Concealment is barely defined here; "
                f"the binary reading has a denominator but is a two-state "
                f"reduction, and the evidence-level estimators are the honest "
                f"route.")
    if cnt["n_contra"] == 0:
        notes.append(
            "no agent-round had the visible majority contradict its belief: this "
            "team was never internally split about the answer, so concealment has "
            "no denominator. Use the evidence-level estimators instead.")
    elif cnt["n_contra"] < MIN_N:
        notes.append(f"only {cnt['n_contra']} contradicted agent-rounds; c and a "
                     f"are reported but WEAK.")
    if cnt["flip_down"] == 0 and cnt["n_right_rows"] > 0:
        notes.append(
            "no correct agent was ever talked out of its answer, so r pins at "
            "1.0 and gamma collapses onto rho. c* is an upper bound here.")
    if cnt["n_right_rows"] == 0:
        notes.append("no correct agent ever saw dissent, so r is not identified.")
    tc = est.trigger_check(dec)
    if (min(tc["n_tied"], tc["n_decided"]) >= 40 and np.isfinite(tc["ratio"])
            and not (0.67 <= tc["ratio"] <= 1.5)):
        notes.append(
            f"the model gates re-derivation on visible dissent alone, but this "
            f"team re-derives {tc['ratio']:.2f}x as often per unit of dissent "
            f"when the panel it saw was tied as when it was decided "
            f"(n = {tc['n_tied']} / {tc['n_decided']}). d is not a sufficient "
            f"statistic for engagement here, so rho and gamma carry a "
            f"specification-dependent component and should not be compared "
            f"across runs with different ratios.")
    if np.isfinite(R.get("gamma", np.nan)) and R["gamma"] <= 0:
        if cnt["flip_up"] == 0 and cnt["n_wrong_rows"] > 0:
            notes.append(
                f"gamma = 0 from {cnt['n_wrong_rows']} opportunities: not one "
                f"wrong agent that saw dissent ended the round correct. There is "
                f"no measured current out of a wrong consensus, so c* = 0 and any "
                f"concealment is above critical. This is a measurement, not a "
                f"missing denominator -- but it also means r is a ratio of zero "
                f"to zero and is not identified.")
        else:
            notes.append(
                "gamma <= 0: re-derivations lose more correct answers than they "
                "win (r <= 1/2), so deliberation is not truth-seeking on this arm, "
                "c* = 0, and any concealment is above critical.")
    # The caveat belongs on the rows it applies to. Emitting it on every row
    # read as if a separate-probe estimate were also suspect, and the note is
    # carried into sweep.csv, so the sweep said "probe=separate" and then warned
    # about one-call elicitation in the same sentence.
    if t.probe == "separate":
        notes.append(
            "probe=separate. The private answer was asked in its own call, with "
            "the same transcript and no mention of publication, so the pair "
            "consistency a model maintains after writing PUBLIC is not in this "
            "estimate. On four matched pairs -- same items, same model, same N, "
            "only the probe differing -- moving from one call to two shifts a by "
            "0.49, c by 0.11, D by 0.06 and c_ev by 0.010 on average, always "
            "toward more concealment, so a joint-probe number for this arm would "
            "read lower than this one.")
    else:
        notes.append(
            f"probe={t.probe}. Asking for the private and public answers in one "
            f"call attributes to concealment what two separate calls attribute "
            f"to a changed mind. Measured on four matched pairs -- same items, "
            f"same model, same N, only the probe differing -- the mean absolute "
            f"shift is 0.49 for a, 0.11 for c, 0.06 for D and 0.010 for c_ev. So "
            f"a is close to a measurement of the probe, c moves but survives, "
            f"and the evidence-level rate is the one to quote when the probe is "
            f"not under your control.")

    ph_series = p_hat(t)
    meta = dict(n_tasks=t.n_tasks, n_agents=t.n_agents, T=t.rounds,
                probe=t.probe, model=str(t.meta.get("model", "")),
                topology=str(t.meta.get("topology", "")), n_labels=len(labels),
                p_hat=float(ph_series.mean()) if len(ph_series) else np.nan,
                error_concentration=kappa_mean)
    q = int(q or max(3, min(t.n_agents - 1, 9)))
    meta["q"] = q
    meta["p_star"] = ph.p_star(R.get("c"), R.get("a"), R.get("rho"), R.get("r"), q=q)
    meta["trigger_ratio"] = tc["ratio"]
    bn = est.base_noise(t, condition)
    meta["eps_belief"] = bn["eps_belief"]
    meta["eps_express"] = bn["eps_express_loose"]
    if bn["n_strict"] >= 100 and np.isfinite(bn["eps_belief"]) \
            and bn["eps_belief"] > 0.15:
        notes.append(
            f"base noise: in {bn['n_strict']} slots where every visible "
            f"neighbour expressed what this agent already believed, and nobody "
            f"had disclosed anything, the belief still moved at a redraw rate of "
            f"{bn['eps_belief']:.2f}. The model has no channel that does that, so "
            f"that fraction of the belief dynamics is outside it.")

    post = None
    if bayes:
        post = bay.summary(bay.posterior(dec, n_draw=n_draw, seed=seed))
        if cnt["flip_down"] == 0 and cnt["n_right_rows"] > 0:
            notes.append(
                f"the point estimate pins r at 1.0 and the bootstrap interval "
                f"collapses to a width of zero, because every resample has the "
                f"same empty numerator. The posterior reports "
                f"{post['r'][0]:.3f} [{post['r'][1]:.3f}, {post['r'][2]:.3f}] "
                f"instead, which is what the data support.")

    ev = None
    if evidence is not None:
        ev = evidence_rates(t, condition, evidence)
        if ev.get("tagged") is False:
            notes.append("the item set does not tag which answer each finding "
                         "supports, so c_ev and the D/U split are withheld; the "
                         "share and silence rates still apply.")

    return Diagnosis(run=run or t.source, condition=condition,
                     projection=projection, trigger=trigger, rates=R,
                     intervals=intervals, meta=meta, notes=notes, evidence=ev,
                     alt_rates=alt, post=post)


def diagnose_all(t: Transcript, conditions=None, **kw) -> pd.DataFrame:
    frames = [diagnose(t, c, **kw).to_frame() for c in (conditions or t.conditions)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
