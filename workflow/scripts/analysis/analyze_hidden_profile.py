"""Hidden-profile transcript -> parameters, information pooling, phase boundary.

The MCQ runs could measure the model's parameters but not its central claim,
because the items never put a team in the state the claim is about. Here the
state is a design variable: ``k`` agents out of N hold enough private evidence to
be right alone, so the initial competence is p = k/N by construction and the
phase boundary p_c(phi) can be traced rather than assumed.

Three families of number come out.

Manipulation check   round-0 accuracy per k level, against the designed p. If
                     the model cannot count findings, p_hat will not track
                     p_design and nothing downstream means anything.
Mechanism            phi (concealment), lambda (internalisation), rho*r (repair)
                     as before -- and, new here, *disclosure*: whether an agent
                     puts one of its own private findings on the table. The
                     theory says repair runs on visible dissent; disclosure is
                     that quantity, observed rather than inferred.
Outcome              collective accuracy at the last round, by k level and
                     framing condition, against the round-0 independent vote on
                     the same items. This is the empirical phase diagram.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_llm import load, majority, wilson  # noqa: E402


# Items generated before the seat -> agent rename use the old key names. The
# generator is deterministic, so regenerating gives the same items under the new
# names -- but the transcripts already on disk are keyed to the old file, and
# rewriting it would make snakemake rerun every model call that depends on it.
_ITEM_ALIASES = {"seat_facts": "agent_facts"}
_DESIGN_ALIASES = {"n_seats": "n_agents", "per_seat_tally": "per_agent_tally",
                   "solo_correct_seats": "solo_correct_agents"}


def _rename_legacy(item):
    for old, new in _ITEM_ALIASES.items():
        if new not in item and old in item:
            item[new] = item[old]
    design = item.get("design", {})
    for old, new in _DESIGN_ALIASES.items():
        if new not in design and old in design:
            design[new] = design[old]
    return item


def load_items(path):
    return {json.loads(l)["id"]: _rename_legacy(json.loads(l))
            for l in Path(path).read_text().splitlines() if l.strip()}


def manipulation_check(rows, items):
    """Did the design do what it says? One row per designed competence level.

    ``p_hat`` should track ``p_design``; if it does not, the panel is not using
    the evidence and nothing downstream is about concealment. ``count_correct``
    is the agent's own tally of the findings it can see, which the protocol makes
    it state before answering, so a panel that cannot count is visible as such
    rather than as noise.
    """
    per_k = defaultdict(lambda: dict(correct=[], view=[], counted=[]))
    for r in rows:
        if r["round"] != 0 or not r["private"]:
            continue
        it = items[r["task_id"]]
        k = it["design"]["n_informed"]
        tally = it["design"]["per_agent_tally"][str(r["agent"])]
        view_winner = "A" if tally["A"] > tally["B"] else "B"
        per_k[k]["correct"].append(r["private"] == r["answer"])
        per_k[k]["view"].append(r["private"] == view_winner)
        if r.get("count") is not None:
            per_k[k]["counted"].append(bool(r.get("count_correct")))
    out = []
    for k in sorted(per_k):
        d = per_k[k]
        out.append(dict(
            k_informed=k, n_agent_answers=len(d["correct"]),
            p_design=k / items[next(iter(items))]["design"]["n_agents"],
            p_hat=float(np.mean(d["correct"])),
            picks_own_view_winner=float(np.mean(d["view"])),
            count_correct=float(np.mean(d["counted"])) if d["counted"] else np.nan))
    return pd.DataFrame(out)


def measure(rows, items, rounds):
    by = {(r["task_id"], r["agent"], r["round"], r["branch"]): r for r in rows}
    answer = {r["task_id"]: r["answer"] for r in rows}
    k_of = {t: items[t]["design"]["n_informed"] for t in items}
    p_design = {t: items[t]["design"]["p_design"] for t in items}

    r0 = defaultdict(list)
    r0_pub = defaultdict(list)
    for r in rows:
        if r["round"] == 0:
            r0[r["task_id"]].append(r["private"] == r["answer"])
            r0_pub[r["task_id"]].append(r["public"])
    p_hat = {t: float(np.mean(v)) for t, v in r0.items() if v}
    vote_hit = {t: majority(v) == answer[t] for t, v in r0_pub.items()}

    conditions = sorted({r["branch"] for r in rows if r["branch"] != "r0"})
    param_rows, acc_rows = [], []
    for cond in conditions:
        n_contra = n_conceal = n_conc_pairs = n_internalise = 0
        d_wrong = d_right = 0.0
        flip_up = flip_down = 0
        n_share_slots = n_shared_any = n_shared_truth = 0
        n_holds_truth_note = n_holds_and_shares = 0
        pooled = defaultdict(set)      # task -> ids of truth-favouring notes disclosed

        for r in rows:
            if r["branch"] != cond or r["round"] == 0:
                continue
            it = items[r["task_id"]]
            truth = answer[r["task_id"]]
            notes = it["agent_facts"][str(r["agent"])]
            prev_branch = "r0" if r["round"] == 1 else cond
            prev = by.get((r["task_id"], r["agent"], r["round"] - 1, prev_branch))
            if prev is None or prev["private"] is None:
                continue
            c_prev, priv, pub = prev["private"], r["private"], r["public"]
            nb_pub = r.get("nb_public") or []
            M = majority(nb_pub)
            if priv is None or pub is None:
                continue

            # ---- disclosure: did the agent put a private finding on the table?
            n_share_slots += 1
            if r.get("share"):
                n_shared_any += 1
                if r.get("share_about") == truth:
                    n_shared_truth += 1
                    pooled[r["task_id"]].add((r["agent"], r["share"]))
            if any(f["about"] == truth for f in notes):
                n_holds_truth_note += 1
                if r.get("share_about") == truth:
                    n_holds_and_shares += 1

            # ---- concealment / internalisation / repair, as in the MCQ runs
            if M is None:
                continue
            seen = [a for a in nb_pub if a]
            d = sum(a != c_prev for a in seen) / len(seen) if seen else 0.0
            contradicted = M != c_prev
            if contradicted:
                n_contra += 1
            concealed = contradicted and (pub == M)
            if concealed:
                n_conceal += 1
                n_conc_pairs += 1
                n_internalise += int(priv == pub)
            elif d > 0:
                if c_prev != truth:
                    d_wrong += d
                    flip_up += int(priv == truth)
                else:
                    d_right += d
                    flip_down += int(priv != truth)

        phi = wilson(n_conceal, n_contra)
        lam = wilson(n_internalise, n_conc_pairs)
        rr = wilson(flip_up, int(round(d_wrong)))
        bad = wilson(flip_down, int(round(d_right)))
        rho = rr[0] + bad[0]
        share_any = wilson(n_shared_any, n_share_slots)
        share_truth = wilson(n_holds_and_shares, n_holds_truth_note)
        param_rows.append(dict(
            condition=cond,
            phi_hat=phi[0], phi_lo=phi[1], phi_hi=phi[2], n_contradicted=n_contra,
            lambda_hat=lam[0], lambda_lo=lam[1], lambda_hi=lam[2],
            n_concealed_pairs=n_conc_pairs,
            rho_r_hat=rr[0], rho_r_lo=rr[1], rho_r_hi=rr[2],
            rho_hat=min(rho, 1.0), r_hat=rr[0] / rho if rho > 0 else np.nan,
            share_rate=share_any[0], share_lo=share_any[1], share_hi=share_any[2],
            n_share_slots=n_share_slots,
            disclose_truth_rate=share_truth[0], disclose_lo=share_truth[1],
            disclose_hi=share_truth[2], n_holds_truth_note=n_holds_truth_note))

        # ---- was the disclosed information enough to decide correctly?
        # Separates "the panel never put the evidence on the table" from "the
        # evidence was on the table and the panel voted the other way anyway".
        # The board plus the shared briefing is exactly what every agent could
        # see, so this is a property of the transcript, not of the model.
        disclosed = defaultdict(lambda: {"A": 0, "B": 0})
        for r in rows:
            if r["branch"] == cond and r.get("share_about"):
                disclosed[r["task_id"]][r["share_about"]] += 1
        sufficient = {}
        for t, it in items.items():
            base = Counter(f["about"] for f in it["shared_facts"])
            tally = {lab: base.get(lab, 0) + disclosed[t][lab] for lab in ("A", "B")}
            other = "A" if answer[t] == "B" else "B"
            sufficient[t] = tally[answer[t]] > tally[other]

        # ---- outcome, by design level
        final = defaultdict(list)
        for r in rows:
            if r["branch"] == cond and r["round"] == rounds:
                final[r["task_id"]].append(r["public"])
        levels = sorted({k_of[t] for t in final})
        for level in list(levels) + ["all"]:
            sel = [t for t in final
                   if (level == "all" or k_of[t] == level)]
            if not sel:
                continue
            hits = [majority(final[t]) == answer[t] for t in sel]
            a = wilson(int(np.sum(hits)), len(hits))
            votes = [vote_hit[t] for t in sel if t in vote_hit]
            # how much of the decisive private evidence reached the table
            frac_pooled = []
            for t in sel:
                total = sum(1 for s in items[t]["agent_facts"]
                            for f in items[t]["agent_facts"][s]
                            if f["about"] == answer[t])
                frac_pooled.append(len(pooled.get(t, ())) / total if total else np.nan)
            suff = [t for t in sel if sufficient[t]]
            insuff = [t for t in sel if not sufficient[t]]
            acc_rows.append(dict(
                condition=cond, k_informed=level,
                p_design=(np.nan if level == "all"
                          else float(np.mean([p_design[t] for t in sel]))),
                p_hat=float(np.mean([p_hat[t] for t in sel if t in p_hat])),
                n_items=len(sel), accuracy=a[0], acc_lo=a[1], acc_hi=a[2],
                vote_accuracy=float(np.mean(votes)) if votes else np.nan,
                pooled_fraction=float(np.nanmean(frac_pooled)),
                sufficient_rate=len(suff) / len(sel),
                acc_if_sufficient=(float(np.mean([majority(final[t]) == answer[t]
                                                  for t in suff])) if suff else np.nan),
                acc_if_insufficient=(float(np.mean([majority(final[t]) == answer[t]
                                                    for t in insuff]))
                                     if insuff else np.nan)))
    return pd.DataFrame(param_rows), pd.DataFrame(acc_rows), p_hat


# --------------------------------------------------------------- concealment
# The estimator in ``measure`` is the one the MCQ runs used, kept so the numbers
# stay comparable. It has two known weaknesses, and this section measures both
# rather than arguing about them.
#
#   persuasion counted as concealment
#       it asks "was the agent's PREVIOUS private answer contradicted by the
#       neighbour majority, and did it then publish that majority?" An agent that
#       was genuinely persuaded -- new private answer = majority = public -- has
#       no private/public gap at all, but is counted.
#   independence
#       the Wilson interval treats every agent-round as one independent draw.
#       Rows repeat within an item (N agents x T rounds on the same question),
#       so the effective sample is closer to the number of items.
#
# The strict estimator conditions on the SAME-round private answer and requires
# an actual gap:  phi_strict = P(public = M and public != private | private != M).
# The dose-response splits it by how many neighbours were visibly dissenting,
# which is the quantity the theory's trigger is a function of and which the
# N = 25 panels are the first to have any range in.


def _boot_ci(pairs, n_boot=2000, seed=0):
    """Cluster bootstrap over items: resample TASKS, not agent-rounds.

    ``pairs`` maps task_id -> (hits, trials). Returns (rate, lo, hi).
    """
    tasks = sorted(pairs)
    hits = np.array([pairs[t][0] for t in tasks], dtype=float)
    trials = np.array([pairs[t][1] for t in tasks], dtype=float)
    if trials.sum() == 0:
        return (np.nan, np.nan, np.nan)
    rate = float(hits.sum() / trials.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tasks), size=(n_boot, len(tasks)))
    h, n = hits[idx].sum(axis=1), trials[idx].sum(axis=1)
    ok = n > 0
    draws = h[ok] / n[ok]
    if draws.size == 0:
        return (rate, np.nan, np.nan)
    return (rate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def concealment_detail(rows, items, d_bins=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """phi under both definitions, split by dissent dose and by round.

    One row per (condition, split, bin). ``split`` is "all", "dose" or "round";
    for "all" the row also carries the decomposition of the legacy estimator
    into a real private/public gap and an agent that had simply changed its mind.
    """
    by = {(r["task_id"], r["agent"], r["round"], r["branch"]): r for r in rows}
    out = []
    conditions = sorted({r["branch"] for r in rows if r["branch"] != "r0"})
    for cond in conditions:
        # per-task counters: (hits, trials)
        legacy, strict, gap_of_legacy = defaultdict(lambda: [0, 0]), defaultdict(
            lambda: [0, 0]), defaultdict(lambda: [0, 0])
        by_dose = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        by_round = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in rows:
            if r["branch"] != cond or r["round"] == 0:
                continue
            prev_branch = "r0" if r["round"] == 1 else cond
            prev = by.get((r["task_id"], r["agent"], r["round"] - 1, prev_branch))
            priv, pub = r["private"], r["public"]
            nb_pub = [a for a in (r.get("nb_public") or []) if a]
            M = majority(nb_pub)
            if M is None or priv is None or pub is None:
                continue
            t = r["task_id"]

            # legacy: conditioned on the PREVIOUS private answer
            if prev is not None and prev["private"] is not None:
                if M != prev["private"]:
                    legacy[t][1] += 1
                    if pub == M:
                        legacy[t][0] += 1
                        # of those, how many were a real gap rather than an agent
                        # that had already come round to the majority view?
                        gap_of_legacy[t][1] += 1
                        gap_of_legacy[t][0] += int(pub != priv)

            # strict: conditioned on the answer the agent holds NOW
            if M != priv:
                strict[t][1] += 1
                hit = int(pub == M and pub != priv)
                strict[t][0] += hit
                d = sum(a != priv for a in nb_pub) / len(nb_pub)
                lo = max([b for b in d_bins if b <= d], default=d_bins[0])
                by_dose[lo][t][1] += 1
                by_dose[lo][t][0] += hit
                by_round[r["round"]][t][1] += 1
                by_round[r["round"]][t][0] += hit

        lg = _boot_ci(legacy)
        st = _boot_ci(strict)
        gp = _boot_ci(gap_of_legacy)
        out.append(dict(condition=cond, split="all", bin=np.nan,
                        phi_legacy=lg[0], legacy_lo=lg[1], legacy_hi=lg[2],
                        n_legacy=int(sum(v[1] for v in legacy.values())),
                        phi_strict=st[0], strict_lo=st[1], strict_hi=st[2],
                        n_strict=int(sum(v[1] for v in strict.values())),
                        real_gap_share=gp[0], gap_lo=gp[1], gap_hi=gp[2],
                        n_tasks=len(strict)))
        for d, per_task in sorted(by_dose.items()):
            c = _boot_ci(per_task)
            out.append(dict(condition=cond, split="dose", bin=float(d),
                            phi_strict=c[0], strict_lo=c[1], strict_hi=c[2],
                            n_strict=int(sum(v[1] for v in per_task.values())),
                            n_tasks=len(per_task)))
        for rnd, per_task in sorted(by_round.items()):
            c = _boot_ci(per_task)
            out.append(dict(condition=cond, split="round", bin=float(rnd),
                            phi_strict=c[0], strict_lo=c[1], strict_hi=c[2],
                            n_strict=int(sum(v[1] for v in per_task.values())),
                            n_tasks=len(per_task)))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out-params", required=True)
    ap.add_argument("--out-accuracy", required=True)
    ap.add_argument("--out-manipulation", required=True)
    ap.add_argument("--out-phi", default=None,
                    help="optional: the concealment estimators in detail")
    args = ap.parse_args(argv)

    rows = load(args.events)
    items = load_items(args.items)
    meta_path = Path(args.events).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rounds = int(meta.get("rounds", max(r["round"] for r in rows)))

    params, acc, p_hat = measure(rows, items, rounds)
    manip = manipulation_check(rows, items)
    phi_detail = concealment_detail(rows, items)
    if args.out_phi:
        Path(args.out_phi).parent.mkdir(parents=True, exist_ok=True)
        phi_detail.to_csv(args.out_phi, index=False)
    for path, df in ((args.out_params, params), (args.out_accuracy, acc),
                     (args.out_manipulation, manip)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    print("manipulation check:")
    print(manip.round(3).to_string(index=False))
    cols = ["condition", "phi_hat", "lambda_hat", "r_hat", "share_rate",
            "disclose_truth_rate", "n_contradicted", "n_share_slots"]
    print(params[cols].round(3).to_string(index=False))
    print(acc.round(3).to_string(index=False))
    print("\nconcealment, legacy vs strict (cluster bootstrap over items):")
    cols = ["condition", "phi_legacy", "legacy_lo", "legacy_hi", "phi_strict",
            "strict_lo", "strict_hi", "real_gap_share", "n_strict", "n_tasks"]
    print(phi_detail[phi_detail["split"] == "all"][cols].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
