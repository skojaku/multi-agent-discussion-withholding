"""Turn an LLM debate transcript into the four CBRM parameters, then test the theory.

Every parameter of the model is an observed frequency in the transcript, so the
comparison has nothing left to fit:

    p_hat        round-0 accuracy per item, measured before any interaction
    phi_hat      P(public = neighbour majority | majority contradicts private)
    lambda_hat   P(private_{t+1} = public_t | concealed at t)
    (rho r)_hat  P(private_{t+1} correct | disclosed at t and dissent was visible)

Outputs
-------
llm_params.csv    one row per framing condition: the measured parameters, with
                  Wilson intervals and the denominators they rest on.
llm_accuracy.csv  observed collective accuracy per condition, split by whether
                  the item started with a wrong majority (p_hat < 1/2) -- the
                  only items where interaction has anything to add.
llm_vs_theory.csv the predicted accuracy curve against phi, simulated at the
                  MEASURED lambda and rho*r and at the run's own N, T and
                  topology, with each condition placed on it at its own phi_hat.

Binarisation caveat: the model is binary, the task is a 4-10 way MCQ. Items are
scored correct/incorrect and the collective answer is the plurality, so the
theory's "majority" is being read as "plurality". The script reports how
concentrated the wrong answers are on a single distractor, because that is what
decides whether the binary reading is fair on this item set.

Two projections of the transcript onto the model, reported side by side
----------------------------------------------------------------------
``letter``  (columns without a suffix) reads the model's states as the literal
            answer letters: "the majority contradicts me" means the modal
            neighbour letter differs from mine, and "dissent" is any neighbour
            saying a different letter. This is what a reader means by
            disagreement, but it is NOT the model's state space: two agents
            holding two different wrong letters are in the same CBRM state
            (c = 0) while this projection scores them as dissenting, which
            inflates the repair denominator and pushes ``r_hat`` down.
``binary``  (columns suffixed ``_bin``) projects every answer onto the model's
            actual state, correct/incorrect, before any statistic is taken.
            Dissent is then disagreement about *the truth*, which is the only
            disagreement the repair channel can act on. The theory curve is
            driven by these, because they are the quantities the equations use.

On a 10-way item set the two can differ a lot, so both are written out and the
gap between them is itself a reported number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))
from cbrm_theory import Params, simulate  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (ph, max(0.0, c - h), min(1.0, c + h))


def load(events_path):
    """Rows keyed on (task_id, agent, round, branch), last write wins.

    The log is append-only and a killed job can leave the same key on disk twice
    (the resumed process dedupes what it reads, but the two files are already
    concatenated). Counting a duplicated round-0 row twice would bias p_hat, so
    the de-duplication happens here rather than being assumed upstream.
    """
    rows = {}
    for line in Path(events_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            # Transcripts written before the seat -> agent rename carry "seat".
            # They are the only copy of several thousand dollars of model calls,
            # so they are read, not regenerated.
            if "agent" not in r and "seat" in r:
                r["agent"] = r["seat"]
            rows[(r["task_id"], r["agent"], r["round"], r["branch"])] = r
    return list(rows.values())


def majority(labels):
    """Plurality label; None on a tie or an empty list."""
    labels = [x for x in labels if x]
    if not labels:
        return None
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def _bit_majority(bits):
    """Majority of correct/incorrect bits; None on a tie or an empty panel."""
    bits = [b for b in bits if b is not None]
    if not bits:
        return None
    n1 = sum(bits)
    if 2 * n1 == len(bits):
        return None
    return int(2 * n1 > len(bits))


def measure(rows, rounds):
    by = {(r["task_id"], r["agent"], r["round"], r["branch"]): r for r in rows}
    answer = {r["task_id"]: r["answer"] for r in rows}

    # p_hat per item from round 0.
    r0 = defaultdict(list)
    for r in rows:
        if r["round"] == 0:
            r0[r["task_id"]].append(r["private"] == r["answer"])
    p_hat = {t: float(np.mean(v)) for t, v in r0.items() if v}

    conditions = sorted({r["branch"] for r in rows if r["branch"] != "r0"})
    param_rows, acc_rows = [], []
    for cond in conditions:
        n_contra = n_conceal = 0
        n_conc_pairs = n_internalise = 0
        # Repair is a moment estimator, not a frequency. An agent that sees
        # dissent engages only with probability rho*d, where d is the fraction
        # of its neighbours expressing something other than its belief; if it
        # does not engage it keeps the belief it had. So P(flip) = rho*d*r for a
        # wrong agent and rho*d*(1-r) for a correct one, and dividing the flips
        # by the summed d recovers rho*r and rho*(1-r) -- whereas the raw flip
        # frequency would be pulled towards the base rate by the agents that
        # never engaged at all.
        d_wrong = d_right = 0.0
        flip_up = flip_down = 0
        n_repair_pairs = 0
        # Same four quantities under the binary projection: every answer is
        # replaced by whether it is the truth before anything is counted.
        b_contra = b_conceal = b_conc_pairs = b_internalise = 0
        b_d_wrong = b_d_right = 0.0
        b_flip_up = b_flip_down = 0
        b_repair_pairs = 0
        # The CBRM has exactly one way for a wrong agent to become right: it must
        # SEE a neighbour expressing the truth (the repair channel is gated on
        # visible dissent, and in a two-state model dissent from a wrong belief
        # is by definition the truth). So the rate below is a number the model
        # says is zero. Measuring it is a direct test of the model's structure,
        # not of its parameters: an LLM re-derives from the question itself, and
        # two teammates disagreeing on two different WRONG letters is a prompt to
        # think again that a binary model cannot represent.
        n_no_correct_visible = n_spontaneous = 0
        n_no_correct_dissent = 0
        for r in rows:
            if r["branch"] != cond or r["round"] == 0:
                continue
            # The belief that faced the majority is the one the agent held BEFORE
            # this round, not the one it reports after having read the room:
            # PRIVATE_t already carries whatever repair or internalisation the
            # round produced, so using it to define "was this agent contradicted"
            # scores an agent that changed its mind as having concealed.
            prev_branch = "r0" if r["round"] == 1 else cond
            prev = by.get((r["task_id"], r["agent"], r["round"] - 1, prev_branch))
            if prev is None or prev["private"] is None:
                continue
            c_prev = prev["private"]
            nb_pub = r.get("nb_public") or []
            # A letter-level tie leaves the plurality undefined, but the same
            # panel can still have a well defined majority once the answers are
            # projected onto correct/incorrect, so the two projections are gated
            # separately rather than sharing one skip.
            M = majority(nb_pub)
            priv, pub = r["private"], r["public"]
            if priv is None or pub is None or not nb_pub:
                continue
            seen = [a for a in nb_pub if a]
            if not seen:
                continue
            d = sum(a != c_prev for a in seen) / len(seen)
            contradicted = M is not None and M != c_prev

            # ---- structural test: repair with no correct answer in sight
            truth = answer[r["task_id"]]
            if c_prev != truth and all(a != truth for a in seen):
                n_no_correct_visible += 1
                n_spontaneous += int(priv == truth)
                if len(set(seen)) > 1 or any(a != c_prev for a in seen):
                    n_no_correct_dissent += 1

            # ---- binary projection: states are correct/incorrect, not letters
            c_bit, x_bit, priv_bit = int(c_prev == truth), int(pub == truth), int(priv == truth)
            nb_bits = [int(a == truth) for a in seen]
            M_bit = _bit_majority(nb_bits)
            if M_bit is not None:
                d_bit = sum(b != c_bit for b in nb_bits) / len(nb_bits)
                b_contradicted = M_bit != c_bit
                if b_contradicted:
                    b_contra += 1
                b_concealed = b_contradicted and (x_bit == M_bit)
                if b_concealed:
                    b_conceal += 1
                    b_conc_pairs += 1
                    b_internalise += int(priv_bit == x_bit)
                elif d_bit > 0:
                    b_repair_pairs += 1
                    if c_bit == 0:
                        b_d_wrong += d_bit
                        b_flip_up += int(priv_bit == 1)
                    else:
                        b_d_right += d_bit
                        b_flip_down += int(priv_bit == 0)

            if M is None:
                continue
            if contradicted:
                n_contra += 1
            concealed = contradicted and (pub == M)
            if concealed:
                n_conceal += 1
                n_conc_pairs += 1
                n_internalise += int(priv == pub)
            elif d > 0:
                n_repair_pairs += 1
                correct = answer[r["task_id"]]
                if c_prev != correct:
                    d_wrong += d
                    flip_up += int(priv == correct)
                else:
                    d_right += d
                    flip_down += int(priv != correct)
        phi = wilson(n_conceal, n_contra)
        lam = wilson(n_internalise, n_conc_pairs)
        b_phi = wilson(b_conceal, b_contra)
        b_lam = wilson(b_internalise, b_conc_pairs)
        b_rr = wilson(b_flip_up, int(round(b_d_wrong)))
        b_bad = wilson(b_flip_down, int(round(b_d_right)))
        b_rho = b_rr[0] + b_bad[0]
        b_r = b_rr[0] / b_rho if b_rho > 0 else np.nan
        spont = wilson(n_spontaneous, n_no_correct_visible)
        # Wilson on the flip count with the summed d as the effective number of
        # trials: an approximation, exact only if every d were equal.
        rr = wilson(flip_up, int(round(d_wrong)))
        r_bad = wilson(flip_down, int(round(d_right)))
        rho_hat = rr[0] + r_bad[0]
        r_hat = rr[0] / rho_hat if rho_hat > 0 else np.nan
        param_rows.append(dict(
            condition=cond,
            phi_hat=phi[0], phi_lo=phi[1], phi_hi=phi[2], n_contradicted=n_contra,
            lambda_hat=lam[0], lambda_lo=lam[1], lambda_hi=lam[2],
            n_concealed_pairs=n_conc_pairs,
            rho_r_hat=rr[0], rho_r_lo=rr[1], rho_r_hi=rr[2],
            rho_hat=min(rho_hat, 1.0), r_hat=r_hat,
            n_repair_pairs=n_repair_pairs,
            phi_bin_hat=b_phi[0], phi_bin_lo=b_phi[1], phi_bin_hi=b_phi[2],
            n_contradicted_bin=b_contra,
            lambda_bin_hat=b_lam[0], lambda_bin_lo=b_lam[1], lambda_bin_hi=b_lam[2],
            n_concealed_pairs_bin=b_conc_pairs,
            rho_r_bin_hat=b_rr[0], rho_r_bin_lo=b_rr[1], rho_r_bin_hi=b_rr[2],
            rho_bin_hat=min(b_rho, 1.0), r_bin_hat=b_r,
            n_repair_pairs_bin=b_repair_pairs,
            spontaneous_repair=spont[0], spontaneous_lo=spont[1],
            spontaneous_hi=spont[2], n_no_correct_visible=n_no_correct_visible,
            frac_wrong_wrong_dissent=(n_no_correct_dissent / n_no_correct_visible
                                      if n_no_correct_visible else np.nan)))

        # Collective accuracy at the final round, split by how the item started.
        final = defaultdict(list)
        # The private-expressed gap is a per-item quantity, so it is accumulated
        # per item and averaged inside each stratum: a gap taken over the whole
        # condition is dominated by the easy items, where there is nothing to
        # hide, and says little about the hard ones the theory is about.
        priv_by_item, pub_by_item = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["branch"] != cond:
                continue
            if r["round"] == rounds:
                final[r["task_id"]].append(r["public"])
            if r["round"] >= 1:
                if r["private"]:
                    priv_by_item[r["task_id"]].append(r["private"] == r["answer"])
                if r["public"]:
                    pub_by_item[r["task_id"]].append(r["public"] == r["answer"])
        # ``recoverable`` is the theory's wedge and the only stratum where the
        # model predicts anything non-trivial: the majority is wrong at the start
        # but at least one agent holds the truth, so repair has something to work
        # with. ``dead`` is p_hat = 0 exactly -- nobody in the team has the
        # answer, no channel in the model can invent it, and the predicted
        # accuracy is 0 at every phi. Splitting them matters because a
        # homogeneous LLM team puts most of its hard items in ``dead``.
        # Accuracy round by round, not only at the end: round 0 is the
        # independent-vote baseline (no interaction has happened yet), so the
        # difference between round 0 and round T is exactly what the deliberation
        # bought -- or cost -- under this framing.
        by_round = {}
        for t in range(0, rounds + 1):
            branch = "r0" if t == 0 else cond
            pub = defaultdict(list)
            for r in rows:
                if r["branch"] == branch and r["round"] == t:
                    pub[r["task_id"]].append(r["public"])
            by_round[t] = pub

        for stratum, keep in (("hard", lambda t: p_hat.get(t, 1.0) < 0.5),
                              ("recoverable", lambda t: 0.0 < p_hat.get(t, 1.0) < 0.5),
                              ("dead", lambda t: p_hat.get(t, 1.0) == 0.0),
                              ("easy", lambda t: p_hat.get(t, 0.0) >= 0.5),
                              ("all", lambda t: True)):
            items = [t for t in final if keep(t)]
            hits = [majority(final[t]) == answer[t] for t in items]
            a = wilson(int(np.sum(hits)), len(hits))
            gaps = [np.mean(priv_by_item[t]) - np.mean(pub_by_item[t])
                    for t in items if priv_by_item[t] and pub_by_item[t]]
            row = dict(condition=cond, stratum=stratum, round=rounds,
                       n_items=len(hits), accuracy=a[0], acc_lo=a[1], acc_hi=a[2],
                       mean_p_hat=float(np.mean([p_hat[t] for t in items]))
                       if items else np.nan,
                       gap=float(np.mean(gaps)) if gaps else np.nan)
            for t, pub in by_round.items():
                sel = [x for x in items if pub.get(x)]
                if sel:
                    row[f"accuracy_round{t}"] = float(np.mean(
                        [majority(pub[x]) == answer[x] for x in sel]))
            acc_rows.append(row)
    return pd.DataFrame(param_rows), pd.DataFrame(acc_rows), p_hat


def distractor_concentration(rows):
    """Share of wrong round-0 answers that land on the single modal distractor.

    High concentration is what makes the binary reading fair: the wrong side
    behaves like one bloc, which is what a two-state model assumes.
    """
    per_item = []
    wrong = defaultdict(list)
    for r in rows:
        if r["round"] == 0 and r["private"] and r["private"] != r["answer"]:
            wrong[r["task_id"]].append(r["private"])
    for t, ws in wrong.items():
        if ws:
            per_item.append(Counter(ws).most_common(1)[0][1] / len(ws))
    return float(np.mean(per_item)) if per_item else np.nan


def theory_curve(params, p_hard, N, T, topology, q=3, n_reps=4000, n_phi=41):
    """Predicted accuracy against phi at the measured lambda and rho*r."""
    # The binary projection is the model's own state space, so the prediction is
    # driven by it; the letter-level numbers stay in the table for comparison.
    lam = float(np.nanmean(params["lambda_bin_hat"]))
    rho = float(np.nanmean(params["rho_bin_hat"]))
    r = float(np.nanmean(params["r_bin_hat"]))
    lam = 0.5 if not np.isfinite(lam) else float(np.clip(lam, 0, 1))
    rho = 1.0 if not np.isfinite(rho) else float(np.clip(rho, 0, 1))
    r = 0.8 if not np.isfinite(r) else float(np.clip(r, 0, 1))
    topo = topology if topology in ("complete", "ring", "star") else "panel"
    out = []
    for phi in np.linspace(0, 1, n_phi):
        prm = Params(phi=float(phi), lam=lam, rho=rho, r=r, q=q,
                     trigger="fraction")
        s = simulate(N, p_hard, prm, topo, n_rounds=T, n_reps=n_reps, seed=7)
        out.append(dict(phi=float(phi), predicted_accuracy=s["accuracy"],
                        predicted_gap=s["gap"], lam_used=lam, rho_used=rho,
                        r_used=r, N=N, T=T, topology=topo, p_hard=p_hard))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out-params", required=True)
    ap.add_argument("--out-accuracy", required=True)
    ap.add_argument("--out-theory", required=True)
    args = ap.parse_args(argv)

    rows = load(args.events)
    meta_path = Path(args.events).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rounds = int(meta.get("rounds", max(r["round"] for r in rows)))
    agents = int(meta.get("agents", 1 + max(r["agent"] for r in rows)))
    topology = meta.get("topology", "complete")

    params, acc, p_hat = measure(rows, rounds)
    hard = [t for t, v in p_hat.items() if v < 0.5]
    p_hard = float(np.mean([p_hat[t] for t in hard])) if hard else 0.35
    recoverable = [t for t, v in p_hat.items() if 0.0 < v < 0.5]
    theory = theory_curve(params, p_hard, agents, rounds, topology,
                          q=min(agents - 1, 3) | 1)

    for path, df in ((args.out_params, params), (args.out_accuracy, acc),
                     (args.out_theory, theory)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    print(f"items: {len(p_hat)}  hard (p_hat < 1/2): {len(hard)}  "
          f"recoverable (0 < p_hat < 1/2): {len(recoverable)}  "
          f"mean p_hat on hard: {p_hard:.3f}")
    print(f"modal-distractor concentration: {distractor_concentration(rows):.3f}")
    cols = ["condition", "phi_hat", "phi_bin_hat", "lambda_hat", "lambda_bin_hat",
            "rho_r_hat", "rho_r_bin_hat", "r_hat", "r_bin_hat",
            "n_contradicted", "n_contradicted_bin", "n_repair_pairs_bin",
            "spontaneous_repair", "n_no_correct_visible"]
    print(params[cols].to_string(index=False))
    print(acc[acc['stratum'] == 'hard'].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
