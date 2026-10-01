"""X2: the same shape of measurement, one level down -- over evidence, not opinions.

The opinion-level rates answer "did the agent say what it believed". They stop
working exactly where the interesting arms are: a protocol that forbids opinion
exchange and only lets agents trade evidence has no expressed majority to
conceal from, so ``c`` has an empty denominator. Across the transcripts in this
repo that is 39 of 144 (run, condition) cells -- every ``evidence_only`` arm and
every open-ended task.

The evidence level restores it. The estimand is

    c_ev = P(never disclosed a decisive finding within T rounds
             | held one, and the board on the table pointed the other way)

Note what the conditioning event is *not*. Reading "disagreed with the majority"
off the expressed opinions would reimport the very dependency that makes X1
unusable here -- in an evidence-only arm nobody expresses an opinion, so that
condition never fires and the denominator is zero again. The board this
conditions on is the disclosed evidence itself: the shared briefing plus
everything anyone has put on the table so far. An agent holding a finding that
favours the truth, looking at a board that currently favours something else, and
saying nothing, is the event the estimator is named after, and it is observable
whether or not the protocol lets anyone state an opinion. The opinion-conditioned
version is reported alongside as ``c_ev_opinion`` for comparison with X1.

and it decomposes the outcome into the two rates that actually gate it:

    D  disclosure   P(what reached the table was enough to decide correctly)
    U  usage        P(the team decided correctly | it was enough)

Collective accuracy is close to ``D * U`` on these tasks, and which of the two
is small is the design-relevant answer. A strong model and a weak model can land
on the same accuracy through opposite failures: the weak one puts the evidence
on the table and cannot use it (high D, low U); the strong one reads the table
perfectly and only tables what supports its current answer (low D, high U).

``c_ev`` depends on T -- evidence, once disclosed, stays disclosed -- so T is
carried on every result and two runs with different T are not comparable.

What a task has to supply
-------------------------
Which findings exist, who holds them, and which answer each one supports. That
last part is task knowledge, so it lives behind an adapter rather than in the
estimator. When an item set does not tag its findings (HiddenBench states clues
in natural language and never says which option each one favours), the tagged
quantities are reported as unmeasurable and the untagged ones -- share rate,
pooling coverage, silence under disagreement -- still come out.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import BASELINE_BRANCH, majority


@dataclass(frozen=True)
class Piece:
    """One finding. ``favours`` is None when the item set does not tag them."""

    id: str
    favours: str | None = None


class EvidenceAdapter:
    """Task knowledge, kept out of the estimators.

    Four methods. Implement them for a new task family and every number in this
    module comes out; the estimators themselves never learn what a "finding" is.
    """

    tagged: bool = True

    def held(self, task_id, agent) -> list[Piece]:
        """The private findings this agent starts with, in disclosure order."""
        raise NotImplementedError

    def pool0(self, task_id) -> list[Piece]:
        """Findings every agent can see from the start (the shared briefing)."""
        raise NotImplementedError

    def resolve(self, task_id, agent, row) -> Piece | None:
        """Which finding an event row disclosed, if any."""
        raise NotImplementedError

    def decides(self, task_id, tally: Counter) -> str | None:
        """The answer a pool of findings implies. None if it decides nothing."""
        raise NotImplementedError


class FactTallyAdapter(EvidenceAdapter):
    """The ``{shared_facts, agent_facts: [{about, text}]}`` item schema.

    Used by the hidden-profile generator and by the HiddenBench conversion. When
    ``about`` is filled the findings are additive and the pool decides by
    plurality; when it is None (HiddenBench: free-text clues that eliminate
    options rather than score them) the adapter reports itself as untagged and
    the sufficiency-based quantities are withheld rather than guessed.
    """

    def __init__(self, items: dict):
        self.items = {k: self._normalise(v) for k, v in items.items()}
        self.tagged = any(
            f.get("about")
            for it in self.items.values()
            for f in it["shared_facts"] + [g for v in it["agent_facts"].values()
                                           for g in v])

    @staticmethod
    def _normalise(item):
        # Items generated before the seat -> agent rename use the old key names.
        it = dict(item)
        it.setdefault("agent_facts", it.get("seat_facts", {}))
        it.setdefault("shared_facts", [])
        d = dict(it.get("design", {}))
        for old, new in (("n_seats", "n_agents"),
                         ("per_seat_tally", "per_agent_tally"),
                         ("solo_correct_seats", "solo_correct_agents")):
            if new not in d and old in d:
                d[new] = d[old]
        it["design"] = d
        return it

    def held(self, task_id, agent):
        facts = self.items[task_id]["agent_facts"].get(str(agent), [])
        return [Piece(id=f"{agent}:{i + 1}", favours=f.get("about"))
                for i, f in enumerate(facts)]

    def pool0(self, task_id):
        return [Piece(id=f"S:{i}", favours=f.get("about"))
                for i, f in enumerate(self.items[task_id]["shared_facts"])]

    def resolve(self, task_id, agent, row):
        s = row.get("share")
        if not s:
            return None
        held = self.held(task_id, agent)
        # ``share`` is the 1-based index the protocol prints as "P<k>";
        # ``share_about`` is the runner's own resolution and wins when present.
        piece = None
        if isinstance(s, int) and 1 <= s <= len(held):
            piece = held[s - 1]
        if piece is None:
            piece = Piece(id=f"{agent}:{s}", favours=None)
        about = row.get("share_about")
        return Piece(id=piece.id, favours=about if about else piece.favours)

    def decides(self, task_id, tally):
        if not self.tagged:
            return None
        labels = [lab for lab in tally if lab]
        if not labels:
            return None
        top = sorted(labels, key=lambda lab: -tally[lab])
        if len(top) > 1 and tally[top[0]] == tally[top[1]]:
            return None
        return top[0]


# ------------------------------------------------------------------ estimators
def evidence_rates(t, condition: str, adapter: EvidenceAdapter,
                   baseline_branch: str = BASELINE_BRANCH) -> dict:
    """c_ev, the disclosure/usage split, and the selectivity of what was shared."""
    truth = t.truth()
    T = t.rounds
    rows = [r for r in t.rows if r["branch"] == condition and r["round"] >= 1]
    if not rows:
        return dict(T=T, n_tasks=0, n_slots=0, note="no rows in this condition")

    tasks = sorted({r["task_id"] for r in rows})
    held, shared = {}, defaultdict(list)
    dissented = defaultdict(bool)          # the room's opinions contradict mine
    n_slots = n_shared = n_shared_truth = n_self_serving = 0
    n_tagged_shares = 0
    # Disclosures in the order they happened, so the board can be replayed round
    # by round: what the board said when an agent chose to speak or not is the
    # conditioning event, and it is not the board at the end.
    timeline = defaultdict(list)           # task -> [(round, agent, Piece)]

    for r in rows:
        key = (r["task_id"], r["agent"])
        if key not in held:
            try:
                held[key] = adapter.held(r["task_id"], r["agent"])
            except KeyError:
                held[key] = []
        n_slots += 1
        piece = adapter.resolve(r["task_id"], r["agent"], r)
        if piece is not None:
            n_shared += 1
            shared[key].append(piece)
            timeline[r["task_id"]].append((int(r["round"]), r["agent"], piece))
            if piece.favours is not None:
                n_tagged_shares += 1
                n_shared_truth += int(piece.favours == truth.get(r["task_id"]))
                n_self_serving += int(piece.favours == r.get("private"))
        M = majority(r.get("nb_public") or [])
        if M is not None and r.get("private") and M != r["private"]:
            dissented[key] = True

    # -- c_ev: held something decisive, the board pointed the other way, stayed
    #    quiet. The board is replayed round by round because an agent that
    #    discloses in round 3 was still silent in rounds 1 and 2, and because the
    #    board it faced then is not the board at the end.
    n_dec = n_dec_silent = 0
    n_dec_op = n_dec_op_silent = 0
    if adapter.tagged:
        board_against = defaultdict(bool)
        for task in tasks:
            tr = truth.get(task)
            tally = Counter(p.favours for p in adapter.pool0(task) if p.favours)
            seen_ids = set()
            events = sorted(timeline[task], key=lambda e: e[0])
            holders = [key for key in held if key[0] == task]
            for rnd in range(1, T + 1):
                # Anyone still holding a decisive finding faces the board as it
                # stood at the start of this round.
                if adapter.decides(task, tally) != tr:
                    for key in holders:
                        board_against[key] = True
                for r_, ag, p in [e for e in events if e[0] == rnd]:
                    if p.favours and p.id not in seen_ids:
                        seen_ids.add(p.id)
                        tally[p.favours] += 1
        for key, pieces in held.items():
            tr = truth.get(key[0])
            if not any(p.favours == tr for p in pieces):
                continue
            silent = not any(p.favours == tr for p in shared[key])
            if board_against[key]:
                n_dec += 1
                n_dec_silent += int(silent)
            if dissented[key]:
                n_dec_op += 1
                n_dec_op_silent += int(silent)
    # Untagged fallback: held anything, disagreed, disclosed nothing at all.
    n_any = n_any_silent = 0
    for key, pieces in held.items():
        if not pieces or not dissented[key]:
            continue
        n_any += 1
        n_any_silent += int(not shared[key])

    # -- D and U. The pool is the shared briefing plus every disclosure, which is
    #    exactly what every agent could see, so sufficiency is a property of the
    #    transcript rather than of the model.
    D = U = pooled_fraction = np.nan
    n_suff = 0
    if adapter.tagged:
        final_pub = defaultdict(list)
        for r in rows:
            if r["round"] == T and r.get("public"):
                final_pub[r["task_id"]].append(r["public"])
        by_task = defaultdict(list)
        for (tk, ag), pieces in shared.items():
            by_task[tk].extend(pieces)
        held_by_task = defaultdict(list)
        for (tk, ag), pieces in held.items():
            held_by_task[tk].extend(pieces)

        suff, hit, cover = [], [], []
        for task in tasks:
            tr = truth.get(task)
            tally = Counter(p.favours for p in adapter.pool0(task) if p.favours)
            for p in {p.id: p for p in by_task[task]}.values():
                if p.favours:
                    tally[p.favours] += 1
            ok = adapter.decides(task, tally) == tr
            suff.append(ok)
            if final_pub[task]:
                hit.append((majority(final_pub[task]) == tr, ok))
            total = sum(1 for p in held_by_task[task] if p.favours == tr)
            got = len({p.id for p in by_task[task] if p.favours == tr})
            cover.append(got / total if total else np.nan)
        n_suff = int(np.sum(suff))
        D = float(np.mean(suff)) if suff else np.nan
        num = [h for h, ok in hit if ok]
        U = float(np.mean(num)) if num else np.nan
        with np.errstate(invalid="ignore"):
            pooled_fraction = float(np.nanmean(cover)) if cover else np.nan

    def rate(k, n):
        return float(k) / n if n else np.nan

    return dict(
        T=T, n_tasks=len(tasks), n_slots=n_slots,
        share_rate=rate(n_shared, n_slots),
        share_truth_rate=rate(n_shared_truth, n_tagged_shares),
        self_serving_rate=rate(n_self_serving, n_tagged_shares),
        c_ev=rate(n_dec_silent, n_dec), n_c_ev=n_dec,
        c_ev_opinion=rate(n_dec_op_silent, n_dec_op), n_c_ev_opinion=n_dec_op,
        c_ev_any=rate(n_any_silent, n_any), n_c_ev_any=n_any,
        D=D, U=U, n_sufficient=n_suff, pooled_fraction=pooled_fraction,
        tagged=bool(adapter.tagged))


def limiting_factor(D, U) -> str:
    """Which of the two gates is the brake. Both must be measured to answer."""
    if not (np.isfinite(D) and np.isfinite(U)):
        return "unknown"
    if abs(D - U) < 0.05:
        return "balanced"
    return "disclosure (D)" if D < U else "usage (U)"


def evidence_frame(t, adapter, conditions=None) -> pd.DataFrame:
    out = []
    for c in (conditions or t.conditions):
        row = dict(condition=c, **evidence_rates(t, c, adapter))
        row["limiting_factor"] = limiting_factor(row.get("D", np.nan),
                                                 row.get("U", np.nan))
        out.append(row)
    return pd.DataFrame(out)
