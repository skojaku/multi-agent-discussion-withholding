"""Transcripts in, one normalised decision table out.

Everything downstream counts rows in the table this module builds, so all the
awkward parts of a real log — legacy field names, a killed job that appended the
same key twice, round 0 living in its own branch — are handled once, here.

The central object is not the event row but the *decision*: one (task, agent,
round) at which an agent held a belief, saw some neighbours, and chose what to
say. A decision carries five things the estimators need and nothing else:

    c_prev   the belief the agent held ENTERING the round -- the one that faced
             the majority. Using the belief it reports afterwards instead scores
             an agent that was talked round as having concealed.
    priv     the belief it holds after the round
    pub      what it expressed
    M        the majority of what it saw (None if tied)
    d        the fraction of what it saw that disagreed with c_prev

Everything else is arithmetic on those columns.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Transcripts written before the seat -> agent rename carry "seat". They are the
# only copy of several thousand dollars of model calls, so they are read as they
# are rather than rewritten.
_ROW_ALIASES = {"seat": "agent"}

BASELINE_BRANCH = "r0"


# --------------------------------------------------------------------- loading
@dataclass
class Transcript:
    """A de-duplicated event log plus whatever the run recorded about itself."""

    rows: list[dict]
    meta: dict = field(default_factory=dict)
    source: str = ""

    # -- things a caller should not have to recompute from rows
    @property
    def conditions(self) -> list[str]:
        return sorted({r["branch"] for r in self.rows
                       if r["branch"] != BASELINE_BRANCH})

    @property
    def rounds(self) -> int:
        """T, from the run's own metadata when it recorded it."""
        if "rounds" in self.meta:
            return int(self.meta["rounds"])
        return int(max(r["round"] for r in self.rows))

    @property
    def n_agents(self) -> int:
        return len({r["agent"] for r in self.rows})

    @property
    def n_tasks(self) -> int:
        return len({r["task_id"] for r in self.rows})

    @property
    def labels(self) -> list[str]:
        """Distinct answers seen. len() > 2 is the projection warning."""
        return sorted({r["private"] for r in self.rows if r.get("private")}
                      | {r["public"] for r in self.rows if r.get("public")})

    @property
    def has_truth(self) -> bool:
        return any(r.get("answer") for r in self.rows)

    @property
    def has_evidence(self) -> bool:
        return any(r.get("share") for r in self.rows)

    @property
    def probe(self) -> str:
        """"joint" = one call returned private and public; "separate" = two.

        Worth carrying around because the split between concealment and
        internalisation is known to depend on it: asking for both answers in one
        call reads as concealment what two calls read as a changed mind.
        """
        return str(self.meta.get("probe", "unknown"))

    def truth(self) -> dict:
        return {r["task_id"]: r["answer"] for r in self.rows if r.get("answer")}


def load_events(path, meta=None) -> Transcript:
    """Read an events.jsonl. Sidecar events.meta.json is picked up if present.

    The log is append-only and a killed job can leave the same key on disk twice
    (the resumed process dedupes what it reads, but the two files are already
    concatenated). Counting a duplicated round-0 row twice would bias the
    baseline, so de-duplication happens here rather than being assumed upstream.
    """
    path = Path(path)
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for old, new in _ROW_ALIASES.items():
            if new not in r and old in r:
                r[new] = r[old]
        rows[(r["task_id"], r["agent"], r["round"], r["branch"])] = r
    if meta is None:
        side = path.with_suffix(".meta.json")
        meta = json.loads(side.read_text()) if side.exists() else {}
    return Transcript(rows=list(rows.values()), meta=dict(meta), source=str(path))


def load_items(path) -> dict:
    """Item definitions keyed by id, for the evidence-level estimators."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            it = json.loads(line)
            out[it["id"]] = it
    return out


# ----------------------------------------------------------------- projections
def majority(labels):
    """Plurality label; None on a tie or an empty list."""
    labels = [x for x in labels if x is not None and x != ""]
    if not labels:
        return None
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


PROJECTIONS = ("plurality", "binary", "letter")


def _project(label, truth, projection):
    """Map a raw answer onto the state space the estimator counts in.

    ``plurality`` (the default, and ``letter`` is an accepted alias) keeps the
    raw answer. On K options the model's "visible majority" is the most common
    answer among the expressions an agent can see, with no majority on a tie, and
    concealment is conceding to *that specific answer*. So this is not a
    convenience reading -- it is the K-choice rule, and on two options it
    coincides exactly with ``binary``.

    ``binary`` replaces every answer by whether it is correct before anything is
    counted. That is the two-choice reduction, and it is right only when the
    wrong side is concentrated on one distractor: three agents holding three
    different wrong answers become a unanimous "F" bloc that no agent would ever
    concede to, so a contradiction is scored where the model sees none.
    Measured against transcripts generated by the K-choice rules, ``plurality``
    recovers a true c = 0.600 to within 0.003 at every K and every concentration
    of the wrong side, while ``binary`` reads 0.441 at K = 10 with the errors
    spread -- and carries c* from 0.532 to 0.717, which is enough to flip the
    verdict.

    The one case that inverts this is an effectively unbounded answer set (free
    text, where every neighbour says something different). There the plurality is
    undefined on almost every row and ``binary`` is the only reading with a
    denominator at all -- a two-state reduction whose bias has to be stated
    rather than a measurement.
    """
    if label is None:
        return None
    if projection in ("plurality", "letter"):
        return label
    if projection == "binary":
        if truth is None:
            return None
        return "T" if label == truth else "F"
    raise ValueError(f"unknown projection {projection!r}; expected one of "
                     f"{PROJECTIONS}")


# ------------------------------------------------------------ decision table
def decisions(t: Transcript, condition: str, projection: str = "plurality",
              baseline_branch: str = BASELINE_BRANCH) -> pd.DataFrame:
    """One row per (task, agent, round) at which the agent could have concealed.

    Rounds whose predecessor is missing, whose answers failed to parse, or where
    the agent saw nobody are dropped -- they carry no information about any of
    the four rates and keeping them would only inflate denominators.
    """
    by = {(r["task_id"], r["agent"], r["round"], r["branch"]): r for r in t.rows}
    truth = t.truth()
    out = []
    for r in t.rows:
        if r["branch"] != condition or r["round"] == 0:
            continue
        # Round 1's predecessor is the independent baseline, which lives in its
        # own branch; later rounds chain within the condition.
        prev = (by.get((r["task_id"], r["agent"], r["round"] - 1, condition))
                or by.get((r["task_id"], r["agent"], r["round"] - 1,
                           baseline_branch)))
        if prev is None:
            continue
        tr = truth.get(r["task_id"])
        c_prev = _project(prev.get("private"), tr, projection)
        priv = _project(r.get("private"), tr, projection)
        pub = _project(r.get("public"), tr, projection)
        seen = [_project(a, tr, projection) for a in (r.get("nb_public") or [])]
        seen = [a for a in seen if a is not None]
        if c_prev is None or priv is None or pub is None or not seen:
            continue
        M = majority(seen)
        out.append(dict(
            task_id=r["task_id"], agent=r["agent"], round=int(r["round"]),
            c_prev=c_prev, priv=priv, pub=pub, M=M,
            n_seen=len(seen), n_dissent=sum(a != c_prev for a in seen),
            d=sum(a != c_prev for a in seen) / len(seen),
            truth=_project(tr, tr, projection) if tr is not None else None,
            was_correct=(None if tr is None else prev.get("private") == tr),
            now_correct=(None if tr is None else r.get("private") == tr),
            shared=bool(r.get("share")),
            share_about=r.get("share_about"),
        ))
    df = pd.DataFrame(out)
    if df.empty:
        # A typed empty frame keeps every downstream .loc / groupby working.
        return pd.DataFrame(columns=[
            "task_id", "agent", "round", "c_prev", "priv", "pub", "M", "n_seen",
            "n_dissent", "d", "truth", "was_correct", "now_correct",
            "shared", "share_about"])
    return df


def round0(t: Transcript, baseline_branch: str = BASELINE_BRANCH) -> pd.DataFrame:
    """Per-task independent competence p, measured before any interaction."""
    truth = t.truth()
    out = []
    for r in t.rows:
        if r["round"] != 0 or not r.get("private"):
            continue
        tr = truth.get(r["task_id"])
        out.append(dict(task_id=r["task_id"], agent=r["agent"],
                        private=r["private"], public=r.get("public"),
                        correct=(None if tr is None else r["private"] == tr)))
    df = pd.DataFrame(out)
    if df.empty or df["correct"].isna().all():
        return df
    return df


def p_hat(t: Transcript) -> pd.Series:
    """p per task: the fraction of agents that were right on their own."""
    df = round0(t)
    if df.empty or "correct" not in df:
        return pd.Series(dtype=float)
    return df.groupby("task_id")["correct"].mean()


def error_concentration(t: Transcript, round_no: int = 0) -> pd.Series:
    """Per task: of the agents that are wrong, what share hold the same answer?

    The second initial condition K > 2 introduces, alongside competence p. The
    trap is a consensus on one wrong answer, so before a team can fall into it
    the wrong side has to agree on which wrong answer to hold. A concentrated
    wrong side puts the team in the two-choice regime, where the boundary sits at
    c* = gamma/(gamma+a); a split one recovers well above it, because agents
    disagreeing about *which* wrong answer is right still trigger each other's
    re-derivations and each of those finds the truth with probability r.

    Ranges from 1/(K-1) (uniform over the distractors) to 1 (all on one). It is
    1.0 by definition on a two-answer task, and near 1 for a homogeneous LLM
    team, whose correlated priors put it on the same distractor from the start.
    """
    truth = t.truth()
    branch_rows = [r for r in t.rows if r["round"] == round_no and r.get("private")]
    per_task = {}
    for r in branch_rows:
        tr = truth.get(r["task_id"])
        if tr is None or r["private"] == tr:
            continue
        per_task.setdefault(r["task_id"], []).append(r["private"])
    out = {}
    for task, wrongs in per_task.items():
        if not wrongs:
            continue
        counts = Counter(wrongs)
        out[task] = counts.most_common(1)[0][1] / len(wrongs)
    return pd.Series(out, dtype=float)
