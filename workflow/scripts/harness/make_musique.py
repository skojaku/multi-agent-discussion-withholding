"""A retrieval hidden profile from MuSiQue, where the fragmentation is theirs.

Why this family closes the gap
------------------------------
The two structural variables this report ended up caring about -- how many
private pieces the answer needs (fragmentation) and whether agents also hold
evidence that pulls the wrong way (conflict) -- were introduced by hand in the
AssetOps build, which is a fair objection: they could be artefacts of what we
chose to construct.

MuSiQue (Trivedi et al., TACL 2022) has both already, and neither was put there
for this purpose:

  fragmentation   its questions are 2, 3 or 4 hop by construction, and each hop
                  has its own supporting paragraph. Two hops means two documents
                  neither of which answers the question alone. The hop count is
                  the dataset's own label, not our split.
  conflict        each item ships 20 paragraphs of which only 2-4 support the
                  answer; the rest are retrieved distractors, topically adjacent
                  and chosen to mislead. That is the corpus a real RAG pipeline
                  returns.

The split policy is the only thing this file decides, and it mirrors how a real
multi-agent retrieval system ends up asymmetric: the shallow hits that every
query returns are shared, and the deep hit that one agent's query surfaced is
private to it.

  shared briefing   several distractor paragraphs -- what everyone's search
                    returned, plausible and wrong
  private notes     one supporting paragraph each for the F hop-holders, plus
                    distractors for everyone, so holding something is not itself
                    a signal
  answer            the dataset's answer string and its aliases; free form, so
                    guessing is not a strategy
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def clip(text, words=90):
    parts = text.split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def make_item(row, rng, n_agents=5, shared_distractors=3, extra_each=1):
    supporting = [p for p in row["paragraphs"] if p.get("is_supporting")]
    distractors = [p for p in row["paragraphs"] if not p.get("is_supporting")]
    hops = len(supporting)
    if hops < 2 or hops > n_agents or len(distractors) < shared_distractors + \
            n_agents * extra_each:
        return None

    rng.shuffle(distractors)
    shared = distractors[:shared_distractors]
    pool = distractors[shared_distractors:]

    holders = rng.sample(range(n_agents), hops)
    agent_facts = defaultdict(list)
    for slot, agent in enumerate(holders):
        p = supporting[slot]
        agent_facts[str(agent)].append({
            "about": None, "decisive": True,
            "text": f"[{p['title']}] {clip(p['paragraph_text'])}"})
    for i, agent in enumerate(range(n_agents)):
        for j in range(extra_each):
            p = pool[(i * extra_each + j) % len(pool)]
            agent_facts[str(agent)].append({
                "about": None, "decisive": False,
                "text": f"[{p['title']}] {clip(p['paragraph_text'])}"})
        rng.shuffle(agent_facts[str(agent)])

    return {
        "id": f"mq{hops}_{row['id']}",
        "kind": "musique",
        "question": (
            "A research team is answering one question. Everyone has read the "
            "same set of search results, and each member additionally found one "
            "or more documents the others did not. Answer the question.\n\n"
            f"QUESTION: {row['question']}"),
        "choices": [],
        "answer": row["answer"],
        "answer_aliases": row.get("answer_aliases") or [],
        "shared_facts": [{"about": None,
                          "text": f"[{p['title']}] {clip(p['paragraph_text'])}"}
                         for p in shared],
        "agent_facts": dict(agent_facts),
        "design": {
            "n_agents": n_agents,
            "fragmentation": hops,
            "n_informed": hops,
            "p_design": 0.0,   # no single agent holds every hop
            "source": "musique",
            "musique_id": row["id"],
            "decisive_atoms": [f"[{p['title']}] {clip(p['paragraph_text'])}"
                               for p in supporting],
        },
    }


def main(src, out_path, per_hop=40, hops=(2, 3, 4), n_agents=5, seed=0):
    rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    made = defaultdict(list)
    for row in rows:
        if not row.get("answerable", True):
            continue
        h = len([p for p in row["paragraphs"] if p.get("is_supporting")])
        if h not in hops or len(made[h]) >= per_hop:
            continue
        it = make_item(row, rng, n_agents)
        if it:
            made[h].append(it)
        if all(len(made[x]) >= per_hop for x in hops):
            break
    items = [it for h in hops for it in made[h]]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                           for x in items))
    print(f"wrote {out} ({len(items)} items; per hop "
          f"{ {h: len(made[h]) for h in hops} })")
    ex = made[hops[0]][0]
    print(f"\nexample F={ex['design']['fragmentation']}: {ex['answer']!r}")
    print("  shared:", len(ex["shared_facts"]), "distractor paragraphs")
    for a, notes in sorted(ex["agent_facts"].items()):
        print(f"   agent {a}: " + " || ".join(
            ("*" if n["decisive"] else " ") + n["text"][:52] for n in notes))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/external/musique/dev.jsonl")
    ap.add_argument("--out", default="data/items/musique.jsonl")
    ap.add_argument("--per-hop", type=int, default=40)
    ap.add_argument("--hops", default="2,3,4")
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.src, a.out, a.per_hop, [int(x) for x in a.hops.split(",")],
         a.agents, a.seed)
