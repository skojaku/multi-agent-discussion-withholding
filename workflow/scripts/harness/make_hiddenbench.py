"""Convert HiddenBench into the item schema this experiment's runner reads.

Why an external benchmark
-------------------------
Every LLM result in this report so far comes from one task family: a synthetic
hiring panel whose evidence is additive, so "pool the evidence" reduces to
"count the findings". That is a real limit on what the findings mean. HiddenBench
(Li et al., arXiv:2505.11556; 65 tasks, MIT licence) is an independent,
published hidden-profile benchmark whose clues work by *elimination* -- a clue
rules an option out rather than adding weight to one -- and whose scenarios were
written by other people for another purpose.

So it tests the same claim under a different aggregation rule and without this
repo's authorship: does the conformity collapse still happen, and do the
interventions in 7.16 still fix it.

What changes, and what does not
-------------------------------
Same:   one shared briefing every agent reads, private information split across
        agents, the answer reachable only by pooling, one round-0 answer per
        agent as the no-interaction baseline.
Differs: each agent holds exactly ONE clue here, not a bundle, so disclosure is a
        binary act per agent and a single mandated disclosure round can put
        everything on the table. The competence p is a property of the task
        rather than a design variable -- these tasks are built so that nobody
        can solve them alone, which is the p = 0 level of the hiring task.
        Nothing here can trace the phase boundary; it tests one point on it.

The conversion keeps the benchmark's own text verbatim. The only additions are
the option list rendered into the question and the ids the SHARE line refers to.
"""

import argparse
import json
from pathlib import Path


def convert(task):
    """One HiddenBench task -> one item in this experiment's schema."""
    clues = task["hidden_information"]
    options = task["possible_answers"]
    answer = task["correct_answer"]
    if answer not in options:
        raise SystemExit(f"task {task['id']}: correct answer is not an option")
    n_agents = len(clues)
    return {
        "id": f"hb_{task['id']:03d}_{task['name']}",
        "kind": "hiddenbench",
        # The description already states the task and the options in prose; the
        # explicit list is appended so the answer line has an exact vocabulary.
        "question": task["description"].strip(),
        "choices": list(options),
        "answer": answer,
        "shared_facts": [{"about": None, "text": t}
                         for t in task["shared_information"]],
        "agent_facts": {str(i): [{"about": None, "text": clue}]
                        for i, clue in enumerate(clues)},
        "design": {
            "n_agents": n_agents,
            "source": "hiddenbench",
            "name": task["name"],
            # Not a design variable here: these tasks are constructed so that no
            # single agent can solve them, which is the p = 0 level of the
            # hiring task. Round 0 measures whether that holds for a given model.
            "n_informed": 0,
            "p_design": 0.0,
        },
    }


def main(src, out_path, limit=0):
    tasks = json.loads(Path(src).read_text())
    items = [convert(t) for t in tasks]
    if limit:
        items = items[:limit]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in items))
    sizes = {}
    for it in items:
        sizes[it["design"]["n_agents"]] = sizes.get(it["design"]["n_agents"], 0) + 1
    print(f"wrote {out} ({len(items)} items; agents per task: {sizes}; "
          f"options: {sorted({len(i['choices']) for i in items})})")
    return items


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        main(sm.input.source, sm.output.items_file,
             int(getattr(sm.params, "limit", 0) or 0))
    else:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--source", default="data/external/hiddenbench.json")
        ap.add_argument("--out", default="data/items/hiddenbench.jsonl")
        ap.add_argument("--limit", type=int, default=0)
        a = ap.parse_args()
        main(a.source, a.out, a.limit)
