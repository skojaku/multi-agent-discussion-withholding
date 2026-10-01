"""Build a hidden profile out of MedEinst, whose decoy is somebody else's work.

The gap this fills
------------------
Silo-Bench (7.19) turned out to be a poor transfer test for the central claim,
and for a structural reason: it has no shared information at all, so a wrong
majority can only appear by coincidence, and the conformity framing has nothing
to act on. HiddenBench has decoys but is saturated for a current reasoning model.
What the claim needs is a task family where (i) a wrong majority forms BY
CONSTRUCTION, (ii) frontier models still fail, and (iii) the misleading part is
not something this repo wrote.

MedEinst (Zhu et al., ACL 2026; CC-BY-4.0, grounded in DDXPlus) is built for
exactly (i) and (ii). Each case comes as a pair:

  control   a typical presentation whose statistical prior matches the diagnosis
  trap      the same patient with ONE discriminative finding altered, which flips
            the true diagnosis while the rest of the picture still looks like the
            control's disease

Their paper reports that frontier models keep answering the control's diagnosis
on trap cases -- the Einstellung effect -- so the decoy demonstrably works.

The construction
----------------
Diffing a pair recovers the discriminative finding automatically, because the
median pair differs in exactly one line. From a trap case:

  shared briefing   the findings common to both cases. On their own they support
                    the CONTROL diagnosis, which is wrong here: this is the
                    misleading common prior, and it is theirs, not ours.
  private notes     every agent holds exactly one finding. ``k`` agents hold the
                    discriminative one; the rest hold a held-out non-decisive
                    finding, so that having a private note is not itself a signal.
  options           the trap diagnosis (correct) against the control diagnosis
                    (the decoy). Two options, as in the hiring panel.

So p = k/N by construction again, and the wedge can be traced -- but the content,
the diagnoses and the trap are the benchmark's.
"""

import argparse
import difflib
import json
import random
from collections import defaultdict
from pathlib import Path


def norm_lines(narrative):
    return [l.rstrip() for l in narrative.splitlines() if l.strip()]


def is_finding(line):
    """A top-level symptom bullet, not a header, sub-bullet or preamble.

    Sub-bullets ("  - The rash is pink.") are qualifiers of the bullet above
    them and mean nothing on their own, so handing one to an agent as its whole
    private note would be handing it a fragment.
    """
    return (line.startswith("- ") and not line.lower().startswith("- sex")
            and len(line) > 8)


def too_similar(a, b, threshold=0.6):
    """Near-duplicate findings, which would leak the decisive one.

    The generated narratives contain variants of the same finding ("coughing up
    blood" and "coughing up blood recently"), and one of those handed to an
    uninformed agent as a filler note would quietly make it informed.
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def discriminative(control, trap):
    """The finding(s) the trap adds or changes. Empty when the diff is not clean."""
    cl, tl = norm_lines(control["narrative"]), norm_lines(trap["narrative"])
    sm = difflib.SequenceMatcher(None, cl, tl)
    added, common = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            common.extend(tl[j1:j2])
        else:
            added.extend(tl[j1:j2])
    added = [l for l in added if is_finding(l)]
    common = [l for l in common if is_finding(l)]
    return added, common


def make_item(item_id, control, trap, rng, n_agents=5, n_informed=1,
              distractors=(), notes_per_agent=1):
    """One item. ``notes_per_agent`` > 1 gives every agent a MENU.

    Selective disclosure -- tabling the finding that supports the position you
    already hold rather than the one that would change it (7.15) -- needs
    something to choose between. With one note each, an agent can only speak or
    stay silent, and that is why the single-note build of this family came out
    use-bound rather than disclosure-bound. Here each agent holds several
    findings, exactly one of which is decisive for the informed agents, so
    "which note did it table" is observable.

    A case carries about eleven top-level findings, which is not enough to give
    five agents three private ones each without repeats. The fillers are
    therefore drawn from a held-out pool WITH overlap between agents: partially
    shared information is what a real hidden profile looks like, and the
    decisive finding is still held by exactly ``n_informed`` agents.
    """
    added, common = discriminative(control, trap)
    if len(added) != 1 or len(common) < n_agents + 2:
        return None
    decisive = added[0]
    # One held-out finding per uninformed agent, taken out of the briefing so
    # that every agent has exactly one private note and the same amount of it.
    pool = [l for l in common if not too_similar(l, decisive)]
    # The filler pool only has to be big enough for one agent to draw distinct
    # notes, because draws overlap between agents. Keeping it small leaves the
    # briefing long, and the briefing IS the decoy: strip it down and the
    # misleading prior this family exists for stops pulling.
    n_fillers = notes_per_agent + 1
    if len(pool) < n_fillers + 6:
        return None
    rng.shuffle(pool)
    holdout = pool[:n_fillers]
    shared = [l for l in common if l not in holdout]
    if len(shared) < 6:
        return None

    informed = set(rng.sample(range(n_agents), n_informed))
    agent_facts = {}
    for a in range(n_agents):
        n_extra = notes_per_agent - (1 if a in informed else 0)
        fillers = rng.sample(holdout, min(n_extra, len(holdout)))
        notes = ([decisive] if a in informed else []) + fillers
        rng.shuffle(notes)
        agent_facts[str(a)] = [{"about": None, "text": t} for t in notes]
    # Two options put chance at 1/2 and the first pilot came back at 0.51 for
    # agents holding no decisive finding: with only the trap and the decoy on
    # the table a model can reason its way to the answer instead of falling for
    # the prior, which is the effect this family is here to exercise. Additional
    # diagnoses from the benchmark's own label set push chance down and let the
    # decoy do its work.
    options = [trap["ground_truth"], control["ground_truth"]]
    extra = [d for d in distractors
             if d not in options][:max(0, len(distractors))]
    rng.shuffle(extra)
    options += extra[:3]
    rng.shuffle(options)
    return {
        "id": item_id,
        "kind": "medeinst",
        "question": (
            f"A patient (sex {trap['sex']}, age {trap['age']}) is being "
            "discussed by a panel of clinicians. Each clinician has read the "
            "same case summary and, in addition, holds one finding from the "
            "record that the others have not seen. Which diagnosis is correct?"),
        "choices": options,
        "answer": trap["ground_truth"],
        "shared_facts": [{"about": None, "text": l} for l in shared],
        "agent_facts": agent_facts,
        "design": {
            "n_agents": n_agents,
            "n_informed": n_informed,
            "p_design": n_informed / n_agents,
            "source": "medeinst",
            "case_id": trap["case_id"],
            "decoy_diagnosis": control["ground_truth"],
            "decisive_finding": decisive,
            "notes_per_agent": notes_per_agent,
            "informed_agents": sorted(informed),
        },
    }


def main(src, out_path, n_per_level=30, levels=(0, 1, 2, 3), n_agents=5, seed=0,
         notes_per_agent=1):
    rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r["case_id"]][r["case_type"]] = r
    usable = [v for v in pairs.values() if len(v) == 2]
    labels = sorted({r["ground_truth"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(usable)

    items, cursor, dropped = [], 0, 0
    for k in levels:
        made = 0
        while made < n_per_level and cursor < len(usable):
            pair = usable[cursor]
            cursor += 1
            it = make_item(f"me_k{k}_{made:03d}", pair["control"], pair["trap"],
                           rng, n_agents, k, rng.sample(labels, 8),
                           notes_per_agent)
            if it is None:
                dropped += 1
                continue
            items.append(it)
            made += 1
        if made < n_per_level:
            raise SystemExit(f"ran out of usable pairs at k={k}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in items))
    print(f"wrote {out} ({len(items)} items, k in {list(levels)}, N={n_agents}; "
          f"skipped {dropped} pairs whose diff was not a single finding)")
    ex = items[n_per_level]  # first k=1 item
    print("\nexample (k=1):")
    print("  answer:", ex["answer"], "| decoy:", ex["design"]["decoy_diagnosis"])
    print("  options:", ex["choices"])
    print("  decisive:", ex["design"]["decisive_finding"])
    print("  shared briefing:", len(ex["shared_facts"]), "findings")
    for a, notes in ex["agent_facts"].items():
        mark = "*" if any(n["text"] == ex["design"]["decisive_finding"]
                          for n in notes) else " "
        print(f"   {mark}agent {a}: " + " | ".join(n["text"][:34] for n in notes))
    return items


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        main(sm.input.source, sm.output.items_file,
             int(sm.params.n_per_level), list(sm.params.levels),
             int(sm.params.n_agents), int(sm.params.seed),
             int(getattr(sm.params, "notes_per_agent", 1)))
    else:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--source", default="data/external/medeinst/test.jsonl")
        ap.add_argument("--out", default="data/items/medeinst.jsonl")
        ap.add_argument("--n-per-level", type=int, default=30)
        ap.add_argument("--levels", default="0,1,2,3")
        ap.add_argument("--agents", type=int, default=5)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--notes", type=int, default=1,
                        help="private findings per agent; >1 gives a menu")
        a = ap.parse_args()
        main(a.source, a.out, a.n_per_level,
             [int(x) for x in a.levels.split(",")], a.agents, a.seed, a.notes)
