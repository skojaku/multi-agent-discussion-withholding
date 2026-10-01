"""Generate hidden-profile hiring tasks: the wedge, built by construction.

Why this exists
---------------
The MMLU-Pro / FailureSensorIQ runs measured the theory in the one place it says
nothing: per-item competence came back nearly two-valued (every agent right or
every agent wrong), so the region the model is about -- a wrong majority holding
a recoverable answer -- was empty. That is a property of shared pretraining, not
something a better prompt fixes.

A hidden profile fixes it by construction (Stasser & Titus 1985). Two candidates.
The briefing every panelist reads favours the weaker one. The evidence that makes
the other candidate stronger is split across panelists' private notes, so it only
exists in the union. Pool the information and the answer flips; keep it private
and the panel confidently hires the wrong person.

The design parameter
--------------------
`n_informed` (k) is how many of the N agents hold enough private evidence to
prefer the stronger candidate on their own. The initial competence the theory
calls `p` is then k/N *by construction* -- it is set, not measured and hoped for.
k = 0 is the classic hidden profile: nobody can get it alone, and only sharing
gets the panel there.

The arithmetic, so the ground truth is not a matter of taste
------------------------------------------------------------
Every bullet is one independent, equally weighted positive finding, and the
stronger candidate is whoever has more of them in the pooled information. With
the defaults below, writing W for the weaker candidate and S for the stronger:

    shared briefing        3 findings for W,  1 for S      (seen by every agent)
    each agent's notes      1 finding for W,  u_i for S     (seen by that agent)
    u_i = 4 for k agents (informed), 2 for the other N-k

    agent i sees            W: 3+1 = 4        S: 1+u_i        -> prefers S iff u_i >= 4
    the panel together     W: 3+N = 8        S: 1+ sum u_i = 11+2k  (N=5)

so every agent with u_i = 2 rationally prefers W, every agent with u_i = 4
rationally prefers S, and the pooled evidence favours S at every k. No ties.

Which label (A or B) the stronger candidate gets is randomised per item, and both
candidates' findings are drawn from the same pool, so nothing distinguishes them
except the counts.
"""

import json
import random
from pathlib import Path

# One pool for both candidates: content cannot be what makes a candidate the
# stronger one, only the number of findings can.
#
# The pool is split in two on purpose. An item draws its findings without
# replacement, so ENLARGING the pool changes every draw and would silently
# regenerate items that transcripts already exist for. ``FINDINGS_CORE`` is the
# original 56 and is frozen; the extras below it are used only by items too big
# to be dealt from the core alone (N = 25 needs up to 109), so the N = 5 items
# this repo already ran are byte-identical to what they were.
FINDINGS_CORE = [
    "rewrote the payments API and cut p99 latency from 900 ms to 120 ms",
    "ran the migration off the legacy monolith with no customer-visible downtime",
    "is the maintainer of an open-source library with 12k stars",
    "cut the CI suite from 40 minutes to 6 without dropping coverage",
    "found and fixed a data-corruption bug that had survived two audits",
    "mentored three junior engineers who were all promoted within the year",
    "designed the sharding scheme that carried a 20x traffic increase",
    "wrote the incident review that changed how the whole org does on-call",
    "shipped the mobile checkout flow that lifted conversion by 4 points",
    "reduced cloud spend by 38% without a measurable latency regression",
    "led the SOC 2 remediation and closed every finding before the deadline",
    "built the feature-flag system now used by every team in the company",
    "took a stalled machine-learning project to production in one quarter",
    "rewrote the on-call runbooks, and page volume fell by half",
    "negotiated the API contract that unblocked the largest partner integration",
    "diagnosed a memory leak that three other engineers had failed to reproduce",
    "introduced property-based testing and it caught two production-class bugs",
    "kept a 99.99% availability target through a full datacentre failover",
    "wrote the internal design doc that is still the reference for new services",
    "shipped the search reranking change that cut zero-result queries by a third",
    "automated the release process, taking deploys from weekly to hourly",
    "handled the customer escalation that saved a renewal worth seven figures",
    "built the data pipeline that replaced four fragile cron jobs",
    "presented at an industry conference on the team's storage architecture",
    "rebuilt the permissions model after finding a privilege-escalation path",
    "cut the onboarding time for new engineers from three weeks to four days",
    "owned the GDPR data-deletion work end to end and passed the external audit",
    "wrote the load generator that exposed the capacity ceiling before launch",
    "refactored the billing logic that had accumulated nine years of exceptions",
    "shipped the offline mode that became the most-requested feature's answer",
    "led the postmortem culture change away from blame and towards mechanisms",
    "built the schema-migration tool the platform team now depends on",
    "took over an unowned service and brought its error budget back into the green",
    "improved model inference throughput 5x by rewriting the batching layer",
    "wrote the fuzzing harness that found the parser bug before an attacker did",
    "moved the analytics stack to columnar storage and queries got 30x faster",
    "designed the retry and backoff policy that ended a recurring thundering herd",
    "shipped accessibility fixes that took the product to WCAG 2.1 AA",
    "built the canary analysis that now blocks a third of bad deploys automatically",
    "documented the undocumented protocol that two teams had been guessing at",
    "ran the vendor evaluation that avoided a costly database migration",
    "wrote the caching layer that took read load off the primary by 70%",
    "turned an ad-hoc spreadsheet process into a service the finance team uses daily",
    "found the race condition behind a class of intermittent test failures",
    "led the API versioning strategy that let three clients upgrade independently",
    "reduced the alerting false-positive rate from 60% to under 10%",
    "shipped the localisation work that opened two new markets on schedule",
    "rebuilt the deployment pipeline so rollbacks take 90 seconds, not an hour",
    "wrote the query planner change that removed the nightly batch window",
    "took the flaky end-to-end suite from 12% failure to under 1%",
    "designed the event schema that unblocked the whole experimentation platform",
    "led the effort that cut mean time to recovery from 4 hours to 25 minutes",
    "built the internal CLI that half the engineering org now uses daily",
    "handled a security disclosure end to end, including the customer comms",
    "reworked the index strategy and dropped storage cost by a fifth",
    "shipped the streaming ingestion path that retired the old batch importer",
]

# Added for the N = 25 panels. Interchangeable with the core by construction.
FINDINGS_EXTRA = [
    "cut the cold-start time of the serving stack from 30 seconds to 4",
    "wrote the capacity model the whole org now uses for headcount planning",
    "found the memory leak that had been restarting the fleet nightly for a year",
    "replaced the hand-rolled scheduler with a queue and halved job latency",
    "ran the postmortem process for the region-wide outage in 2024",
    "built the feature store that three teams now depend on",
    "reduced the on-call page volume by 70% without silencing real alerts",
    "designed the tenancy model that let the product serve regulated customers",
    "shipped the audit-log subsystem that unblocked two enterprise deals",
    "wrote the load generator that catches regressions before they reach staging",
    "moved the primary datastore across cloud regions with 8 minutes of downtime",
    "authored the API deprecation policy the org still follows",
    "cut image build times from 20 minutes to 3 with a layer-cache rewrite",
    "fixed the clock-skew bug behind a class of duplicate charges",
    "led the migration to typed configuration and removed a whole failure class",
    "wrote the runbooks that let a new on-call rotation start in two weeks",
    "built the canary analysis that stops bad rollouts automatically",
    "negotiated the vendor contract that cut observability spend by a third",
    "designed the retry semantics that made the checkout flow idempotent",
    "rewrote the permission checks and closed a horizontal privilege escalation",
    "shipped the offline mode that halved support tickets from field technicians",
    "wrote the schema-evolution tooling that ended breaking consumer changes",
    "cut p95 query time on the reporting database from 12 seconds to 800 ms",
    "built the synthetic monitoring that catches outages before customers call",
    "led the rewrite of the notification service and dropped delivery failures 10x",
    "designed the sampling strategy that made tracing affordable at full traffic",
    "found the race condition behind a long-standing intermittent data loss",
    "shipped the bulk-export API that removed the top support escalation",
    "built the test-data generator that let integration tests run in parallel",
    "wrote the migration that removed 200k lines of dead code safely",
    "designed the rate limiter that survived the launch-day traffic spike",
    "rebuilt the search indexing pipeline and cut freshness lag from hours to minutes",
    "led the accessibility audit and remediation for the customer portal",
    "wrote the cost attribution system that made per-team spend visible",
    "shipped the change-data-capture pipeline that retired nightly dumps",
    "fixed the connection-pool exhaustion that caused weekly brownouts",
    "designed the multi-region failover that was exercised in a real outage",
    "built the developer environment that cut new-hire ramp from weeks to days",
    "wrote the static analysis rule that eliminated a recurring injection bug",
    "led the effort to make every service emit structured logs",
    "shipped the backfill framework that made historical corrections routine",
    "designed the queue partitioning that ended head-of-line blocking",
    "cut the storage footprint of the event archive by 60% with no data loss",
    "built the on-call handoff tooling now used by every team in the division",
    "wrote the compatibility layer that let two products merge their accounts",
    "found the misconfiguration that had silently disabled backups for months",
    "shipped the SDK rewrite that cut integration time for partners in half",
    "designed the deduplication logic that fixed double-counted revenue",
    "led the upgrade of the whole fleet across three major runtime versions",
    "built the replay tooling that turned incident debugging into minutes",
    "wrote the throttling policy that kept the API up during a scraping attack",
    "shipped the config service that removed the need for redeploys to tune limits",
    "designed the archival tiering that cut cold-storage cost by half",
    "rebuilt the webhook delivery system and reached five nines of delivery",
    "led the effort that brought the flaky mobile release train back on schedule",
    "wrote the fuzzing harness that surfaced three parser vulnerabilities",
    "shipped the read-replica routing that removed the last single point of failure",
    "designed the idempotency keys that made retries safe across the platform",
    "built the anomaly detection that caught a billing regression within an hour",
    "cut the size of the client bundle by 40% without dropping features",
    "wrote the data-retention automation that closed a compliance finding",
    "shipped the internal metrics catalog that ended duplicate dashboards",
    "designed the shadow-traffic setup used to validate every major rewrite",
    "led the incident that exposed the gap in cross-region monitoring, and closed it",
    "built the migration harness that moved 4 billion rows without a maintenance window",
    "wrote the linting rules that stopped a recurring class of null-pointer crashes",
    "shipped the partial-failure semantics that made batch endpoints usable",
    "designed the leader election that ended split-brain in the coordinator",
    "cut the tail latency of the recommendation service by half with better batching",
    "built the regression suite that made the annual dependency upgrade routine",
    "wrote the capacity alerts that gave three weeks of warning before saturation",
    "shipped the encryption-at-rest rollout across every data store",
    "designed the request-hedging policy that hid a flaky downstream dependency",
    "led the deprecation of the old auth stack, including every client migration",
]

FINDINGS = FINDINGS_CORE + FINDINGS_EXTRA

ROLES = [
    ("staff backend engineer", "the payments platform team"),
    ("senior infrastructure engineer", "the core platform team"),
    ("machine-learning engineer", "the ranking team"),
    ("site reliability engineer", "the availability team"),
    ("security engineer", "the product security team"),
    ("data engineer", "the analytics platform team"),
]

NAMES = ["Avery", "Blake", "Casey", "Devon", "Ellis", "Finley", "Harper",
         "Jordan", "Kendall", "Logan", "Morgan", "Parker", "Quinn", "Riley",
         "Sawyer", "Tatum"]


def make_item(item_id, rng, n_agents=5, n_informed=1,
              shared_weak=3, shared_strong=1, notes_weak=1,
              notes_strong_hi=4, notes_strong_lo=2):
    """One hidden-profile item. ``n_informed`` agents can get it right alone."""
    role, team = rng.choice(ROLES)
    name_a, name_b = rng.sample(NAMES, 2)
    strong_label = rng.choice(["A", "B"])          # no position bias
    weak_label = "B" if strong_label == "A" else "A"
    names = {"A": name_a, "B": name_b}

    u = ([notes_strong_hi] * n_informed
         + [notes_strong_lo] * (n_agents - n_informed))
    rng.shuffle(u)                                  # which agents are informed
    need = shared_weak + shared_strong + n_agents * notes_weak + sum(u)
    # Deal from the frozen core when it is big enough, so small panels keep the
    # items they had before the pool grew.
    source = FINDINGS_CORE if need <= len(FINDINGS_CORE) else FINDINGS
    pool = rng.sample(source, need)
    take = iter(pool)

    shared = ([{"about": weak_label, "text": next(take)} for _ in range(shared_weak)]
              + [{"about": strong_label, "text": next(take)} for _ in range(shared_strong)])
    rng.shuffle(shared)

    agent_notes = {}
    for agent in range(n_agents):
        notes = ([{"about": weak_label, "text": next(take)} for _ in range(notes_weak)]
                 + [{"about": strong_label, "text": next(take)} for _ in range(u[agent])])
        rng.shuffle(notes)
        agent_notes[str(agent)] = notes

    def tally(facts):
        return {lab: sum(f["about"] == lab for f in facts) for lab in ("A", "B")}

    pooled = tally(shared + [f for v in agent_notes.values() for f in v])
    per_agent = {s: tally(shared + v) for s, v in agent_notes.items()}
    solo_correct = [s for s, t in per_agent.items()
                    if t[strong_label] > t[weak_label]]

    assert pooled[strong_label] > pooled[weak_label], "pooled evidence must favour the answer"
    assert len(solo_correct) == n_informed, "informed-agent count must match the design"

    return {
        "id": item_id,
        "kind": "hidden_profile",
        "question": (f"{team} is hiring one {role}. Two finalists are left: "
                     f"Candidate A ({names['A']}) and Candidate B ({names['B']}). "
                     "Which one should the panel recommend?"),
        "choices": [f"Candidate A ({names['A']})", f"Candidate B ({names['B']})"],
        "answer": strong_label,
        "shared_facts": shared,
        "agent_facts": agent_notes,
        "design": {
            "n_agents": n_agents,
            "n_informed": n_informed,
            "p_design": n_informed / n_agents,
            "solo_correct_agents": sorted(int(s) for s in solo_correct),
            "pooled_tally": pooled,
            "per_agent_tally": per_agent,
            "role": role,
            "strong_label": strong_label,
        },
    }


def main(out_path, n_per_level=30, levels=(0, 1, 2, 3), n_agents=5, seed=0):
    rng = random.Random(seed)
    items = []
    for k in levels:
        for i in range(n_per_level):
            items.append(make_item(f"hp_k{k}_{i:03d}", rng, n_agents=n_agents,
                                   n_informed=k))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in items))
    print(f"wrote {out} ({len(items)} items, k in {list(levels)}, N={n_agents})")
    return items


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        main(sm.output.items_file, n_per_level=sm.params.n_per_level,
             levels=tuple(sm.params.levels), n_agents=sm.params.n_agents,
             seed=sm.params.seed)
    else:
        import argparse
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--out", default="data/items/hidden_profile.jsonl")
        ap.add_argument("--n-per-level", type=int, default=30)
        ap.add_argument("--levels", default="0,1,2,3")
        ap.add_argument("--agents", type=int, default=5)
        ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args()
        main(a.out, a.n_per_level, tuple(int(x) for x in a.levels.split(",")),
             a.agents, a.seed)
