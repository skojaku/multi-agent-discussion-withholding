"""Every transcript in the repo, one row per (run, condition), with its verdict.

The question is whether these numbers behave like measurements. Three ways to
fail that a single-task study cannot see:

- the estimate is really a property of one task design, so it does not travel;
- the estimate travels but its *limits* do not, so a cell with eight usable rows
  looks exactly like one with eight hundred;
- the projection needed to compare a K-way task to a two-way one changes the
  answer, and nothing in the output says so.

So each row carries the operating point, the measurability verdict, the error
concentration that decides whether the two-choice reduction applies, and the
reading under the other projection. Outcome columns (round-0 vote accuracy
against final-round accuracy) go alongside, because the point of measuring the
margin is to order runs the same way the outcome does.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cbrm  # noqa: E402
from cbrm.io import majority  # noqa: E402

# Item files for the evidence-level estimators, matched by run prefix. Set from
# the workflow config (benchmarks.*.items); a family with no entry gets no
# evidence-level rates.
ITEMS: dict[str, str] = {}


def outcome(t, condition):
    """Round-0 independent vote against the final-round collective answer."""
    truth = t.truth()
    T = t.rounds
    first, last = {}, {}
    for r in t.rows:
        if r["round"] == 0 and r.get("public"):
            first.setdefault(r["task_id"], []).append(r["public"])
        if r["branch"] == condition and r["round"] == T and r.get("public"):
            last.setdefault(r["task_id"], []).append(r["public"])
    def acc(d):
        hits = [majority(v) == truth.get(k) for k, v in d.items() if truth.get(k)]
        return float(np.mean(hits)) if hits else np.nan
    return acc(first), acc(last)


def main(out_path, data_root=None, items_root=None, n_boot=600):
    root = Path(data_root)
    items_dir = Path(items_root or (root / "items"))
    adapters = {}
    frames = []
    for ev in sorted(root.glob("**/events.jsonl")):
        if "external" in str(ev) or "/items/" in str(ev):
            continue
        run = str(ev.parent.relative_to(root))
        try:
            t = cbrm.load_events(ev)
        except Exception as exc:                              # noqa: BLE001
            print(f"  skip {run}: {exc}")
            continue
        if not t.conditions or not t.has_truth:
            continue
        family = run.split("/")[0]
        ad = None
        if family in ITEMS and t.has_evidence:
            path = items_dir / ITEMS[family]
            if path.exists():
                if path not in adapters:
                    adapters[path] = cbrm.FactTallyAdapter(cbrm.load_items(path))
                ad = adapters[path]
        vote, final = {}, {}
        for cond in t.conditions:
            try:
                d = cbrm.diagnose(t, cond, run=run, n_boot=n_boot, evidence=ad,
                              bayes=True)
            except Exception as exc:                          # noqa: BLE001
                print(f"  skip {run}/{cond}: {exc}")
                continue
            row = d.to_frame()
            row["family"] = family
            v, f = outcome(t, cond)
            row["vote_accuracy"], row["final_accuracy"] = v, f
            row["interaction_gain"] = f - v
            frames.append(row)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 260)
    print(f"{len(df)} (run, condition) cells over {df['family'].nunique()} "
          f"task families\n")
    g = df.groupby("family").agg(
        cells=("condition", "size"),
        labels=("n_labels", "median"),
        kappa=("error_concentration", "median"),
        measurable=("measurable", "mean"),
        n_contra=("n_contra", "median"),
        c=("c", "median"), c_star=("c_star", "median"), margin=("margin", "median"),
        trapped=("verdict", lambda s: (s == "TRAPPED").mean()),
        inconclusive=("verdict", lambda s: (s == "INCONCLUSIVE").mean()),
        gain=("interaction_gain", "median"))
    print(g.round(3).to_string())

    print("\nverdicts:")
    print(df["verdict"].value_counts().to_string())

    ok = df[df["measurable"] & df["margin"].notna() & df["interaction_gain"].notna()]
    if len(ok) > 3:
        print(f"\nacross the {len(ok)} measurable cells, the margin c - c* and the "
              f"gain from interaction correlate at "
              f"r = {ok['margin'].corr(ok['interaction_gain']):.2f} "
              f"(Spearman {ok['margin'].corr(ok['interaction_gain'], method='spearman'):.2f}); "
              f"c alone gives {ok['c'].corr(ok['interaction_gain']):.2f}.")
        by = ok.groupby(ok["verdict"]).agg(n=("condition", "size"),
                                           gain=("interaction_gain", "mean"))
        print(by.round(3).to_string())

    alt = df.dropna(subset=["c", "c_alt_projection"])
    if len(alt):
        gap = (alt["c"] - alt["c_alt_projection"]).abs()
        print(f"\non the {len(alt)} cells with more than two answer labels, the "
              f"two projections differ by a median of {gap.median():.3f} "
              f"(max {gap.max():.3f}); the gap tracks the error concentration at "
              f"r = {alt['error_concentration'].corr(gap):.2f}.")
    return df


if "snakemake" in sys.modules:
    ITEMS.update(snakemake.params["items_map"])  # noqa: F821
    main(snakemake.output["sweep"], data_root=snakemake.params["transcripts_dir"],  # noqa: F821
         items_root=snakemake.params["items_dir"],  # noqa: F821
         n_boot=int(snakemake.params["n_boot"]))  # noqa: F821
elif __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="data/sweep.csv")
    ap.add_argument("--data-root", required=True,
                    help="transcript tree to read (data/transcripts)")
    ap.add_argument("--items-root", default=None,
                    help="item files for the evidence-level rates; default <data-root>/items")
    ap.add_argument("--n-boot", type=int, default=600)
    ap.add_argument("--items-map", default="{}",
                    help='json {"run prefix": "items file name"}')
    a = ap.parse_args()
    import json
    ITEMS.update(json.loads(a.items_map))
    main(a.out, data_root=a.data_root, items_root=a.items_root, n_boot=a.n_boot)
