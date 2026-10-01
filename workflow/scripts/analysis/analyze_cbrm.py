"""Re-measure one transcript through libs/cbrm, and show what it changes.

The parameter tables in this experiment were written by `analyze_llm.py` and
`analyze_hidden_profile.py`, which grew alongside the runs. `libs/cbrm` is the
same estimators cut out, checked against transcripts whose parameters are known
(estimator validation, not part of this repository), and corrected in three places that the recovery
tests exposed:

1. **The plurality is taken before the projection.** On K options the model's
   visible majority is the most common *answer*; replacing every answer by
   whether it is correct first turns three agents holding three different wrong
   answers into a unanimous bloc that nobody would concede to. On synthetic
   K-choice logs the raw-answer reading recovers a true c = 0.500 to within
   0.002 at every K, while the correct/incorrect reading falls to 0.363 at
   K = 10 with the errors split.
2. **Rows with a tied panel stay in the repair denominator.** The old estimator
   skipped them. Right for concealment -- with no majority there is nothing to
   concede to -- and wrong for repair, which the model gates on visible dissent
   alone, and a tied panel is the most dissenting panel there is. It biases rho
   low by 0.10 at N = 5, which is every run here.
3. **The summed dissent is not rounded.** The old code rounded it to a whole
   number because a Wilson interval needs an integer n. Intervals here come from
   a cluster bootstrap over tasks, which also gives the derived quantities --
   gamma, c*, and the margin c - c* -- a joint interval instead of pretending c
   and c* are independent.

Writes both readings side by side so the size of each correction is on the page
rather than in a commit message. Nothing overwrites the published tables.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import cbrm  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--items", default="", help="enables the evidence-level rates")
    ap.add_argument("--out-params", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-bayes", action="store_true",
                    help="skip the posteriors; the counting estimates alone")
    args = ap.parse_args(argv)

    t = cbrm.load_events(args.events)
    ad = None
    if args.items and Path(args.items).exists() and t.has_evidence:
        ad = cbrm.FactTallyAdapter(cbrm.load_items(args.items))

    run = str(Path(args.events).parent.name)
    frames = []
    for cond in t.conditions:
        d = cbrm.diagnose(t, cond, run=run, n_boot=args.n_boot, evidence=ad,
                          bayes=not args.no_bayes)
        row = d.to_frame()
        # The published estimator, reproduced exactly, so the delta is visible.
        dec = cbrm.decisions(t, cond, projection="letter")
        old = cbrm.rates(cbrm.counts(dec, repair_requires_majority=True),
                         legacy_round=True)
        for k in ("c", "a", "rho", "r", "gamma", "c_star", "margin"):
            row[f"{k}_published"] = old.get(k)
            row[f"{k}_delta"] = row[k] - old.get(k)
        frames.append(row)
    if not frames:
        raise SystemExit(f"no conditions in {args.events}")
    df = pd.concat(frames, ignore_index=True)
    Path(args.out_params).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_params, index=False)

    pd.set_option("display.width", 240)
    cols = ["condition", "c", "a", "rho", "r", "gamma", "c_star", "margin",
            "verdict", "n_contra"]
    print(df[cols].round(3).to_string(index=False))
    print("\nchange from the published estimator:")
    print(df[["condition"] + [f"{k}_delta" for k in
                              ("c", "a", "rho", "r", "c_star", "margin")]]
          .round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
