"""Per-model table of the hidden-profile runs: accuracy, pooling, what is tabled.

One row per (model, condition). Fig. 2 (g) reads ``count_correct`` (the manipulation
check: can the model count the evidence it sees) to order the models; the share
of tabled findings that back the agent's own answer or the truth is recomputed
from the transcripts by the figure itself.

Note: ``sufficient_rate`` and ``acc_if_sufficient`` here are not the evidence
pooling and utilization rates of Section 4.1 -- those are ``ev_D`` and ``ev_U``
in cbrm_params.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_llm import load  # noqa: E402

LADDER = ["A0", "C0", "C1", "C2", "C3", "C4"]
# Ordered weakest to strongest on this task, which is also the order the
# disclosure rate falls in -- the point of the figure.

def read_run(label, run_dir, items, events_dir=None):
    """Per-condition outcome, mechanism and disclosure-content numbers."""
    run_dir = Path(run_dir)
    acc = pd.read_csv(run_dir / "hp_accuracy.csv")
    prm = pd.read_csv(run_dir / "hp_params.csv").set_index("condition")
    man = pd.read_csv(run_dir / "hp_manipulation.csv")
    allrows = acc[acc["k_informed"] == "all"].set_index("condition")

    # What the agents chose to disclose, which no csv carries: whether the shared
    # finding backed the agent's own current answer or the true candidate.
    share = {}
    for r in load(Path(events_dir or run_dir) / "events.jsonl"):
        if r["round"] == 0 or not r.get("share_about"):
            continue
        c = r["branch"]
        s = share.setdefault(c, dict(n=0, own=0, truth=0))
        s["n"] += 1
        s["own"] += int(r["share_about"] == r["private"])
        s["truth"] += int(r["share_about"] == r["answer"])

    out = []
    for cond in LADDER:
        if cond not in allrows.index or cond not in prm.index:
            continue
        s = share.get(cond, dict(n=0, own=0, truth=0))
        out.append(dict(
            model=label, condition=cond,
            accuracy=float(allrows.loc[cond, "accuracy"]),
            vote=float(allrows.loc[cond, "vote_accuracy"]),
            sufficient_rate=float(allrows.loc[cond, "sufficient_rate"]),
            acc_if_sufficient=float(allrows.loc[cond, "acc_if_sufficient"]),
            pooled_fraction=float(allrows.loc[cond, "pooled_fraction"]),
            phi_hat=float(prm.loc[cond, "phi_hat"]),
            lambda_hat=float(prm.loc[cond, "lambda_hat"]),
            share_rate=float(prm.loc[cond, "share_rate"]),
            disclose_truth_rate=float(prm.loc[cond, "disclose_truth_rate"]),
            shares_own_side=(s["own"] / s["n"]) if s["n"] else np.nan,
            shares_truth_side=(s["truth"] / s["n"]) if s["n"] else np.nan,
            count_correct=float(np.nanmean(man["count_correct"])),
            p_hat_gap=float(np.nanmean(np.abs(man["p_hat"] - man["p_design"]))),
        ))
    return pd.DataFrame(out)


def main(runs, items_path, out_table):
    """runs: list of (label, estimates directory, transcript directory)."""
    items = {json.loads(l)["id"]: json.loads(l)
             for l in Path(items_path).read_text().splitlines() if l.strip()}
    frames = [read_run(label, d, items, ev) for label, d, ev in runs if
              (Path(d) / "hp_accuracy.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    Path(out_table).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False)

    cols = ["model", "condition", "accuracy", "vote", "sufficient_rate",
            "acc_if_sufficient", "shares_truth_side", "shares_own_side",
            "phi_hat", "count_correct"]
    print(df[df["condition"].isin(["A0", "C0", "C4"])][cols].round(3)
          .to_string(index=False))
    print("\nper model, averaged over the ladder:")
    print(df.groupby("model")[["accuracy", "vote", "sufficient_rate",
                               "acc_if_sufficient", "shares_truth_side",
                               "count_correct"]].mean().round(3).to_string())
    print("wrote", out_table)


# The label is what lands in the table's `model` column, and F2(g) selects rows
# by the model names in the config, so the workflow passes those names exactly.
if __name__ == "__main__":
    sm = globals()["snakemake"]
    main(list(zip(sm.params.labels, sm.params.dirs, sm.params.events_dirs)),
         sm.input.items_file, sm.output.table)
