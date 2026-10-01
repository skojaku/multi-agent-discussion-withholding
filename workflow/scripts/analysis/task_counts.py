"""Per-task withholding counts, one row per (run, condition, task).

``sweep.csv`` keeps one row per setting, with the counts summed over tasks. The
one-stage decomposition (``decomposition.py --one-stage``) fits those counts
directly and needs them split by task, because the rows within a task are not
independent: the task effect it puts in is what the Beta over tasks is in the
per-setting estimate of Section 3.1. The counting rule is the one ``cbrm.bayes``
uses -- a contradicted row is one whose visible majority differs from the
agent's previous private answer, and a withheld row is a contradicted row whose
public answer is that majority -- so the per-task counts sum to ``n_contra`` and
``n_conceal`` of the sweep, which the script checks.

    python task_counts.py SWEEP.csv OUT.csv --data-root data/transcripts
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import cbrm  # noqa: E402
from cbrm.io import decisions  # noqa: E402


def counts(dec: pd.DataFrame) -> pd.DataFrame:
    contradicted = dec["M"].notna() & (dec["M"] != dec["c_prev"])
    withheld = contradicted & (dec["pub"] == dec["M"])
    g = pd.DataFrame(dict(task_id=dec["task_id"], n_contra=contradicted.astype(int),
                          n_conceal=withheld.astype(int)))
    return g.groupby("task_id", as_index=False).sum()


def main(sweep_path, out_path, data_root=None):
    root = Path(data_root)
    sweep = pd.read_csv(sweep_path)
    frames = []
    for run, cells in sweep.groupby("run"):
        t = cbrm.load_events(root / run / "events.jsonl")
        for cond, projection in zip(cells["condition"], cells["projection"]):
            c = counts(decisions(t, cond, projection=projection))
            c.insert(0, "condition", cond)
            c.insert(0, "run", run)
            frames.append(c)
    out = pd.concat(frames, ignore_index=True)
    tot = out.groupby(["run", "condition"])[["n_contra", "n_conceal"]].sum()
    ref = sweep.set_index(["run", "condition"])[["n_contra", "n_conceal"]]
    bad = (tot - ref.loc[tot.index]).abs().sum(axis=1) > 0
    if bad.any():
        raise SystemExit(f"per-task counts do not sum to the sweep on {int(bad.sum())} "
                         f"settings:\n{tot[bad].join(ref, rsuffix='_sweep')}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"{len(out)} (run, condition, task) rows over {len(tot)} settings; "
          f"totals match the sweep")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("out")
    ap.add_argument("--data-root", required=True, help="transcript tree (data/transcripts)")
    a = ap.parse_args()
    main(a.sweep, a.out, a.data_root)
