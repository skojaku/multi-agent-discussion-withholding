"""The evidence-level estimates Fig. 2 (f, g) draws, and the Wilson interval.

These functions are the part of the paper repository's ``fig_F4.py`` that
``fig_F2.py`` and ``fig_B3.py`` imported; the rest of that script draws figures
the paper does not include and is not part of this repository. The code is
unchanged apart from where the files are read from.

* ``du_estimate`` -- evidence pooling rate D (``ev_D``) and utilization rate U
  (``ev_U``) of one model at one arm, with 95 % Wilson intervals (F2 f);
* ``vote_accuracy`` -- the round-0 majority vote, the dashed contour in (f);
* ``tabled_summary`` -- what each model puts on the table, backing its own
  answer or the correct one, with cluster-bootstrap intervals (F2 g).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

MODELS = fs.MODELS
# Two equal-accuracy hyperbolae, not three, and none of them through the data.
# 0.15 and 0.45 bracket every plotted estimate (A = D U runs 0.22 .. 0.33) and
# cross U = 1 clear of every mark.
ISO_LEVELS = [0.15, 0.45]

NOTES: list[str] = []


def note(msg: str) -> None:
    NOTES.append(msg)


# ------------------------------------------------------------------- data loaders
def params(model: str) -> pd.DataFrame:
    return pd.read_csv(fs.MODEL_DIR[model] / "cbrm_params.csv").set_index("condition")


def vote_accuracy() -> tuple[float, dict[str, float]]:
    per = {}
    for m in MODELS:
        acc = pd.read_csv(fs.MODEL_DIR[m] / "hp_accuracy.csv")
        acc = acc[acc["k_informed"].astype(str) == "all"]
        per[m] = float(acc["vote_accuracy"].mean())
    return float(np.mean(list(per.values()))), per


# ------------------------------------------------ intervals
# A per-TASK binary rate (accuracy, sufficiency D, use U) -> Wilson interval on
# the task counts; a rate counted over agent turns (what gets tabled) -> cluster
# bootstrap over tasks, because the N turns of one task are not N draws.
Z95 = 1.959963984540054


def wilson(k: float, n: float, z: float = Z95) -> tuple[float, float]:
    """95 % Wilson score interval for ``k`` successes in ``n`` tasks."""
    if not n or not np.isfinite(n) or n <= 0:
        return (np.nan, np.nan)
    k, n = float(k), float(n)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def du_estimate(model: str, cond: str) -> tuple[float, tuple, float, tuple]:
    """D and U at one condition, each with its 95 % Wilson interval.

    ``ev_D`` is a mean over tasks and ``ev_U`` a mean over the sufficient ones,
    so the counts are recovered exactly from the published rates and n_tasks;
    the recovery is checked rather than assumed.
    """
    p = params(model)
    n = float(p.loc[cond, "n_tasks"])
    D = float(p.loc[cond, "ev_D"])
    U = float(p.loc[cond, "ev_U"])
    k_d, k_u = D * n, U * D * n
    if max(abs(k_d - round(k_d)), abs(k_u - round(k_u))) > 1e-6:
        note("(d) WARNING: %s %s does not resolve to whole task counts "
             "(D*n = %.4f, U*D*n = %.4f); interval widths are approximate"
             % (model, cond, k_d, k_u))
    return D, wilson(round(k_d), n), U, wilson(round(k_u), round(k_d))


# ------------------------------------------------------- what gets tabled, per task
def _load_events(path: Path) -> list[dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "agent" not in r and "seat" in r:
            r["agent"] = r["seat"]
        rows[(r["task_id"], r["agent"], r["round"], r["branch"])] = r
    return list(rows.values())


_SHARE_CACHE: dict[str, tuple] = {}


def share_by_task(model: str):
    """Per (task, condition) counts of tabled findings and which side they back.

    Same rule as ``compare_models.read_run``: every turn after round 0 that
    named a finding is one tabled finding; it backs the agent's own current
    answer if it is about that answer, and the correct side if it is about the
    truth. Kept per task so the rate can be bootstrapped over tasks. Only the
    arms every model was run at enter, so each model's rate is averaged over the
    same series.
    """
    if model in _SHARE_CACHE:
        return _SHARE_CACHE[model]
    rows = _load_events(fs.MODEL_EVENTS_DIR[model] / "events.jsonl")
    tasks = sorted({r["task_id"] for r in rows})
    ti = {t: i for i, t in enumerate(tasks)}
    ci = {c: j for j, c in enumerate(fs.RAW_ARMS)}
    shape = (len(tasks), len(fs.RAW_ARMS))
    n = np.zeros(shape)
    own = np.zeros(shape)
    truth = np.zeros(shape)
    for r in rows:
        if r["round"] == 0 or not r.get("share_about") or r["branch"] not in ci:
            continue
        i, j = ti[r["task_id"]], ci[r["branch"]]
        n[i, j] += 1
        own[i, j] += int(r["share_about"] == r["private"])
        truth[i, j] += int(r["share_about"] == r["answer"])
    _SHARE_CACHE[model] = (n, own, truth)
    return _SHARE_CACHE[model]


def share_stats(model: str, n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean over the instruction series of the per-condition rate, and its interval."""
    n, own, truth = share_by_task(model)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.shape[0], size=(n_boot, n.shape[0]))
    den = n[idx].sum(1)
    out = {}
    for key, k in (("own", own), ("truth", truth)):
        point = float(np.mean(k.sum(0) / n.sum(0)))
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = np.mean(k[idx].sum(1) / den, axis=1)
        lo, hi = np.percentile(draws[np.isfinite(draws)], [2.5, 97.5])
        out[key] = (point, float(lo), float(hi))
    return out


def tabled_summary():
    """The models ordered by tally accuracy, that accuracy, and what each tabled.

    Returned rather than recomputed in the caption, so the panel and the sentence
    under it cannot drift apart.
    """
    df = pd.read_csv(fs.MODEL_COMPARISON)
    df = df[df["model"].isin(MODELS)]
    missing = [m for m in MODELS if m not in set(df["model"])]
    if missing:
        raise SystemExit("model_comparison.csv has no rows for %s; rerun the "
                         "compare_models rule" % ", ".join(missing))
    tally = df.groupby("model")["count_correct"].mean()
    order = list(tally.sort_values().index)
    return order, tally, {m: share_stats(m) for m in order}
