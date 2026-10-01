"""B3 -- what the reasoning switch does, in four quantities.

Appendix B. Models with a second reasoning arm (the config's `ablation`
entries of kind "switch") were run twice on the hidden profile, on the same
tasks and instructions: once at the reasoning setting the paper reports and
once with reasoning off (or the lowest setting the API accepts). Each row is
one model; the two marks on it are the two settings. Top to bottom:
withholding c-hat, internalization a-hat, net correction gamma-hat, and the
gain from discussion.

c-hat and a-hat pool over the instructions from counts and carry a Wilson
interval; gamma-hat is the mean over the instructions with their range as the
bar; the gain pools the accuracy. Pairs of kind "effort" (more reasoning
against less) are named in the caption only. Everything is read from sweep.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
from evidence import wilson  # noqa: E402   the one Wilson in the figure scripts

SWEEP = fs.SWEEP
PROBE = fs.PROBE            # as everywhere else: the private answer in its own call

# model -> (the arm the paper draws, the second arm, what the second arm's
# reasoning setting was), from each model's `ablation` entry in the config with
# kind "switch" (events.meta.json, `reasoning_effort`).
PAIRS = {m: (fs.MODEL_HP_RUN[m], ab["run"], ab["effort_ablation"])
         for m, ab in fs.MODEL_ABLATION.items() if ab["kind"] == "switch"}
# (column, pooling, axis name). "count" pools from the estimator's own numerator
# and denominator; "mean" averages the arms and shows their range.
QUANTITIES = [
    ("c", "count", ("n_conceal", "n_contra"), r"withholding $\hat{c}$"),
    ("a", "count", (None, "n_conceal"), r"internalization $\hat{a}$"),
    ("gamma", "mean", None, r"net correction $\hat{\gamma}$"),
    ("gain", "gain", None, "gain from discussion"),
]
# The pairs that are not a switch (more reasoning against less), kept for the
# caption: (model, main run, main effort, second run, second effort).
EFFORT_PAIRS = [(m, fs.MODEL_HP_RUN[m], ab["effort_main"], ab["run"], ab["effort_ablation"])
                for m, ab in fs.MODEL_ABLATION.items() if ab["kind"] == "effort"]

# review: no colour, and the two settings of one model on ONE row.
# Reasoning on is the filled ink circle, off the open one; both whiskers are
# ink, the off one a step lighter so the two can still be told apart where
# they overlap on the shared row.
ON = dict(mfc=fs.INK, mec=fs.INK, mew=0.7)
OFF = dict(mfc=fs.SURFACE, mec=fs.INK, mew=0.7)
ON_WHISKER, OFF_WHISKER = fs.INK, fs.INK_MID
MS = 3.2                    # F2's marker size: these panels are F2(f)-(i) now
# Standalone render: the four panels stacked in one column (after review),
# the model names on every panel because each panel has its own x.
# 2 x 2, full width (9/24); the row of four stacked panels was 2.75 x 4.60.
FIG_W, FIG_H = 5.5, 3.10
PAN_X, PAN_X2, PAN_W = 0.72, 3.18, 2.12   # left and right columns
PAN_BOTTOM, PAN_TOP = 0.34, 2.92
PAN_GAP = 0.48                  # x tick labels + x name + the next letter
LETTER_DX = 0.62                # a letter over the model names
LETTER_DX_R = 0.20              # a letter over a column with no names
ROW_FOOT = 0.6                  # rows of air under the bottom row
KEY_STRIP = 1.15                # rows opened over the top row of the first panel

NOTES: list[str] = []


def note(msg: str) -> None:
    NOTES.append(msg)


# None pools the three arms (B3); an arm name restricts every quantity to that
# arm alone (F2(e)-(g) set it to "A", 9/24).
ARM_ONLY: str | None = None


def _arms(d: pd.DataFrame, run: str) -> pd.DataFrame:
    g = d[d.run == run]
    missing = sorted(set(fs.ARMS) - set(g.condition))
    if missing:
        raise SystemExit(f"sweep.csv has no {missing} cell for {run}")
    if ARM_ONLY is not None:
        g = g[g.condition == ARM_ONLY]
    return g


def pooled_gain(d: pd.DataFrame, run: str) -> dict:
    """The gain of one run over its three instruction arms, as one number.

    Pooled rather than averaged: the arms carry the same 120 tasks each, so the
    weights are equal, but pooling is what makes the interval a Wilson interval
    on 360 items instead of three intervals with nothing joining them. The
    interval is the accuracy's, carried onto the gain -- the vote it is measured
    against is the same agents on the same items, so it moves the point and not
    the width, which is the convention F2(e) already uses.
    """
    g = _arms(d, run)
    n = float(g.n_tasks.sum())
    acc = float((g.final_accuracy * g.n_tasks).sum() / n)
    vote = float((g.vote_accuracy * g.n_tasks).sum() / n)
    lo, hi = wilson(acc * n, n)
    return dict(value=acc - vote, lo=lo - vote, hi=hi - vote, n=int(n),
                per_arm={r.condition: float(r.interaction_gain)
                         for r in g.itertuples()})


def pooled_ratio(d: pd.DataFrame, run: str, col: str, den_col: str) -> dict:
    """A count ratio pooled over the arms, with a Wilson interval.

    ``c`` is n_conceal / n_contra and ``a`` is its own numerator over
    n_conceal; both reproduce to 1e-16 from those columns, so the three arms
    pool by adding numerators and denominators rather than by averaging three
    rates whose denominators differ.
    """
    g = _arms(d, run)
    den = float(g[den_col].sum())
    num = float((g[col] * g[den_col]).sum())
    if den <= 0:
        return dict(value=float("nan"), lo=float("nan"), hi=float("nan"), n=0,
                    per_arm={})
    lo, hi = wilson(num, den)
    return dict(value=num / den, lo=lo, hi=hi, n=int(den),
                per_arm={r.condition: float(getattr(r, col)) for r in g.itertuples()})


def arm_mean(d: pd.DataFrame, run: str, col: str) -> dict:
    """The mean over the arms, with their RANGE -- not an estimator interval.

    ``gamma`` nets two rates measured on different row sets, so it has no single
    numerator and denominator to pool. Averaging the three arms is the honest
    summary; showing their spread as the bar says what the bar is, and the
    caption repeats it so the bar cannot be read as an error bar.
    """
    g = _arms(d, run)
    v = g[col].astype(float)
    if len(g) == 1 and f"{col}_lo" in g:
        # one arm: its own interval, not a range over arms
        return dict(value=float(v.iloc[0]), lo=float(g[f"{col}_lo"].iloc[0]),
                    hi=float(g[f"{col}_hi"].iloc[0]), n=int(g.n_tasks.sum()),
                    per_arm={r.condition: float(getattr(r, col))
                             for r in g.itertuples()})
    return dict(value=float(v.mean()), lo=float(v.min()), hi=float(v.max()),
                n=int(g.n_tasks.sum()),
                per_arm={r.condition: float(getattr(r, col)) for r in g.itertuples()})


def measure(d: pd.DataFrame, run: str) -> dict:
    """Every quantity B3 draws, for one run."""
    out = {}
    for col, how, cols, _ in QUANTITIES:
        if how == "gain":
            out[col] = pooled_gain(d, run)
        elif how == "count":
            out[col] = pooled_ratio(d, run, col, cols[1])
        else:
            out[col] = arm_mean(d, run, col)
    return out


def load() -> dict:
    d = fs.arms_only(pd.read_csv(SWEEP))
    d = d[d["probe"].fillna("unknown").astype(str) == PROBE]
    out = {}
    for model, (main, off, _) in PAIRS.items():
        out[model] = {"on": measure(d, main), "off": measure(d, off)}
    out["_effort"] = {m: {"main": measure(d, main), "second": measure(d, second)}
                      for m, main, _, second, _ in EFFORT_PAIRS}
    return out


def load_main() -> dict:
    """Every model of fs.MODELS at its main setting only, as ``{"on": ...}``.

    F2(e)-(g) draw these; the reasoning-off arms are B3's (9/24).
    """
    d = fs.arms_only(pd.read_csv(SWEEP))
    d = d[d["probe"].fillna("unknown").astype(str) == PROBE]
    return {m: {"on": measure(d, str(fs.MODEL_DIR[m].relative_to(fs.DATA)))}
            for m in fs.MODELS}


def key(ax, x_frac: float, row: float) -> None:
    """Two entries in the strip the ylim opens over the top row.

    A key, not a direct label: both series are circles of the same size, and a
    label beside each would have to name the row it sits on as well. Placed
    through the y-axis transform -- x in axes fraction, y in ROWS -- so it
    holds its distance from the top row whatever the x range turns out to be.
    """
    fig = ax.figure
    fig.canvas.draw()
    t = ax.text(0, 0, "reasoning on", fontsize=fs.FS_SMALL)
    word = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    width = ax.get_position().width * fig.get_size_inches()[0]
    gap = (word + 0.16) / width           # marker, 3 pt, the word, then air
    for i, (style, text) in enumerate(((ON, "reasoning on"), (OFF, "off"))):
        x = x_frac + i * gap
        ax.plot([x], [row], transform=ax.get_yaxis_transform(), marker="o",
                ms=MS, ls="none", zorder=4, clip_on=False, **style)
        ax.annotate(text, (x, row), xycoords=ax.get_yaxis_transform(),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=fs.FS_SMALL, color=fs.INK2, va="center", ha="left")


def panel(ax, data: dict, order: list, col: str, how: str, name: str,
          strip: float = 0.0, off_mark=None, on_style=None) -> None:
    """One quantity, one row per model, the two settings on that model's row.

    ``strip`` opens that many rows over the top row, for the key.
    """
    for i, m in enumerate(order):
        # F2(e)-(g) pass the main setting alone ("on" only, load_main)
        drawn = [(data[m][s_][col], w, st) for s_, w, st in
                 (("off", OFF_WHISKER, OFF), ("on", ON_WHISKER, ON))
                 if s_ in data[m]]
        if not all(np.isfinite(a["value"]) for a, _, _ in drawn):
            continue
        # Both settings on the model's own row (after review); the fill says
        # which is which. No connector: on one row it ran along the whiskers
        # and could not be told from them.
        for arm, whisk, _ in drawn:
            fs.whisker(ax, [i], [arm["lo"]], [arm["hi"]], whisk,
                       orient="h", zorder=2.4)
        for arm, _, style in drawn:
            if off_mark is not None and style is OFF:
                off_mark(ax, arm["value"], i)   # F2 stripes the off arm
                continue
            if on_style is not None and style is ON:
                style = on_style             # F2 draws the on arm its own way
            style = dict(style)
            ax.plot([arm["value"]], [i], marker="o", ms=style.pop("ms", MS),
                    ls="none", zorder=3.2, **style)
        if len(drawn) == 2:
            on, off = data[m]["on"][col], data[m]["off"][col]
            note("%-9s %-6s on %+.3f [%+.3f, %+.3f]   off %+.3f [%+.3f, %+.3f]"
                 "   shift %+.3f"
                 % (fs.MODEL_SHORT[m], col, on["value"], on["lo"], on["hi"],
                    off["value"], off["lo"], off["hi"], on["value"] - off["value"]))
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([fs.MODEL_SHORT[m] for m in order])
    ax.set_ylim(len(order) - 1 + ROW_FOOT, -ROW_FOOT - strip)
    ax.spines["left"].set_bounds(0, len(order) - 1)

    vals = [data[m][s_][col][f] for m in order for s_ in ("on", "off")
            if s_ in data[m]
            for f in ("lo", "hi") if np.isfinite(data[m][s_][col][f])]
    # Zero is kept in range only where its sign is the reading (net
    # correction, gain); the two rates are fitted to their data.
    zero = [0.0] if col in ("gamma", "gain") else []
    lo, hi = min(vals + zero), max(vals + zero)
    pad = 0.06 * (hi - lo) if hi > lo else 0.05
    ax.set_xlim(lo - pad, hi + pad)
    step = 0.5 if hi - lo > 1.2 else 0.2 if hi - lo > 0.45 else 0.1
    if not zero:
        step = 0.2 if hi - lo > 0.6 else 0.1 if hi - lo > 0.25 else 0.05
    ticks = [0.0 if abs(t) < 1e-9 else round(t, 10)
             for t in np.arange(-2, 2.0001, step) if lo - pad <= t <= hi + pad]
    # more than three: coarsen the step until three or fewer remain, so the
    # ticks stay evenly spaced (picking first/middle/last of 0..0.6 by 0.2
    # printed 0, 0.4, 0.6)
    for coarse in (0.25, 0.5, 1.0):
        if len(ticks) <= 3:
            break
        if coarse <= step:
            continue
        ticks = [0.0 if abs(t) < 1e-9 else round(t, 10)
                 for t in np.arange(-2, 2.0001, coarse) if lo - pad <= t <= hi + pad]
    ax.set_xticks(ticks)
    ax.set_xticklabels([fs.fmt_tick(t) for t in ticks])
    if zero:
        # zero: nothing corrected, nothing gained; over the rows only, so it
        # stays out of a key strip opened above them
        fs.reference(ax, 0.0, "v", lo=-ROW_FOOT, hi=len(order) - 1 + ROW_FOOT)
    ax.set_xlabel(name, labelpad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", length=2, pad=1.5)


def row_order(data: dict) -> list:
    """Ordered by what deliberating was worth with reasoning on: the rows the
    reader is asked to compare are the same rows in all four panels, and the
    ordering variable is the one the last panel is about."""
    return sorted((m for m in data if not m.startswith("_")),
                  key=lambda m: data[m]["on"]["gain"]["value"])


def summary(data: dict, order: list) -> str:
    """The B3 reading, letter-free, for F2(f)-(i)'s caption."""
    def shift(m, col):
        return data[m]["on"][col]["value"] - data[m]["off"][col]["value"]

    mean_shift = {col: np.mean([shift(m, col) for m in order])
                  for col, _, _, _ in QUANTITIES}
    n_a_up = sum(1 for m in order if shift(m, "a") > 1e-9)
    n_gain_down = sum(1 for m in order if shift(m, "gain") < -1e-9)
    switch = [m for m in order if PAIRS[m][2] == "off"]
    n_items = data[order[0]]["on"]["gain"]["n"]
    return " ".join([
        f"{len(order)} models run twice on the same {n_items // len(fs.ARMS)} tasks at each of the",
        "instructions A, B and C, once at the reasoning setting the paper reports (filled)",
        "and once with reasoning off (open), both on the model's own row;",
        "top to bottom, withholding, internalization, net correction and the",
        "gain from discussion, pooled over the three instructions. Rows are ordered by",
        "the gain with reasoning on. Averaged over the rows, turning reasoning",
        "on moves withholding by %+.2f, internalization by %+.2f, net"
        % (mean_shift["c"], mean_shift["a"]),
        "correction by %+.2f and the gain by %+.2f; internalization is higher"
        % (mean_shift["gamma"], mean_shift["gain"]),
        f"with reasoning in {n_a_up} of the {len(order)} rows and the gain is",
        f"lower in {n_gain_down}. Bars are 95 % Wilson intervals of the pooled",
        f"ratio for withholding, internalization and the gain ({n_items} items),",
        "and the range over the three instructions for net correction, which",
        "nets two rates measured on different row sets and so cannot pool.",
        "%s were re-run with reasoning off%s." % (
            _join([fs.MODEL_SHORT[m] for m in switch]),
            "".join("; %s's lower setting is %s" % (fs.MODEL_SHORT[m], PAIRS[m][2])
                    for m in order if PAIRS[m][2] != "off")),
    ])


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def caption(data: dict, order: list) -> None:
    def shift(m, col):
        on = data[m]["on"][col]["value"]
        off = data[m]["off"][col]["value"]
        return on - off

    switch = [m for m in order if PAIRS[m][2] == "off"]
    n_items = data[order[0]]["on"]["gain"]["n"]

    def row_text(col, fmt="%+.2f"):
        return "; ".join(
            ("%s " + fmt + " on against " + fmt + " off")
            % (fs.MODEL_SHORT[m], data[m]["on"][col]["value"],
               data[m]["off"][col]["value"]) for m in order)

    def tally(col, want_positive=True):
        k = sum(1 for m in order
                if (shift(m, col) > 0) == want_positive and abs(shift(m, col)) > 1e-9)
        return k

    mean_shift = {col: np.mean([shift(m, col) for m in order])
                  for col, _, _, _ in QUANTITIES}
    eff = data["_effort"]
    fs.write_caption("B3", [
        "What the reasoning switch is worth, on the hidden profile. One row is",
        "one model and the two circles are the same model at two reasoning",
        "settings, filled with reasoning on and open with it off, run on the",
        "same %d tasks at each of the instructions %s; the line"
        % (n_items // len(fs.ARMS), ", ".join(fs.ARMS)),
        "between them is the shift, and the rows are ordered by (d) with",
        "reasoning on. (a) Withholding, (b) internalisation, (c) net correction",
        "and (d) the gain from discussion -- collective accuracy after discussion",
        "minus the round-0 majority vote of the same agents on the same items.",
        "Averaged over the %d rows, turning reasoning ON moves withholding by"
        % len(order),
        "%+.3f, internalisation by %+.3f, net correction by %+.3f and the gain by"
        % (mean_shift["c"], mean_shift["a"], mean_shift["gamma"]),
        "%+.3f: what a seat SAYS is unchanged, what it comes to BELIEVE is not,"
        % mean_shift["gain"],
        "and the team pays for it -- internalisation is higher with reasoning in",
        "%d of the %d rows and the gain is lower in %d."
        % (tally("a"), len(order), tally("gain", want_positive=False)),
        "Reading the rows: (a) %s. (b) %s. (c) %s. (d) %s."
        % (row_text("c"), row_text("a"), row_text("gamma"), row_text("gain")),
        "In (a) and (b) the estimator is a count ratio -- c-hat is",
        "n\\_conceal/n\\_contra and a-hat is its own numerator over n\\_conceal --",
        "so the three instructions pool by adding numerators and denominators, and the bar",
        "is the 95 \\%% Wilson interval of the pooled ratio. In (d) the same",
        "pooling is applied to the accuracy, on %d items. In (c) it cannot be:"
        % n_items,
        "gamma-hat nets two rates measured on different row sets, so the point is",
        "the mean over the three instructions and THE BAR IS THEIR RANGE -- spread between",
        "instructions, not estimator error.",
        "The setting is read from each run's own record (events.meta.json): %s."
        % "; ".join("%s %s" % (fs.MODEL_SHORT[m], PAIRS[m][2]) for m in order),
        " ".join("%s is not drawn: its second run is %s effort against a main run"
                 " at %s, more reasoning against less rather than on against off"
                 " (gain %+.2f at %s against %+.2f at %s)."
                 % (fs.MODEL_SHORT[m], e2, e1, eff[m]["main"]["gain"]["value"], e1,
                    eff[m]["second"]["gain"]["value"], e2)
                 for m, _, e1, _, e2 in EFFORT_PAIRS),
    ])




def make():
    """2 x 2 (9/24): (a) withholding and (b) internalization on top, (c) net
    correction and (d) gain below. The two panels of a row share their model
    rows, so only the left one names them; the key sits over (a) and the top
    row opens the same strip on both sides, so their rows line up."""
    fs.apply()
    data = load()
    order = row_order(data)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    n = len(order)
    rows_top = n - 1 + 2 * ROW_FOOT + KEY_STRIP
    rows_bot = n - 1 + 2 * ROW_FOOT
    pitch = (PAN_TOP - PAN_BOTTOM - PAN_GAP) / (rows_top + rows_bot)
    h_top, h_bot = rows_top * pitch, rows_bot * pitch
    y_top = PAN_TOP - h_top
    boxes = [(PAN_X, y_top, h_top), (PAN_X2, y_top, h_top),
             (PAN_X, PAN_BOTTOM, h_bot), (PAN_X2, PAN_BOTTOM, h_bot)]
    for j, ((col, how, _, name), (x0, y0, h)) in enumerate(zip(QUANTITIES, boxes)):
        ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, PAN_W / FIG_W, h / FIG_H])
        panel(ax, data, order, col, how, name,
              strip=KEY_STRIP if j < 2 else 0.0)
        if x0 == PAN_X2:
            ax.set_yticklabels([])       # the left panel of the row names them
        dx = LETTER_DX if x0 == PAN_X else LETTER_DX_R
        fig.text((x0 - dx) / FIG_W, (y0 + h + 0.15) / FIG_H,
                 f"({chr(97 + j)})", ha="left", va="top", fontsize=fs.FS_AXIS,
                 fontweight="bold", color=fs.INK)
        if j == 0:
            key(ax, 0.04, -ROW_FOOT - KEY_STRIP / 2 + 0.1)
    caption(data, order)
    return fig


if __name__ == "__main__":
    fig = make()
    for out in fs.save(fig, "B3"):
        print(out)
    print()
    for line in NOTES:
        print(line)
