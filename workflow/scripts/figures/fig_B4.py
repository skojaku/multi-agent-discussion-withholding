"""B4 -- where the evidence goes: evidence pooling and utilization, per model.

Appendix B. Was F2(f) until review moved it out of the main figure
to stand on its own. One panel: what reached the table (the evidence pooling
rate D) against what was used of it (the evidence utilization rate U), one mark
per (model, instruction arm), with the equal-accuracy hyperbolae D x U behind.
9/24: no longer a plane. 13 of its 27 points sat at a utilization of exactly
1 and hid one another, so the two rates are two columns on one row per model
(``du_rows``), the three instructions nudged apart on the row. F3(f) draws the
same panel at instruction A alone, through the same function.

Run: python3 fig_B4.py
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import figstyle as fs
import fig_F2 as f2
import evidence as f4

FIG_W, FIG_H = 3.2, 2.5
PAN_X, PAN_Y, PAN_W, PAN_H = 0.62, 0.40, 2.50, 2.02
DU_DY = {"A": -0.24, "B": 0.0, "C": 0.24}   # the three instructions on a row
KEY_STRIP = 1.0                             # rows over the top row, for the key


def du_rows(ax_d, ax_u, arms=tuple(fs.ARMS), strip: float = KEY_STRIP,
            names: bool = True, xlabel_size=None, d_ticks=(0.2, 0.4, 0.6),
            u_lim=(0.25, 1.04), u_ticks=(0.4, 1),
            labels=("pooling rate", "utilization rate")) -> list:
    """Pooling rate on ``ax_d`` and utilization rate on ``ax_u``, one row per
    model in F3(e)'s order; ``arms`` picks the instructions drawn. Returns the
    row order."""
    order = f4.tabled_summary()[0]
    du = f2.du_points()
    dy = DU_DY if len(arms) > 1 else {a: 0.0 for a in arms}
    for i, m in enumerate(order):
        for arm in arms:
            D, (dlo, dhi), U, (ulo, uhi) = du[(m, arm)]
            style = dict(marker="o", ls="none", zorder=3.2, **f2.mark_style(arm))
            if len(arms) > 1:
                style["ms"] = style["ms"] * 0.8
            for ax, v, lo, hi in ((ax_d, D, dlo, dhi), (ax_u, U, ulo, uhi)):
                fs.whisker(ax, [i + dy[arm]], [lo], [hi], fs.INK_MID,
                           orient="h", lw=0.6, zorder=2.4)
                ax.plot([v], [i + dy[arm]], **style)
    for ax in (ax_d, ax_u):
        ax.set_ylim(len(order) - 1 + 0.6, -0.6 - strip)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([])
        ax.spines["left"].set_bounds(0, len(order) - 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2, pad=1)
    if names:
        ax_d.set_yticklabels([fs.MODEL_SHORT[m] for m in order])
    ax_d.set_xlim(0.15, 0.75)
    fs.ticks01(ax_d, "x", d_ticks)
    ax_u.set_xlim(*u_lim)
    fs.ticks01(ax_u, "x", u_ticks)
    kw = dict(labelpad=1, fontsize=xlabel_size or fs.FS_AXIS)
    ax_d.set_xlabel(labels[0], **kw)
    ax_u.set_xlabel(labels[1], **kw)
    if len(arms) > 1:
        trans = ax_d.get_yaxis_transform()
        yk = -0.6 - strip + 0.45
        for k, arm in enumerate(arms):
            xk = 0.04 + 0.26 * k
            ax_d.plot([xk], [yk], transform=trans, marker="o", ls="none",
                      clip_on=False, **f2.mark_style(arm))
            ax_d.annotate(arm, (xk, yk), xycoords=trans, xytext=(3, 0),
                          textcoords="offset points", fontsize=fs.FS_SMALL,
                          color=fs.INK2, va="center")
    return order


def caption(du: dict) -> None:
    fs.write_caption("B4", [
        "Where the evidence goes, on the hidden profile task: the evidence"
        " pooling rate (left), the share of rounds in which the shared evidence"
        " sufficed to determine the correct answer, and the evidence utilization"
        " rate (right), the share of those rounds in which the team then chose"
        " it. One row per model, in the order of Figure 3e, with the three"
        " instructions on each row (A blue, B grey, C open); bars are 95 %"
        " Wilson intervals.",
    ])


def make():
    fs.apply()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gap = 0.22
    w = (PAN_W - gap) / 2
    ax_d = fig.add_axes([PAN_X / FIG_W, PAN_Y / FIG_H, w / FIG_W, PAN_H / FIG_H])
    ax_u = fig.add_axes([(PAN_X + w + gap) / FIG_W, PAN_Y / FIG_H, w / FIG_W,
                         PAN_H / FIG_H])
    du_rows(ax_d, ax_u)
    caption(f2.du_points())
    for line in f2.NOTES:
        print(line)
    return fig


if __name__ == "__main__":
    fig = make()
    f2.check_layout(fig)
    fs.save(fig, "B4")
