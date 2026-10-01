"""F2 -- the task, what its logs measure on it, and the mechanism underneath.

Seven panels in three columns, 5.5 x 3.6 in (the 2026-09-15 sketch):

  (a) the hidden-profile task in F1's notation: the shared briefing every agent
      reads for free, and each agent as two concentric rings -- outer the answer
      it states in public, inner the belief it holds in private -- with its own
      memos inside its inner ring, where the rest of the team cannot see them.
  (b) one agent carried through the three rounds a run has, which is what says
      which quantity carries which subscript: the belief entering round t reads
      t-1 and everything the round produces reads t.
  (c) the withholding rate c-hat against the instruction arm, one series per
      model.
  (d) c*, the threshold that rate has to be read against, on the same arms: it
      is a property of the setting, not a constant, so a rate is high or low
      only against its own value of it.
  (e) the distance between the two against what deliberating was worth -- the
      collective accuracy minus the round-0 vote of the same agents. The claim
      of 4.2: the DISTANCE orders the gain, not the rate.
  (f) where the evidence goes: what reached the table (D) against what was used
      of it (U), with the equal-accuracy hyperbolae behind.
  (g) what a model tables -- the correct candidate, or its own current answer.

Two channels carry every mark and neither is a hue: the SHAPE is the model and
the FILL is the instruction arm: blue at A, filled ink at B, open at C. The figure is
greyscale apart from the trend line of (e), which is the one claim a reader
could otherwise take or leave.

Nothing in the panels but data, axis names, tick labels, panel letters and the
two keys; every sentence is in figs/captions/F2.tex, written from the same
tables the panels draw.

Run: python3 fig_F2.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs
import evidence as f4   # (e) and appendix B4: the evidence-level estimates
import fig_B3 as b3     # (f)-(i) are fig_B3's panels
from scipy.stats import spearmanr

# ------------------------------------------------------------------- sources
ITEMS = fs.ITEMS_DIR / "hidden_profile.jsonl"

SCHEMATIC_K = 2                # the k the (a) diagram draws; p = k/N by design
C_STAR_COND = "B"              # thresholds are read off the no-instruction run

# The withholding column this figure plots, in ONE place: panel (b)'s top row,
# panel (c)'s x axis, the axis names and the caption wording all follow it, so
# swapping the estimator is this one line and nothing else.
#
#   "c_strict" -- the strict estimator: of the rows where the visible majority
#                 contradicts the answer the agent holds now, the share where
#                 it stated the majority against its own belief (a real
#                 public/private gap is required).
#   "c"        -- the legacy estimator: of the rows where the majority
#                 contradicts the agent's previous answer, the share that
#                 stated the majority, gap or no gap.
#
# NOTE cbrm's own ``margin`` and ``verdict`` columns are computed from "c", not
# from "c_strict" (cbrm/estimators.py: margin = c - c*), so this constant
# changes what the figure draws, not what those columns mean.  Both columns
# carry _lo/_hi in cbrm_params.csv.  The posterior file must name the same
# quantity: fit_F2b_posterior.py's QUANTITIES has to be refitted to match.
C_COL = "c_post"     # posterior of App. B.2; "c"/"c_strict" are the counting
                     # estimators behind it.

C_WORD = "withholding"
C_SYM = "$\\hat{c}$"
C_NAME = f"{C_WORD}  {C_SYM}"
C_DEF = {
    "c_strict": ("the share of the rows where the visible majority contradicts"
                 " the answer a seat holds privately and it states the majority"
                 " anyway, against that belief"),
    "c": ("the share of the rows where the visible majority contradicts an"
          " agent's previous answer and it states the majority"),
}[C_COL.removesuffix("_post")]

# (column, word, symbol). The row is NAMED BY ITS SYMBOL ALONE (9/14): the words
# were set over the symbol in a two-line rotated label, and three of them --
# "internalization" is fifteen characters rotated into a 0.7 in row -- were the
# tallest thing in the gutter and still said less than the symbol the text uses.
# The word is kept here because the caption spells each rate out from it.
ROWS = [(C_COL, C_WORD, C_SYM),
        ("a_post", "internalization", "$\\hat{a}$"),
        ("gamma_post", "net correction", "$\\hat{\\gamma}$")]
ROW_NAME_DY = [0.0, 0.0, 0.0]      # inches; a one-glyph name needs no nudging

# Diagnostics printed under a render, never drawn on it: what each panel read,
# so a number in the caption can be traced to the table it came from. Named
# `diag` and not `note`, which in this module draws a lettered finding box.
NOTES: list[str] = []


def diag(msg: str) -> None:
    NOTES.append(msg)


# ------------------------------------------------------------------ geometry
# Three columns, 5.5 in wide (after review):
#
#   left    the OBJECT -- (a) the task, (b) one agent through three rounds --
#           drawn at a fixed scale in inches, so each axes is exactly as tall as
#           the drawing it holds;
#   middle  what the logs MEASURE: (c) the withholding rate, (d) the distance
#           from the threshold against what deliberating was worth, (e) what a
#           model puts on the table;
#   right   the reasoning switch, (f)-(i): fig_B3's four panels, stacked.
#
# The threshold c* per model is drawn beside the rate on its rows (9/24; the
# appendix Figure B5 that held it is gone), and the evidence pooling/utilization
# plane (was (f)) is appendix Figure B4.
FIG_W, FIG_H = 5.5, 3.50 + fs.MODEL_KEY_H
TOP = FIG_H - 0.18             # the ceiling of every measured panel
BOTTOM = 0.32                  # the floor: x tick labels and an x name
G_X0 = 4.12                    # (g), under (e)-(f) but narrower
D_KEY_H = 0.0                  # (d)'s model key is inside the panel now
PANEL_GAP = 0.42               # between two stacked panels: the upper one's x
                               # tick labels and x name, then the lower one's
                               # letter (0.13 in over its axes)

# (a) and (b) are drawn in the same inch-scale coordinate system, the one the
# rings, boxes and arrows were tuned in (A_* and D_* below); each axes gets the
# y window its tier occupies and an equal aspect makes one data unit one inch.
A_X0, A_W, A_H = 0.06, 1.86, 3.20
A_CHAIN_H = 1.15               # one agent, three rounds
# (a) cannot go below 1.05 in -- the seat ring in draw_a() sits at A_H - 0.78
# with a 0.27 in radius, so its lower edge is A_H - 1.05 -- and 1.15 in clears
# that with margin.
ROW_H = 1.15
A_TASK_Y, A_TASK_H = TOP - ROW_H, ROW_H      # the briefing and the three agents

B_X0, B_W = 1.40, 0.72         # (c), (d), (e); the gutter to the left holds the
                               # model names of (c) and (e)
D_H = 0.95                     # (d), the scatter; its two-line rotated y name
                               # measures 0.67 in end to end
C_X0, C_W_HALF, EF_GAP = 3.56, 0.82, 0.24   # (e) and (f) side by side;
                               # (g) spans both
D_X0, D_W = 1.40, 2.00         # (d), the main panel, full height
TOP_H = 1.20                   # the upper tier: (a), (e), (f)
CS_GAP = 0.12                  # (b) to (c), which shares its rows
NAMES_DX = 0.58                # a letter over a gutter of model names
C_W = 2 * C_W_HALF + EF_GAP         # (f)-(i); the gutter holds six model names

POOL_STRIP = 1.0               # rows under (e)'s key for its pooling note
LETTER_DX = 0.22               # a panel letter, left of its axes
MS = 3.2                       # marker size, pt
ARM_DODGE = 0.10               # kept for fig_B2, which dodges models on an arm

def ax_size(ax) -> tuple:
    """An axes' width and height in inches."""
    box = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    return box.width * fw, box.height * fh


def add_ax(fig, x, y, w, h, **kw):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def row_y(i: int, h: float, gap: float) -> float:
    """Bottom of row ``i`` (0 = top) of a three-row stack of height ``h``."""
    return TOP - (i + 1) * h - i * gap


def fig_text(fig, x, y, s, **kw):
    return fig.text(x / FIG_W, y / FIG_H, s, **kw)


# ---------------------------------------------------------------------- data
def load_params() -> pd.DataFrame:
    """The counting estimator, its 95 % interval and its threshold, per cell.

    The ``_lo``/``_hi`` columns are the cluster bootstrap over tasks that
    ``cbrm`` wrote beside every rate. They are kept because the counting
    estimator is an estimate: panel (b) may not draw it bare.
    """
    out = []
    for m in fs.MODELS:
        d = fs.arms_only(pd.read_csv(fs.MODEL_DIR[m] / "cbrm_params.csv"))
        d["model"] = m
        keep = (["model", "condition", "c_star_post", "c_star_post_lo", "c_star_post_hi",
                 "p_trapped", "n_agents", "T", "n_tasks"]
                + [f"{col}{s}" for col, _, _ in ROWS
                   for s in ("", "_lo", "_hi")])
        out.append(d[keep])
    d = pd.concat(out, ignore_index=True)
    missing = [(m, c) for m in fs.MODELS for c in fs.ARMS
               if not ((d.model == m) & (d.condition == c)).any()]
    if missing:
        raise SystemExit(f"cbrm_params.csv is missing {missing}")
    return d


def load_accuracy() -> pd.DataFrame:
    """Collective accuracy per model x condition x design accuracy p."""
    out = []
    for m in fs.MODELS:
        d = fs.arms_only(pd.read_csv(fs.MODEL_DIR[m] / "hp_accuracy.csv"))
        d = d[d.p_design.notna()].copy()
        d["model"] = m
        out.append(d[["model", "condition", "k_informed", "p_design", "accuracy",
                      "acc_lo", "acc_hi", "n_items"]])
    d = pd.concat(out, ignore_index=True)
    d["p_level"] = d.p_design.round(2)
    return d


def load_accuracy_all() -> pd.DataFrame:
    """Collective accuracy per model x condition, pooled over the design accuracies.

    The ``k_informed == "all"`` row of ``hp_accuracy.csv``: the score of the
    whole sweep at one instruction, which is the quantity 4.0 claims falls. The
    per-p rows stay in ``load_accuracy`` for F3(a).
    """
    out = []
    for m in fs.MODELS:
        d = fs.arms_only(pd.read_csv(fs.MODEL_DIR[m] / "hp_accuracy.csv"))
        d = d[d.k_informed.astype(str) == "all"].copy()
        d["model"] = m
        out.append(d[["model", "condition", "accuracy", "acc_lo", "acc_hi",
                      "vote_accuracy", "n_items"]])
    d = pd.concat(out, ignore_index=True)
    missing = [(m, c) for m in fs.MODELS for c in fs.ARMS
               if not ((d.model == m) & (d.condition == c)).any()]
    if missing:
        raise SystemExit(f'hp_accuracy.csv has no k_informed = "all" row for {missing}')
    return d


def main_probe_name() -> str:
    """How the runs this figure plots elicited the private answer.

    ``load_params`` keeps only the columns the panels draw, so the probe is read
    back from the three source tables. They must agree: a figure whose three
    models were probed differently would put three different quantities on one
    axis, and the assertion is the check.
    """
    probes = set()
    for m in fs.MODELS:
        d = pd.read_csv(fs.MODEL_DIR[m] / "cbrm_params.csv")
        probes |= set(d.get("probe", pd.Series(["unknown"])).astype(str))
    assert len(probes) == 1, f"the three runs were probed differently: {probes}"
    return probes.pop()


def load_task(k: int) -> dict:
    """The composition of one k-informed item, read from the item file itself.

    Every item with this ``k`` holds the same counts; they differ only in which
    seats are informed, which is drawn at random. The diagram takes the one
    whose informed seats are most evenly spread, so the schematic is not
    lopsided for a reason that carries no meaning.
    """
    items = [json.loads(line) for line in ITEMS.open()]
    items = [it for it in items if it["design"]["n_informed"] == k]
    if not items:
        raise SystemExit(f"no k = {k} item in {ITEMS}")
    n = items[0]["design"]["n_seats"]
    want = set(np.round(np.linspace(0, n - 1, k + 2)[1:-1]).astype(int))
    item = min(items, key=lambda it: set(it["design"]["solo_correct_seats"]) != want)
    strong = item["design"]["strong_label"]

    def kinds(facts):
        n_s = sum(f["about"] == strong for f in facts)
        return "S" * n_s + "W" * (len(facts) - n_s)

    seats = sorted(item["seat_facts"], key=int)
    solo = {str(s) for s in item["design"]["solo_correct_seats"]}
    return dict(
        shared=kinds(item["shared_facts"])[::-1],              # W's first, then S
        memos=[kinds(item["seat_facts"][s]) for s in seats],
        informed=[s in solo for s in seats],
        n_seats=n,
        pooled=item["design"]["pooled_tally"],
        strong=strong,
    )


# ------------------------------------------------------------------ panel (a)
# Drawn in F1's notation, because it is the same object seen from outside: a
# seat is two concentric rings -- the outer one the answer it states in public,
# the inner one the belief it holds in private -- and one finding is one
# lettered box.  What (a) adds to F1 is where the evidence sits.  The shared
# briefing is at the CENTRE, with an arrow to every seat, because it is the one
# body of evidence nobody has to disclose and it points at the weak candidate;
# each seat's own memos sit INSIDE its own inner ring, because that is exactly
# what the rest of the team cannot see.
#
# Runs use N = 5 seats.  Five rings around one briefing is a knot at 1.8 in, so
# the diagram draws three -- one informed seat and two others -- and says so
# under the group; the caption carries the whole item.
#
# The column carries a SECOND tier under the group: one round of the protocol,
# same notation, following the informed agent from the belief it enters with to
# the belief it leaves with.  (b) measures two rates off exactly three
# comparisons -- entering belief against the visible majority, statement against
# that majority, leaving belief against that statement -- and a reader who has
# not got those three in the right order cannot read (b) at all.  The tier costs
# the width (b) and (c) gave up; drawn as a fourth panel it would have cost a
# band across the foot of the figure instead.
A_R_OUT = 0.27        # outer ring: the public statement
A_R_IN = 0.215        # inner ring: the private belief, and it holds the memos
A_BOX = 0.105         # one finding, in the briefing and in a memo alike
A_PITCH = 0.18        # briefing boxes, centre to centre
A_N_DRAWN = 3         # seats drawn, of the N the runs use
# The briefing is on TOP with the seats in one row under it, not at the centre
# of a ring of seats: the ring spent 2.4 in of column height on three seats, and
# the row spends 1.2, which is what leaves room for the second tier.
# The upper tier rides on the top of the column, so both heights are measured
# DOWN from A_H: the figure grew 0.5 in when (c) took the D-U plane, and a card
# pinned at a fixed 2.28 would have floated into the middle of the panel.
A_CARD_H = 0.35
A_CARD_Y = A_H - 0.20
A_ROW_Y = A_H - 0.78  # the seats, centre to centre A_SEAT_DX apart
A_SEAT_DX = 0.60      # 0.06 in of air between two outer rings
# 9/24 review ("幅を短くする。丸が大きすぎ。"): the drawing is set at this
# fraction of its inch-scale size, and the memo letters shrink with it
A_SCALE = 0.64
A_TEXT_SCALE = 0.80
A_NAME_W = 0.72       # "shared briefing" at FS_NOTE, in inches
A_TINT = 0.14         # how much of the belief hue the inner disc keeps as a wash
# "3 of the 5 agents" used to sit under the group and the lower tier now needs
# that 0.15 in for its own top labels; the count is in the caption, which is
# where the rest of the item already is.

# --- the lower tier: one agent, three rounds.
# What (b) needs from this panel is the INDEXING, because every count of
# section 3.1 is a comparison between two quantities with different subscripts:
# the belief the agent brings into round t carries t-1, and everything the
# round produces carries t.  A chain says that better than a two-column snapshot
# did, because the subscript is then read off the position: one box per round,
# the private belief z travelling along the chain from box to box, the public
# statement y leaving each box upward, and what every other agent stated in the
# round before arriving from below on that round's own index.  Round t -- the
# one the counts are made on -- is boxed in the palette's neutral area tone.
#
# The tier is DRAWN IN GREY, not in the S/W hues of the group above: it says
# when a quantity is measured, not which candidate it fell on, and colouring it
# would have claimed the second.
D_TOP = 1.30                       # the tier's ceiling; the agent row is above
D_CX = (0.29, 0.90, 1.51)          # rounds t-1, t, t+1, centre to centre
D_BOX = 0.22                       # one round: what the agent does in it
D_BOX_Y = 0.66                     # the chain's lane
D_SAY_Y = 1.06                     # the public statement, over its own round
D_SEEN_Y = 0.28                    # what the round shows the agent, under its box
D_PAD = 0.27                       # the grey box, centre of round t to its edge
D_HEAD_Y = D_TOP                   # the tier's ceiling, for the layout asserts

SIDE_COLOR = {"S": fs.BLUE, "W": fs.VERM}   # strong candidate, weak candidate


def wash(colour: str, strength: float = A_TINT):
    """``colour`` laid over white -- a fill a lettered box can still sit on."""
    return tuple(1 - strength * (1 - v) for v in mpl.colors.to_rgb(colour))


def note(ax, x, y, size, kind, zorder: float = 4):
    """One finding = one lettered box: S favours the strong candidate, W the weak.

    Both the hue and the letter carry it, so the box survives greyscale and both
    dichromacies on the letter alone.
    """
    ax.add_patch(mpl.patches.Rectangle((x - size / 2, y - size / 2), size, size,
                                       fc=SIDE_COLOR[kind], ec="none", zorder=zorder))
    ax.text(x, y, kind, ha="center", va="center",
            fontsize=fs.FS_SMALL * A_TEXT_SCALE, color=fs.SURFACE,
            zorder=zorder + 1)


def seat(ax, x, y, memos, said, held, phase: float):
    """One seat: two rings, with the memos it holds scattered inside the inner one.

    ``said`` colours the outer ring and ``held`` the inner one, so a seat whose
    rings disagree is one that is stating something it does not believe.
    """
    ax.add_patch(mpl.patches.Circle((x, y), A_R_OUT, fc=fs.SURFACE,
                                    ec=SIDE_COLOR[said], lw=1.5, zorder=3))
    ax.add_patch(mpl.patches.Circle((x, y), A_R_IN, fc=wash(SIDE_COLOR[held]),
                                    ec=SIDE_COLOR[held], lw=1.0, zorder=3.2))
    n = len(memos)
    q = 0.140 if n > 3 else 0.112           # radius the boxes are laid out on
    at = [(x + q * np.cos(phase + 2 * np.pi * i / n),
           y + q * np.sin(phase + 2 * np.pi * i / n)) for i in range(n)]
    # the ring is the only layout that fits five 0.115 in boxes inside a 0.24 in
    # circle, and it fits them by 0.01 in, so the fit is asserted, not eyeballed:
    # a memo count or a radius changed later must fail here rather than silently
    # print two boxes on top of each other.
    for (ax0, ay0), (ax1, ay1) in itertools.combinations(at, 2):
        assert max(abs(ax0 - ax1), abs(ay0 - ay1)) > A_BOX, \
            f"{n} memo boxes overlap at q = {q}"
    assert q + A_BOX * 0.708 < A_R_IN, f"{n} memo boxes leave the inner ring"
    for (bx, by), kind in zip(at, memos):
        note(ax, bx, by, A_BOX, kind)


def rings(ax, x, y, said, held, r_out, r_in):
    """The seat mark with no memos in it: the pair (statement, belief) alone."""
    ax.add_patch(mpl.patches.Circle((x, y), r_out, fc=fs.SURFACE,
                                    ec=SIDE_COLOR[said], lw=1.5, zorder=3))
    ax.add_patch(mpl.patches.Circle((x, y), r_in, fc=wash(SIDE_COLOR[held]),
                                    ec=SIDE_COLOR[held], lw=1.1, zorder=3.2))


def arrow(ax, p0, p1, **kw):
    ax.add_patch(mpl.patches.FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=5.5, lw=0.8,
        color=fs.INK_MID, shrinkA=0, shrinkB=0, zorder=1, **kw))


def round_box(ax, cx):
    """One round: the box the agent's own update happens in."""
    ax.add_patch(mpl.patches.FancyBboxPatch(
        (cx - D_BOX / 2, D_BOX_Y - D_BOX / 2), D_BOX, D_BOX,
        boxstyle="round,pad=0,rounding_size=0.03", fc=fs.SURFACE, ec=fs.INK,
        lw=1.1, zorder=3))


def draw_rounds(ax):
    """The lower tier: one agent carried through three rounds of the protocol.

    Reading it left to right: the agent enters a round holding a private belief
    z, the round shows it what every other agent stated in the round before, it
    leaves one public statement y upward, and it leaves the round holding a
    belief that the next round then enters with.  So the subscript is the
    position: everything over and after round t carries t, and the belief that
    entered it carries t-1, which is the pair every count of section 3.1
    compares.  Each round carries its OWN index, under the box as over it --
    round t is shown the statements of t-1, round t+1 those of t -- because a
    single label under the chain would have been right for one box of the three.
    The grey box is round t itself, the round those counts are made on; the last
    statement drawn is the team's answer.
    """
    x0, x1, x2 = D_CX
    half = D_BOX / 2

    # round t, boxed: the round the counts of section 3.1 are made on
    ax.add_patch(mpl.patches.FancyBboxPatch(
        (x1 - D_PAD, D_SEEN_Y - 0.08), 2 * D_PAD, D_TOP - D_SEEN_Y + 0.08,
        boxstyle="round,pad=0,rounding_size=0.04", fc=fs.FILL, ec="none",
        zorder=0))
    ax.text(x1, D_TOP - 0.055, "round $t$", ha="center", va="center",
            fontsize=fs.FS_SMALL, color=fs.INK2, zorder=1)

    # what the round shows the agent: what everyone else stated in the round
    # before, on its own index under each box
    for cx, lab in zip(D_CX, ("$\\{y_{j,t-2}\\}_{j \\neq i}$",
                              "$\\{y_{j,t-1}\\}_{j \\neq i}$",
                              "$\\{y_{j,t}\\}_{j \\neq i}$")):
        arrow(ax, (cx, D_SEEN_Y + 0.085), (cx, D_BOX_Y - half - 0.02))
        ax.text(cx, D_SEEN_Y, lab, ha="center", va="center",
                fontsize=fs.FS_SMALL, color=fs.INK)

    # the private belief travels the chain; the public statement leaves it
    arrow(ax, (x0 - 0.24, D_BOX_Y), (x0 - half - 0.02, D_BOX_Y))
    for cx in D_CX:
        round_box(ax, cx)
        arrow(ax, (cx, D_BOX_Y + half + 0.02), (cx, D_SAY_Y - 0.075))
    for a, b, lab in ((x0, x1, "$z_{i,t-1}$"), (x1, x2, "$z_{i,t}$")):
        arrow(ax, (a + half + 0.02, D_BOX_Y), (b - half - 0.02, D_BOX_Y))
        ax.text((a + b) / 2, D_BOX_Y + 0.085, lab, ha="center", va="bottom",
                fontsize=fs.FS_SMALL, color=fs.INK)

    for cx, lab in zip(D_CX, ("$y_{i,t-1}$", "$y_{i,t}$", "$y_{i,t+1}$")):
        ax.text(cx, D_SAY_Y, lab, ha="center", va="center",
                fontsize=fs.FS_SMALL, color=fs.INK)
    ax.text(x2, D_SAY_Y + 0.135, "answer", ha="center", va="center",
            fontsize=fs.FS_SMALL, color=fs.INK2)


def draw_a(ax, task, rounds: bool = True):
    cx = A_W / 2

    def belief(memo: str) -> str:
        """What one seat prefers on its own: its memos plus the shared briefing."""
        s = memo.count("S") + task["shared"].count("S")
        w = memo.count("W") + task["shared"].count("W")
        return "S" if s > w else "W"

    held = [belief(m) for m in task["memos"]]
    # everybody sees everybody's answer, so the visible majority is what the
    # seats hold; a seat that states it against its own belief is withholding.
    majority = max("SW", key=held.count)
    inf = task["informed"].index(True)      # the informed agent goes in the middle
    others = [i for i, f in enumerate(task["informed"]) if not f][:A_N_DRAWN - 1]
    drawn = [others[0], inf] + others[1:]

    card_w = (len(task["shared"]) - 1) * A_PITCH + A_BOX + 0.09
    # (a) is drawn at A_SCALE, the card's name is not: keep the card as wide
    # as the name it carries
    card_w = max(card_w, A_NAME_W / A_SCALE + 0.08)
    card_h = A_CARD_H
    pos = [(cx + dx, A_ROW_Y) for dx in (-A_SEAT_DX, 0.0, A_SEAT_DX)]
    # (a) rides on TOP like everything else, so a change up there must not push
    # the card off the top of the column, the row into the note under it, or the
    # outer seats into (b)'s rotated row names.
    assert A_CARD_Y + card_h / 2 < A_H - 0.02, "the briefing leaves (a)"
    assert A_ROW_Y - A_R_OUT > D_HEAD_Y + 0.08, "the row sits on the tier below"
    assert A_SEAT_DX + A_R_OUT < cx - 0.02, "the outer seats leave (a)"
    assert A_CARD_Y - card_h / 2 - (A_ROW_Y + A_R_OUT) > 0.08, "no room to arrow"

    # the briefing reaches every seat for free: that is what makes the task a
    # hidden profile, and what the arrows say that a shared box could not.
    for sx, sy in pos:
        dx, dy = sx - cx, sy - A_CARD_Y
        length = float(np.hypot(dx, dy))
        edge = min(abs(card_w / 2 / dx) if dx else np.inf,
                   abs(card_h / 2 / dy) if dy else np.inf)
        t0, t1 = edge + 0.02 / length, (length - A_R_OUT - 0.025) / length
        arrow(ax, (cx + dx * t0, A_CARD_Y + dy * t0),
              (cx + dx * t1, A_CARD_Y + dy * t1))

    ax.add_patch(mpl.patches.FancyBboxPatch(
        (cx - card_w / 2, A_CARD_Y - card_h / 2), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=0.05", fc=fs.FILL, ec="none",
        zorder=2))
    ax.text(cx, A_CARD_Y + 0.10, "shared briefing", ha="center", va="center",
            fontsize=fs.FS_NOTE, color=fs.INK2, zorder=3)
    for j, kind in enumerate(task["shared"]):
        note(ax, cx + (j - (len(task["shared"]) - 1) / 2) * A_PITCH,
             A_CARD_Y - 0.065, A_BOX, kind)

    # a phase per seat so three rings of boxes do not read as one stamped shape
    half = np.pi / 2
    for (sx, sy), i, phase in zip(pos, drawn, (half, half + 0.35, half - 0.35)):
        seat(ax, sx, sy, task["memos"][i], majority, held[i], phase)

    # the round chain is panel (b) since 9/15 and gets an axes of its own; the
    # two are still one drawing in one coordinate system, so nothing moved.
    if rounds:
        draw_rounds(ax)


# 9/24 review: (a) transposed -- the briefing on the left, the three agents
# stacked on the right -- so it is tall and narrow and gives its width to the
# panels beside it. Same marks as draw_a, in the same inch-scale units.
AV_SEAT_DY = 0.60                  # seats, centre to centre, vertically
AV_NAME_H = 0.24                   # "shared / briefing", two lines, in inches
AV_NAME_W = 0.42                   # the wider of the two lines, in inches
AV_GAP = 0.22                      # card edge to ring edge, room for arrows


def draw_a_vertical(ax, task) -> tuple:
    """Draw (a) transposed; returns its (x0, x1, y0, y1) extent in data units."""
    def belief(memo: str) -> str:
        s_ = memo.count("S") + task["shared"].count("S")
        w_ = memo.count("W") + task["shared"].count("W")
        return "S" if s_ > w_ else "W"

    held = [belief(m) for m in task["memos"]]
    majority = max("SW", key=held.count)
    inf = task["informed"].index(True)
    others = [i for i, f in enumerate(task["informed"]) if not f][:A_N_DRAWN - 1]
    drawn = [others[0], inf] + others[1:]

    n_sh = len(task["shared"])
    cols = 2 if n_sh > 2 else n_sh
    grid_rows = int(np.ceil(n_sh / cols))
    name_h = AV_NAME_H / A_SCALE
    card_w = max((cols - 1) * A_PITCH + A_BOX + 0.12, AV_NAME_W / A_SCALE + 0.08)
    card_h = name_h + (grid_rows - 1) * A_PITCH + A_BOX + 0.14
    cy = 0.0
    card_x0 = 0.0
    card_cx = card_x0 + card_w / 2
    seat_x = card_x0 + card_w + AV_GAP + A_R_OUT
    pos = [(seat_x, cy + dy) for dy in (AV_SEAT_DY, 0.0, -AV_SEAT_DY)]

    for sx, sy in pos:
        p0 = (card_x0 + card_w + 0.03, cy + 0.35 * (sy - cy) / AV_SEAT_DY * 0.3)
        dx, dy = sx - p0[0], sy - p0[1]
        length = float(np.hypot(dx, dy))
        t1 = (length - A_R_OUT - 0.03) / length
        arrow(ax, p0, (p0[0] + dx * t1, p0[1] + dy * t1))

    ax.add_patch(mpl.patches.FancyBboxPatch(
        (card_x0, cy - card_h / 2), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=0.05", fc=fs.FILL, ec="none",
        zorder=2))
    top = cy + card_h / 2
    ax.text(card_cx, top - 0.06, "shared\nbriefing", ha="center", va="top",
            fontsize=fs.FS_NOTE, color=fs.INK2, linespacing=1.0, zorder=3)
    g_top = top - 0.06 - name_h - 0.04 - A_BOX / 2
    for j, kind in enumerate(task["shared"]):
        r, c = divmod(j, cols)
        note(ax, card_cx + (c - (cols - 1) / 2) * A_PITCH, g_top - r * A_PITCH,
             A_BOX, kind)

    half = np.pi / 2
    for (sx, sy), i, phase in zip(pos, drawn, (half, half + 0.35, half - 0.35)):
        seat(ax, sx, sy, task["memos"][i], majority, held[i], phase)
    return (card_x0 - 0.03, seat_x + A_R_OUT + 0.03,
            cy - AV_SEAT_DY - A_R_OUT - 0.03, cy + AV_SEAT_DY + A_R_OUT + 0.03)


# 9/24 review, later: (a) is the figure's first column, full height, and
# narrow -- two agents (one uninformed, one informed), one above and one below
# the shared briefing, which sits in the middle and reaches both.
AC_SCALE = 1.12
AC_GAP = 0.10                      # card edge to arrow, arrow to ring


def draw_a_column(ax, task, height: float | None = None) -> tuple:
    """Draw (a) as a column; returns its (x0, x1, y0, y1) extent in data units.

    With ``height`` (data units) the two agents are pushed apart to fill it.
    """
    def belief(memo: str) -> str:
        s_ = memo.count("S") + task["shared"].count("S")
        w_ = memo.count("W") + task["shared"].count("W")
        return "S" if s_ > w_ else "W"

    held = [belief(m) for m in task["memos"]]
    majority = max("SW", key=held.count)
    inf = task["informed"].index(True)
    unin = task["informed"].index(False)
    drawn = [unin, inf]                         # top, bottom

    n_sh = len(task["shared"])
    cols = 2 if n_sh > 2 else n_sh
    grid_rows = int(np.ceil(n_sh / cols))
    name_h = AV_NAME_H / AC_SCALE
    card_w = max((cols - 1) * A_PITCH + A_BOX + 0.12,
                 AV_NAME_W / AC_SCALE + 0.08, 2 * A_R_OUT)
    card_h = name_h + (grid_rows - 1) * A_PITCH + A_BOX + 0.16
    cx = card_w / 2
    natural = 2 * (card_h / 2 + AC_GAP + 0.18 + AC_GAP + A_R_OUT)
    reach = max(natural, height or 0.0) / 2 - A_R_OUT   # centre to a seat
    pos = [(cx, reach), (cx, -reach)]

    ax.add_patch(mpl.patches.FancyBboxPatch(
        (cx - card_w / 2, -card_h / 2), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=0.05", fc=fs.FILL, ec="none",
        zorder=2))
    ax.text(cx, card_h / 2 - 0.06, "shared\nbriefing", ha="center", va="top",
            fontsize=fs.FS_NOTE, color=fs.INK2, linespacing=1.0, zorder=3)
    g_top = card_h / 2 - 0.06 - name_h - 0.05 - A_BOX / 2
    for j, kind in enumerate(task["shared"]):
        r, c = divmod(j, cols)
        note(ax, cx + (c - (cols - 1) / 2) * A_PITCH, g_top - r * A_PITCH,
             A_BOX, kind)
    for sgn in (1, -1):
        arrow(ax, (cx, sgn * (card_h / 2 + AC_GAP)),
              (cx, sgn * (reach - A_R_OUT - AC_GAP / 2)))
    half = np.pi / 2
    for (sx, sy), i, phase in zip(pos, drawn, (half, half + 0.35)):
        seat(ax, sx, sy, task["memos"][i], majority, held[i], phase)
    return (-0.03, card_w + 0.03, -reach - A_R_OUT - 0.03, reach + A_R_OUT + 0.03)


# --------------------------------------------------------------------- marks
# One mark, two channels, and neither of them is a hue: the SHAPE is the model
# and the FILL is the instruction arm: blue at A, filled ink at B, open at C.
#
# Greyscale, on the 9/15 sketch. Five models cannot each take a colour inside
# this palette -- ink, vermillion, blue and pink are the only four that clear
# the 2.8:1 printing floor, and green is out under spec section 1 -- so a fifth
# hue would have been a colour the reader cannot tell from the fourth. Shape
# separates as many models as the figure will ever carry, and the fill ramp is
# ordinal, which the instructions are.
# Colour marks A since 9/19. The honesty instruction is the arm every other
# arm is read against, and it was the one thing the figure made the reader find
# by position; B and C stay ink, filled and open. One table, so (c), (d), (e),
# (f) and appendix B2 cannot disagree about what a mark means.
# 9/24 review: B is grey, not ink -- a filled black mark was the strongest
# thing in every panel and it is the arm with no instruction. The filled marks
# carry a white edge so overlapping marks stay apart, and A is drawn a little
# larger, since it is the arm the others are read against. C stays open, so it
# keeps its ink edge (a white edge on a white face would be no mark at all).
B_GREY = "#949494"      # figstyle's third ink step, 3.0:1 on white
ARM_FACE = {"A": fs.BLUE, "B": B_GREY, "C": fs.SURFACE}
ARM_EDGE = {"A": fs.SURFACE, "B": fs.SURFACE, "C": fs.INK}
ARM_WHISKER = {"A": fs.BLUE, "B": B_GREY, "C": fs.INK_MID}
ARM_GLOSS = {"A": "honesty", "B": "none", "C": "strongest"}
# the white edge eats into the face, so the filled arms get it back in size
ARM_MS = {"A": 4.4, "B": 3.7, "C": 3.2}


def mark_style(arm: str, colour: str | None = None) -> dict:
    """Fill, edge and size for one instruction arm; the shape is its model."""
    return dict(mfc=ARM_FACE[arm], mec=colour or ARM_EDGE[arm], mew=0.5,
                ms=ARM_MS[arm])


def solid_thin_marks(fig) -> None:
    """gemini's star is a thin glyph: at the size of the other shapes it reads
    as a speck (9/24 review), and a white edge all but erases it. Draw it half
    again as large, edged in its own face. Run once every mark is drawn."""
    for ax in fig.axes:
        for ln in ax.get_lines():
            if ln.get_marker() != "*":
                continue
            ln.set_markersize(ln.get_markersize() * 1.6)
            if ln.get_markeredgecolor() == fs.SURFACE:
                ln.set_markeredgecolor(ln.get_markerfacecolor())


D_KEY_X0, D_KEY_Y0 = 0.21, 0.95   # (d)'s key: first marker, axes fractions
D_KEY_COL, D_KEY_ROW = 0.57, 0.12  # its pitch, inches
D_KEY_ROOM = 0.33                  # data units opened over (d)'s data for it


def d_model_key(fig, ax, models, cols: int = 3, x0: float = D_KEY_X0,
                y0: float = D_KEY_Y0) -> None:
    """(d)'s key of the model shapes, inside the panel in the upper right,
    which no setting reaches (9/24: only (d) draws a model by its shape, so
    only (d) carries the key, and in the panel rather than under it).

    ``x0``/``y0`` are the first entry's marker in axes fractions, under the
    "reasoning off" entry.
    """
    fig.canvas.draw()
    w_in, h_in = ax_size(ax)
    col_w, row_h = D_KEY_COL / w_in, D_KEY_ROW / h_in
    # first row: the instruction fills, as (b) keys them (9/24)
    t = ax.text(x0 - 0.02, y0, "Instruction", transform=ax.transAxes,
                fontsize=fs.FS_SMALL, color=fs.INK2, ha="left", va="center",
                zorder=5)
    word = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    for j, arm in enumerate(fs.ARMS):
        xa = x0 - 0.02 + (word + 0.10 + j * 0.22) / w_in
        ax.plot([xa], [y0], transform=ax.transAxes, marker="o", ls="none",
                clip_on=False, zorder=5, **mark_style(arm))
        ax.text(xa + 0.03, y0, arm, transform=ax.transAxes,
                fontsize=fs.FS_SMALL, color=fs.INK2, ha="left", va="center",
                zorder=5)
    for i, m in enumerate(models):
        r, c = divmod(i, cols)
        x, y = x0 + c * col_w, y0 - (r + 1) * row_h
        ax.plot([x], [y], transform=ax.transAxes, ls="none", clip_on=False,
                marker=fs.MODEL_MARKER[m], mfc=fs.INK, mec=fs.INK, mew=0.6,
                ms=3.2 * (1.6 if fs.MODEL_MARKER[m] == "*" else 1), zorder=5)
        ax.text(x + 0.03, y, fs.MODEL_SHORT[m], transform=ax.transAxes,
                fontsize=fs.FS_SMALL, color=fs.INK2, ha="left", va="center",
                zorder=5)


def finish(ax, ticks_pad: float = 0.5) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2, pad=ticks_pad)
    for axis in (ax.xaxis, ax.yaxis):
        axis.label.set_fontsize(fs.FS_AXIS)


def letter(fig, x_in: float, y_in: float, text: str) -> None:
    fig_text(fig, x_in, y_in, f"({text})", ha="left", va="top",
             fontsize=fs.FS_AXIS, fontweight="bold", color=fs.INK)




def arm_key(ax, panel_w_in: float, panel_h_in: float, x0: float, y0: float) -> tuple:
    """The three fills, named in one row inside a panel, in axes fractions.

    Was three rows of ``dy`` = 0.09 axes-fraction apart, which was 0.09 in on
    (f)'s data leaves no rectangle free at the panel's own scale (D, U in data
    units) -- the DU=const hyperbolae arc through most of the lower-left --
    but the corner at high D, low U (a team that pools evidence and still
    fails to use it) is empty of both marks and curves in every run this
    figure has drawn, so the key sits there, one row per arm, right-aligned.
    Returns the (x0, y0, x1, y1) box in axes fractions, so the caller can
    frame exactly what was drawn instead of a separately guessed rectangle.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    sample, gap = 0.045, 0.015                     # inches
    dy_in = 0.145            # row pitch; FS_SMALL's glyph is 0.113 in tall,
                              # so this clears it (the 9/16 bug: 0.09 in rows
                              # on a 1.00 in panel overlapped outright)
    dy = dy_in / panel_h_in
    widths = []
    for arm in fs.ARMS:
        t = ax.text(0, 0, arm, fontsize=fs.FS_SMALL)
        widths.append(t.get_window_extent(r).width / fig.dpi)
        t.remove()
    max_w = max(widths)
    row_w = sample + gap + max_w
    x_mark = x0 - row_w / panel_w_in
    for i, (arm, wd) in enumerate(zip(fs.ARMS, widths)):
        y = y0 - i * dy
        ax.plot([x_mark + sample / 2 / panel_w_in], [y], transform=ax.transAxes,
                marker="o", ls="none", zorder=6, clip_on=False,
                **mark_style(arm))
        ax.text(x_mark + (sample + gap) / panel_w_in, y, arm,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=fs.FS_SMALL, color=fs.INK2, zorder=6)
    pad_x_in, pad_y_in = 0.04, 0.055
    return (x_mark - pad_x_in / panel_w_in, y0 - 2 * dy - pad_y_in / panel_h_in,
            x0 + pad_x_in / panel_w_in, y0 + pad_y_in / panel_h_in)


def read_stars(prm) -> dict:
    """Each model's own critical withholding rate, at the no-instruction arm.

    (d) draws it at every arm; this is the one number the appendix quotes beside
    a rate, and fig_B2's caption reads it from here so the two cannot drift.
    """
    stars = {}
    for m in fs.MODELS:
        row = prm.loc[(prm.model == m) & (prm.condition == C_STAR_COND)].iloc[0]
        stars[m] = {k: float(row[c]) for k, c in
                    (("c", "c_star_post"), ("lo", "c_star_post_lo"),
                     ("hi", "c_star_post_hi"))}
    return stars


# ------------------------------------------------ the withholding rate and c*
# Rows of models, not columns of arms: the reader compares one model's arms
# ALONG its row and the models DOWN the column. Since the review the three arms of
# one model sit on that model's row itself, not dodged around it, so a model is
# one line of the panel; the fill tells the arms apart (A blue, B filled ink,
# C open) and the shape is still the model's, the key (d) is read with.
# arm_panel also draws the per-model threshold c* beside the rate.
def model_rows(prm) -> list:
    """The row order (c) and (d) share: the withholding rate at B, ascending.

    B is the arm with no instruction on it, so the order is what the models do
    left alone, and both panels take it from here rather than each sorting
    itself -- two stacked panels whose rows disagreed would be two panels.
    """
    at_b = prm[prm.condition == "B"].set_index("model")[C_COL]
    return sorted(fs.MODELS, key=lambda m: float(at_b[m]))


def cd_arm_key(ax, row: float) -> None:
    """"Instruction" then A / B / C, in the strip the ylim opens over the top
    row (9/24: the letters alone did not say what they were)."""
    fig = ax.figure
    fig.canvas.draw()
    w_in = ax.get_position().width * fig.get_size_inches()[0]
    t = ax.text(0.0, row, "Instruction", transform=ax.get_yaxis_transform(),
                fontsize=fs.FS_SMALL, color=fs.INK2, va="center", ha="left")
    word = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    for i, arm in enumerate(fs.ARMS):
        x = (word + 0.10 + i * 0.20) / w_in
        ax.plot([x], [row], transform=ax.get_yaxis_transform(), marker="o",
                ls="none", zorder=4, clip_on=False, **mark_style(arm))
        ax.annotate(arm, (x, row), xycoords=ax.get_yaxis_transform(),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=fs.FS_SMALL, color=fs.INK2, va="center", ha="left",
                    annotation_clip=False)


# The strip over the top row and the margin under the bottom row, in rows;
# (e) stacks its two series names in a taller strip of its own.
ROW_STRIP, ROW_FOOT = 1.15, 0.60
TABLED_STRIP = 2.45


def data_ticks(lo: float, hi: float, pad_frac: float = 0.06) -> tuple:
    """Limits fitted to [lo, hi] and at most three round ticks inside them.

    review: an axis ends where its data end, not at 1. The ticks are
    the coarsest step that still puts at least three of them in range (ends
    and a middle), falling back to two.
    """
    pad = pad_frac * (hi - lo) if hi > lo else 0.05
    x0, x1 = lo - pad, hi + pad
    best = None
    for step in (0.5, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01):
        ticks = [round(t, 10) for t in np.arange(np.ceil(x0 / step) * step,
                                                  x1 + 1e-9, step)]
        if len(ticks) >= 3 and len(ticks) % 2 == 1:
            best = [ticks[0], ticks[len(ticks) // 2], ticks[-1]]
            break
        if len(ticks) >= 2 and best is None:
            best = [ticks[0], ticks[-1]]
    return (x0, x1), [0.0 if abs(t) < 1e-9 else t for t in best]


def arm_panel(ax, prm, col: str, sym: str, order: list, key: bool = False,
              xlabel: str | None = None, marker: str | None = None) -> None:
    """One rate per model, with its three instruction arms on that model's row.

    Each mark carries the 95 % credible interval of the per-setting posterior of
    section 3.1, drawn along the rate. The arms share the row, so the line the
    older panel drew through them is gone: the row is the line.
    """
    for i, m in enumerate(order):
        g = prm[prm.model == m].set_index("condition").reindex(fs.ARMS)
        xs = g[col].to_numpy(float)
        # whiskers first, all three, so no mark is painted over by another
        # arm's interval
        for j, arm in enumerate(fs.ARMS[::-1]):
            k = fs.ARMS.index(arm)
            fs.whisker(ax, [i], [g[f"{col}_lo"].iloc[k]], [g[f"{col}_hi"].iloc[k]],
                       ARM_WHISKER[arm], orient="h", zorder=2.4 + 0.01 * j)
        for arm in fs.ARMS[::-1]:
            k = fs.ARMS.index(arm)
            # the row already names the model, so F2 draws a plain circle
            # (9/24)
            ax.plot([xs[k]], [i], marker=marker or fs.MODEL_MARKER[m],
                    ls="none", zorder=3.2, **mark_style(arm))
        diag("%s %-10s " % (col, fs.MODEL_SHORT[m])
             + "  ".join("%s %.2f [%.2f, %.2f]"
                         % (a, xs[k], g[f"{col}_lo"].iloc[k], g[f"{col}_hi"].iloc[k])
                         for k, a in enumerate(fs.ARMS)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([fs.MODEL_SHORT[m] for m in order])
    ax.set_ylim(len(order) - 1 + ROW_FOOT, -ROW_FOOT - (ROW_STRIP if key else 0.0))
    ax.spines["left"].set_bounds(0, len(order) - 1)
    sub = prm[prm.model.isin(order)]
    (x0, x1), ticks = data_ticks(float(sub[f"{col}_lo"].min()),
                                 float(sub[f"{col}_hi"].max()))
    ax.set_xlim(x0, x1)
    fs.ticks01(ax, "x", tuple(ticks))
    ax.set_xlabel(xlabel or sym, labelpad=1)
    if key:
        cd_arm_key(ax, -ROW_FOOT - ROW_STRIP / 2 + 0.1)
    finish(ax)


# ----------------------------------------------------------------- panel (e)
def gain_points(prm, accall) -> pd.DataFrame:
    """One row per (model, arm): the margin, and what deliberating was worth.

    The gain is the collective accuracy after discussion minus the round-0
    majority vote of the same agents on the same items, so it is what the
    interaction added over not interacting at all. The interval is the accuracy
    interval carried onto the gain: the vote is the same agents on the same
    items, so it moves the point but not the width by much, and the caption
    says so.
    """
    pr = prm.set_index(["model", "condition"])
    ai = accall.set_index(["model", "condition"])
    rows = []
    for m in fs.MODELS:
        for arm in fs.ARMS:
            p, a = pr.loc[(m, arm)], ai.loc[(m, arm)]
            vote = float(a["vote_accuracy"])
            rows.append(dict(
                model=m, arm=arm,
                margin=float(p[C_COL] - p["c_star_post"]),
                gain=float(a["accuracy"]) - vote,
                lo=float(a["acc_lo"]) - vote, hi=float(a["acc_hi"]) - vote))
    return pd.DataFrame(rows)


FIT_DEG = 2              # (d): degree of the drawn fit
BAND_BOOT = 2000         # (d): bootstrap draws for the fit's band
OFF_ALPHA = 0.45        # (d): a reasoning-off setting, faded behind the main ones


def gain_points_off() -> pd.DataFrame:
    """(d)'s extra points: the reasoning-off arm of fig_B3's six pairs (9/24).

    Same estimator and same gain as ``gain_points``, read from the off run's
    own ``cbrm_params.csv`` and ``hp_accuracy.csv``.
    """
    rows = []
    for m, (_, off, _) in b3.PAIRS.items():
        p = fs.arms_only(pd.read_csv(fs.DATA / off / "cbrm_params.csv"))
        a = fs.arms_only(pd.read_csv(fs.DATA / off / "hp_accuracy.csv"))
        p = p.set_index("condition")
        a = a[a.k_informed.astype(str) == "all"].set_index("condition")
        for arm in fs.ARMS:
            vote = float(a.loc[arm, "vote_accuracy"])
            rows.append(dict(
                model=m, arm=arm,
                margin=float(p.loc[arm, C_COL] - p.loc[arm, "c_star_post"]),
                gain=float(a.loc[arm, "accuracy"]) - vote,
                lo=float(a.loc[arm, "acc_lo"]) - vote,
                hi=float(a.loc[arm, "acc_hi"]) - vote))
    return pd.DataFrame(rows)


# (e)-(g): reasoning on is every model's default setting, so it is the plain
# mark -- an open circle -- and off is the one marked, striped (9/24)
# filled in A's blue, white-edged, as A is everywhere else in F2 (9/24)
ON_OPEN = dict(mfc=fs.BLUE, mec=fs.SURFACE, mew=0.5, ms=ARM_MS["A"])
OFF_HATCH = "//////////////"     # (d): a reasoning-off setting is striped


def off_mark(ax, x, y, marker, arm, transform=None, ms=None) -> None:
    """One reasoning-off setting: the model's shape, white inside, striped and
    edged in its arm's colour (C, the open arm, in ink)."""
    c = {"A": fs.BLUE, "B": B_GREY, "C": fs.INK}[arm]
    ms = ms or ARM_MS[arm] * (1.6 if marker == "*" else 1.0) + 0.6
    kw = dict(transform=transform) if transform is not None else {}
    with mpl.rc_context({"hatch.color": c, "hatch.linewidth": 0.45}):
        ax.scatter([x], [y], s=ms ** 2, marker=marker, facecolor=fs.SURFACE,
                   edgecolor=c, linewidth=0.6, hatch=OFF_HATCH, zorder=3.1,
                   clip_on=transform is None, **kw)


def gain_panel(ax, pts: pd.DataFrame, off: pd.DataFrame | None = None) -> None:
    """The gain against the distance from the threshold, one point per setting.

    The claim of 4.2 in one panel: what orders the gain is not the withholding
    rate but the DISTANCE from that setting's own critical value. The trend line
    is the only saturated thing in the figure, because it is the only claim the
    figure makes that a reader could otherwise take or leave.
    """
    # 9/24: no whisker per setting -- the marks are the point estimates and
    # the uncertainty is the fit's, one 95 % band around the OLS line
    # (bootstrap over settings, as seaborn's regplot draws it)
    for _, r in pts.iterrows():
        ax.plot([r.margin], [r.gain], marker=fs.MODEL_MARKER[r.model],
                ls="none", zorder=3.2, **mark_style(r.arm))
    if off is not None:
        # the reasoning-off arms, striped in their arm's colour (9/24); they
        # enter the fit and its band with the main runs (9/24, later)
        for _, r in off.iterrows():
            off_mark(ax, r.margin, r.gain, fs.MODEL_MARKER[r.model], r.arm)
    x, y = pts.margin.to_numpy(), pts.gain.to_numpy()
    y_all = y if off is None else np.r_[y, off.gain.to_numpy()]
    x_all = x if off is None else np.r_[x, off.margin.to_numpy()]
    x, y = x_all, y_all              # every drawn setting, reasoning on and off
    slope, intercept = np.polyfit(x, y, 1)
    # the drawn fit is quadratic (9/24): the gain falls with the distance and
    # then flattens near zero, and a line through that floor misses both ends
    # (the 27 reasoning-on settings: AIC -146 linear vs -166 quadratic,
    # leave-one-out MSE 0.0056 vs 0.0024; the x^2 term holds without the
    # leftmost setting)
    coef = np.polyfit(x, y, FIT_DEG)
    rho, p_rho = spearmanr(x, y)
    xx = np.linspace(x.min(), x.max(), 100)
    rng = np.random.default_rng(0)
    boot = np.empty((BAND_BOOT, xx.size))
    for b in range(BAND_BOOT):
        i = rng.integers(0, x.size, x.size)
        boot[b] = np.polyval(np.polyfit(x[i], y[i], FIT_DEG), xx)
    band_lo, band_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ax.fill_between(xx, band_lo, band_hi, color=fs.VERM, alpha=0.18, lw=0,
                    zorder=2.0)
    ax.plot(xx, np.polyval(coef, xx), "-", color=fs.VERM, lw=1.2,
            solid_capstyle="round", zorder=3.0)
    diag("(e) %d settings, slope %+.3f, Spearman %+.2f (p = %.1g)"
         % (len(x), slope, rho, p_rho))
    pad = 0.06 * (x_all.max() - x_all.min())
    ax.set_xlim(x_all.min() - pad, x_all.max() + pad)
    lo = min(y_all.min(), band_lo.min(), 0.0)
    # the band is not allowed to set the range: at the far left one setting
    # carries the curve and its band opens wide, so it runs off the top
    hi = max(y_all.max(), 0.0) + D_KEY_ROOM   # (d)'s key on top
    ypad = 0.08 * (hi - lo)
    ax.set_ylim(lo - ypad, hi + ypad)
    # zero in x is the threshold itself and zero in y is the vote: left of the
    # first a team can still recover, under the second discussion cost it
    fs.reference(ax, 0.0, "v")
    fs.reference(ax, 0.0, "h")
    # three ticks an axis (spec section 5); zero is a reference line here, not a
    # tick, so the ticks are the round values either side of the range
    ticks = [t for t in (-0.2, 0.2, 0.6, 1.0)
             if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]][:3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([fs.fmt_tick(t) for t in ticks])
    yt = [t for t in (-0.2, 0, 0.2, 0.4, 0.6) if lo - ypad <= t <= hi + ypad]
    ax.set_yticks(yt)
    ax.set_yticklabels([fs.fmt_tick(t) for t in yt])
    ax.set_xlabel("$\\hat{c} - c^{*}$", labelpad=1)
    ax.set_ylabel("gain from\ndiscussion", labelpad=1, linespacing=1.0)
    if off is not None:
        # one entry: the striped mark is the only thing in (d) no other key
        # names; it closes d_model_key's grid, in its last row
        kx = D_KEY_X0 + D_KEY_COL / ax_size(ax)[0]      # second column
        off_mark(ax, kx, D_KEY_Y0 - 4 * D_KEY_ROW / ax_size(ax)[1], "o", "C",
                 transform=ax.transAxes)
        ax.text(kx + 0.03, D_KEY_Y0 - 4 * D_KEY_ROW / ax_size(ax)[1],
                "reasoning off", transform=ax.transAxes,
                fontsize=fs.FS_SMALL, color=fs.INK2, ha="left", va="center")
    finish(ax)
    return dict(slope=float(slope), rho=float(rho), p=float(p_rho),
                coef=[float(c) for c in coef])


# ----------------------------------------------------------------- panel (f)
def du_points() -> dict:
    """D and U with their intervals, per (model, arm). fig_F4 owns the estimates."""
    raw = {v: k for k, v in fs.ARM_NAME.items()}     # fig_F4 reads raw keys
    return {(m, arm): f4.du_estimate(m, raw[arm])
            for m in fs.MODELS for arm in fs.ARMS}


def du_panel(ax, du: dict) -> dict:
    """Where the evidence goes: what reached the table, and what was used of it.

    A team can fail by tabling nothing (low D) or by tabling and then not using
    it (low U), and the two want different repairs. The grey hyperbolae are
    levels of the product D x U, which is the accuracy the plane implies; the
    dashed one is what the same agents score by voting without discussing.
    """
    xs = [v for (D, (dlo, dhi), U, (ulo, uhi)) in du.values() for v in (dlo, dhi)]
    ys = [v for (D, (dlo, dhi), U, (ulo, uhi)) in du.values() for v in (ulo, uhi)]
    pad_x, pad_y = 0.055 * (max(xs) - min(xs)), 0.055 * (max(ys) - min(ys))
    x0, x1 = max(0.0, min(xs) - pad_x), min(1.02, max(xs) + pad_x)
    y0, y1 = max(0.0, min(ys) - pad_y), min(1.06, max(ys) + pad_y)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    d = np.linspace(max(x0, 1e-3), x1, 400)
    for a in f4.ISO_LEVELS:
        u = a / d
        ok = (u >= y0) & (u <= y1)
        ax.plot(d[ok], u[ok], color=fs.INK3, lw=0.5, zorder=0.5)
    a_vote = f4.vote_accuracy()[0]
    uv = a_vote / d
    ok = (uv >= y0) & (uv <= y1)
    ax.plot(d[ok], uv[ok], color=fs.INK3, lw=0.5, ls=(0, (2.4, 1.6)), zorder=0.6)
    for m in fs.MODELS:
        line_d = [du[(m, arm)][0] for arm in fs.ARMS]
        line_u = [du[(m, arm)][2] for arm in fs.ARMS]
        ax.plot(line_d, line_u, "-", color=fs.INK3, lw=0.6, zorder=1.8)
    for (m, arm), (D, (dlo, dhi), U, (ulo, uhi)) in du.items():
        ax.plot([D], [U], marker=fs.MODEL_MARKER[m], ls="none",
                zorder=3.2, **mark_style(arm))
        diag("(f) %-10s %s  D %.3f [%.3f, %.3f]  U %.3f [%.3f, %.3f]  DU %.3f"
             % (fs.MODEL_SHORT[m], arm, D, dlo, dhi, U, ulo, uhi, D * U))
    xt = [t for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7) if x0 <= t <= x1][::2] or [round(x0, 1)]
    yt = [t for t in (0.4, 0.7, 1.0) if y0 <= t <= y1]
    fs.ticks01(ax, "x", tuple(xt))
    fs.ticks01(ax, "y", tuple(yt))
    ax.spines["bottom"].set_bounds(xt[0], xt[-1])
    ax.spines["left"].set_bounds(yt[0], yt[-1])
    ax.set_xlabel("evidence pooling rate", labelpad=1)
    ax.set_ylabel("evidence\nutilization rate", labelpad=1, linespacing=1.0)
    finish(ax)
    w_in, h_in = ax_size(ax)
    bx0, by0, bx1, by1 = arm_key(ax, w_in, h_in, 0.97, 0.32)
    leg_box = mpl.patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                    transform=ax.transAxes,
                                    fc=fs.SURFACE, ec=fs.INK, lw=0.8, zorder=5)
    ax.add_patch(leg_box)
    return dict(iso=f4.ISO_LEVELS, vote=a_vote, x0=x0, x1=x1, y0=y0, y1=y1)


# ----------------------------------------------------------------- panel (g)
def tabled_panel(ax) -> tuple:
    """What a model puts on the table: its own answer, or the correct one.

    Filled is the correct candidate and open is the model's own current answer,
    the same fill language the arms use in the panels above; the models are rows
    because four names do not fit along a 1.37 in axis. Pooled over the three
    arms, so this panel has no arm to fill by.
    """
    order, tally, share = f4.tabled_summary()
    diag("(g) tally accuracy (count_correct), ascending: "
         + ", ".join(f"{fs.MODEL_SHORT[k]} {tally[k]:.3f}" for k in order))
    y = np.arange(len(order), dtype=float)
    for i, m in enumerate(order):
        for key, filled in (("truth", True), ("own", False)):
            v, lo, hi = share[m][key]
            fs.whisker(ax, [i], [lo], [hi], fs.INK_MID, orient="h", zorder=2.4)
            ax.plot([v], [i], marker="o", ms=MS, ls="none", zorder=3.2,
                    mfc=fs.INK if filled else fs.SURFACE, mec=fs.INK, mew=0.7)
        diag("(g) %-10s own %.3f [%.3f, %.3f]  correct %.3f [%.3f, %.3f]"
             % ((fs.MODEL_SHORT[m],) + share[m]["own"] + share[m]["truth"]))
    ax.set_yticks(y)
    ax.set_yticklabels([fs.MODEL_SHORT[m] for m in order])
    ax.set_ylim(y[-1] + ROW_FOOT, y[0] - TABLED_STRIP)  # the strip holds
    ax.spines["left"].set_bounds(y[0], y[-1])  # the two series names
    ax.set_xlim(0, 1.06)
    fs.ticks01(ax, "x")
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # The two series names sit in the 1.10-row strip the ylim opens over the top
    # row, so the height is given in ROW units and read through a blended
    # transform -- x in axes fraction, y in data. As a plain axes fraction it
    # drifted with the number of rows and landed on the first row's whisker the
    # moment a sixth model was drawn (9/19: gemini).
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    sample, gap = 0.05, 0.018                      # inches
    entries = [("truth", True, "correct answer"), ("own", False, "own answer")]
    widths = []
    for _, _, label in entries:
        t = ax.text(0, 0, label, fontsize=fs.FS_SMALL)
        widths.append(t.get_window_extent(r).width / fig.dpi)
        t.remove()
    # Stacked, not side by side: the two names together need 1.44 in of a
    # 1.34 in panel, and shortening them to fit ("correct", "own") would leave
    # the reader to guess what the answer is correct or own ABOUT. Two rows of
    # the strip cost nothing -- the strip is 1.55 rows and holds them.
    run = max(widths) + sample + gap
    w_in = ax_size(ax)[0]
    assert run <= w_in, f"(e) legend needs {run:.2f} in of {w_in:.2f}"
    x = (w_in - run) / 2
    for (key, filled, label), legend_y in zip(entries, (-TABLED_STRIP + 0.55,
                                                        -TABLED_STRIP + 1.45)):
        ax.plot([(x + sample / 2) / w_in], [legend_y], transform=trans,
                marker="o", ms=MS, ls="none", zorder=4,
                mfc=fs.INK if filled else fs.SURFACE, mec=fs.INK, mew=0.7)
        ax.text((x + sample + gap) / w_in, legend_y, label,
                transform=trans, fontsize=fs.FS_SMALL, color=fs.INK2,
                va="center", ha="left")
    ax.set_xlabel("share of tabled evidence", labelpad=1)
    finish(ax)
    return order, tally, share

def _join_and(names):
    if not names:
        return ""
    return (names[0] if len(names) == 1
            else f"{', '.join(names[:-1])} and {names[-1]}")


# ------------------------------------------------------------------- caption
def caption(task, prm, accall, pts, off, fit, data3, order3):
    """Every sentence of F2, written from the tables the panels draw.

    Nothing here may be typed in by hand: the estimator behind ``C_COL`` changed
    once already and every withholding number moved with it. Rewritten for the
    9/24 layout: (a) two agents, (b) the rate, (c) its threshold, (d) the gain
    against the distance, (e)-(g) reasoning on and off at instruction A.
    """
    n_seats = int(prm["n_agents"].iloc[0])
    T = int(prm["T"].iloc[0])
    n_tasks = int(prm["n_tasks"].iloc[0])
    memo_inf = task["memos"][task["informed"].index(True)]
    memo_un = task["memos"][task["informed"].index(False)]
    ks = sorted({0, 1, 2, 3})
    n_inf = sum(task["informed"])
    pooled_s = task["shared"].count("S") + sum(m.count("S") for m in task["memos"])
    pooled_w = task["shared"].count("W") + sum(m.count("W") for m in task["memos"])

    def counts(s):
        return f"{s.count('S')} for S and {s.count('W')} for W"

    def with_briefing(memo):
        """What one seat can see on its own: its memos plus the shared briefing."""
        s = memo.count("S") + task["shared"].count("S")
        w = memo.count("W") + task["shared"].count("W")
        return s, w, ("S" if s > w else "W")

    inf_s, inf_w, inf_pick = with_briefing(memo_inf)
    un_s, un_w, un_pick = with_briefing(memo_un)
    held = [with_briefing(m)[2] for m in task["memos"]]
    majority = max("SW", key=held.count)

    idx = prm.set_index(["model", "condition"])

    def spanned(col):
        return "; ".join(
            f"{fs.MODEL_SHORT[m]} "
            + " to ".join(f"{v:.2f}" for v in
                          (idx.loc[(m, fs.ARMS[0]), col], idx.loc[(m, fs.ARMS[-1]), col]))
            for m in fs.MODELS)

    join_and = _join_and

    # (c): which models sit above their own threshold, and which cross it
    def above_at(m, a):
        return idx.loc[(m, a), C_COL] > idx.loc[(m, a), "c_star_post"]
    above = [fs.MODEL_SHORT[m] for m in fs.MODELS
             if all(above_at(m, a) for a in fs.ARMS)]
    crosses = [fs.MODEL_SHORT[m] for m in fs.MODELS
               if any(above_at(m, a) for a in fs.ARMS)
               and not all(above_at(m, a) for a in fs.ARMS)]
    side_txt = "; ".join(part for part in (
        f"{join_and(above)} withhold more than"
        f" {'its' if len(above) == 1 else 'their'} own threshold allows at"
        " under every instruction" if above else "",
        f"{join_and(crosses)} cross{'es' if len(crosses) == 1 else ''} it"
        " inside the series" if crosses else "") if part)

    # (d)
    neg = pts[pts.margin < 0]
    gain_txt = "; ".join(
        f"{fs.MODEL_SHORT[r.model]} at {r.arm} {r.margin:+.2f}, gain {r.gain:+.2f}"
        for _, r in pts.sort_values("margin").iterrows())
    best = pts.sort_values("gain").iloc[-1]
    off_txt = "; ".join(
        f"{fs.MODEL_SHORT[r.model]} at {r.arm} {r.margin:+.2f}, gain {r.gain:+.2f}"
        for _, r in off.sort_values("margin").iterrows())

    # (e)-(g): instruction A, reasoning on -> off, per model
    def switch(col):
        return "; ".join(
            f"{fs.MODEL_SHORT[m]} {data3[m]['on'][col]['value']:.2f} to"
            f" {data3[m]['off'][col]['value']:.2f}"
            for m in order3 if m in b3.PAIRS)
    # the models whose second arm is not reasoning off outright (e.g. `low`)
    partial_off = "; ".join(f"{fs.MODEL_SHORT[m]} '{v[2]}'"
                            for m, v in b3.PAIRS.items() if v[2] != "off")

    models_txt = join_and([fs.MODEL_ID[m].split("/")[-1] for m in fs.MODELS])
    arms_txt = ", ".join(f"{a} ({ARM_GLOSS[a]})" for a in fs.ARMS)

    fs.write_caption("F2", [
        "Every quantity the model needs is measurable from one task's logs, the"
        " instruction series moves withholding, and each model has to be read"
        " against its own threshold. (a) The hidden-profile task, in the"
        f" notation of Figure 1: {n_seats} agents hire one of two candidates, an"
        " agent is two rings -- the outer one the answer it states in public,"
        " the inner one the belief it holds in private -- and one box is one"
        " finding, vermillion W for the weak candidate and blue S for the strong"
        " one. The shared briefing, in the middle, is the evidence every agent"
        " reads without anyone having to disclose it, and it holds"
        f" {task['shared'].count('W')} findings for W against"
        f" {task['shared'].count('S')} for S; the arrows, up and down, say that"
        " it reaches both agents drawn. An agent's own memos sit inside its own"
        " ring, which is where the rest of the team cannot see them."
        f" Drawing all {n_seats} around one briefing is unreadable at this size,"
        f" so the diagram draws 2 of the {n_seats} agents: above the briefing, one"
        " without the decisive evidence, and below it one of the k informed"
        f" agents. The informed agent's memos ({counts(memo_inf)}) turn the"
        f" briefing around, so on everything it can see -- {inf_s} for S against"
        f" {inf_w} for W -- it privately holds {inf_pick}, and its inner ring is"
        f" blue. The upper agent's memos ({counts(memo_un)}) do not: with the"
        f" briefing added it sees {un_s} for S against {un_w} for W and holds"
        f" {un_pick}. {held.count(majority)} of the {n_seats} agents hold"
        f" {majority}, so {majority} is the visible majority and both outer"
        " rings state it: the lower agent's two rings disagree, and stating the"
        " majority against one's own belief is the withholding (b) measures."
        f" Pooled across the {n_seats} agents the drawn item holds {pooled_s}"
        f" findings for S against {pooled_w} for W, so only disclosure reaches"
        f" the right answer. Runs use N = {n_seats} agents and k informed agents"
        f" with k in {{{', '.join(str(int(k)) for k in ks)}}}, so the initial"
        f" accuracy p = k/N is fixed by design, over T = {T} rounds and"
        f" {n_tasks} tasks per model; the diagram draws one k = {n_inf} item.",
        "(b) The withholding rate the logs measure, one row per model and the"
        " three instructions on that model's row itself, so a row is read"
        f" along and the models are read down. The instructions are {arms_txt}: A"
        " instructs each agent to answer honestly, B adds no instruction at"
        " all, and C describes dissent as costly and grades cohesion. A is"
        " the blue mark, edged in white, B the grey one and C the open one"
        " (the Instruction key over the panel), and the same three fills draw"
        " (c) and (d); every mark here is a circle, because the row already"
        " names the model. The rate is c-hat, measured as"
        f" {C_DEF}; the mark is the posterior mean of section 3.1 and the"
        " whisker its 95 % credible interval, and the axis is cropped to the"
        f" whiskers. Over the three instructions it runs {spanned(C_COL)}.",
        "(c) The threshold each rate has to be read against, drawn beside it"
        " on the same rows, with the same marks and intervals: c* ="
        " gamma / (gamma + a), the withholding rate above which a team cannot"
        " recover the right answer, from that setting's own internalization and"
        " net correction (section 2), so it moves with the instruction as the"
        f" rate does. Over the three instructions it runs {spanned('c_star_post')}."
        f" Read against (b), {side_txt}.",
        "(d) The distance between the rate and its threshold, against what"
        " deliberating was worth."
        " One point per (model, instruction), the point estimate alone; the horizontal"
        " position is the margin c-hat - c*, the vertical one the collective"
        " accuracy after discussion minus the round-0 majority vote of the same"
        " agents on the same items, so zero in y is what those agents score"
        " without interacting at all and zero in x is the threshold. The shape"
        " is the model (the key along the top of the panel) and the fill the"
        " instruction, as in (b). The striped marks are the"
        f" {off.model.nunique()} models that can switch reasoning off, run with"
        " it off -- the model's shape, striped in the instruction's colour:"
        f" {off_txt}. The curve is a quadratic fit over all"
        f" {len(pts) + len(off)} settings drawn, the {len(pts)} with reasoning"
        f" on and the {len(off)} with it off, and the band its 95 % interval"
        f" from {BAND_BOOT} bootstrap resamples of those settings; the rank"
        f" correlation over them is {fit['rho']:+.2f}. Reading the"
        f" reasoning-on points: {gain_txt}. The best-paying of them is"
        f" {fs.MODEL_SHORT[best.model]} at {best.arm}, worth {best.gain:+.2f}"
        f" over the vote; {len(neg)} of the {len(pts)} sit on the recovery"
        " side of the threshold.",
        "(e)-(g) Reasoning switched on and off in the same model, for the"
        f" {sum(m in b3.PAIRS for m in order3)} models with a reasoning-off run,"
        " one row per model, at instruction A alone: on, the model's default,"
        " is the filled blue mark edged in white, as A is in (b), and off the"
        " striped blue one. (e) Internalization rate a-hat; (f) net correction"
        " rate gamma-hat; (g) gain from discussion. Bars are 95 % Wilson"
        " intervals in (e) and (g) -- in (g) the accuracy's, carried onto the"
        " gain -- and in (f) the estimate's own 95 % interval. On to off,"
        f" a-hat runs {switch('a')}; gamma-hat runs {switch('gamma')}; the"
        f" gain runs {switch('gain')}. Second arms that are not reasoning off"
        f" outright, as recorded in each run's events.meta.json: {partial_off or 'none'}."
        " Every other off setting is off outright. Appendix Figure B3 draws the same switch pooled"
        " over the three instructions, with the withholding rate."
        f" The models are {models_txt}, and every panel is measured on runs that"
        f" elicit the private answer in a call of its own ({main_probe_name()}"
        " probe, appendix B.2).",
    ])


# ------------------------------------------------------------------- checks
def check_layout(fig):
    """Warn on text outside the canvas, or on two texts whose boxes overlap.

    The canvas is read off the figure, not off this module's constants, so
    fig_B2 -- which is shorter -- can call it on its own render.
    """
    fig.canvas.draw()
    r, D = fig.canvas.get_renderer(), fig.dpi
    fig_w, fig_h = fig.get_size_inches()
    items = list(fig.texts)
    for ax in fig.axes:
        items += [t for t in ax.texts if t.get_text().strip()]
        if not ax.axison:
            continue
        items += [t for t in ax.get_xticklabels() + ax.get_yticklabels()
                  if t.get_text().strip()]
        items += [lab for lab in (ax.xaxis.label, ax.yaxis.label)
                  if lab.get_text().strip()]
    boxes = []
    for t in items:
        bb = t.get_window_extent(r)
        boxes.append((t.get_text(), bb.x0 / D, bb.y0 / D, bb.x1 / D, bb.y1 / D))
        if bb.x0 / D < -0.005 or bb.x1 / D > fig_w + 0.005 \
                or bb.y0 / D < -0.005 or bb.y1 / D > fig_h + 0.005:
            print("WARNING outside the canvas:", repr(t.get_text()))
    for a, b in itertools.combinations(boxes, 2):
        if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
            print("WARNING text boxes overlap:", repr(a[0]), "||", repr(b[0]))


# -------------------------------------------------------------------- figure
def make():
    fs.apply()
    prm = load_params()
    accall = load_accuracy_all()
    task = load_task(SCHEMATIC_K)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # 9/24 sketch: (a) over (c) on the left, (d) -- the main claim -- the
    # largest panel in the middle, the reasoning switch on the right with
    # (e) and (f) side by side on one set of rows over a wider (g). The
    # round chain that was (b) left the figure.
    y_top_row = TOP - TOP_H                 # bottom of the upper tier
    y_low_top = y_top_row - PANEL_GAP - 0.12   # (e), (f) names are two lines

    # (a): the first column, full height, drawn at AC_SCALE and centred
    ax_task = add_ax(fig, A_X0, BOTTOM, 1.0, 1.0)          # resized below
    x_lo, x_hi, y_lo, y_hi = draw_a_column(ax_task, task,
                                           height=(TOP - BOTTOM - 0.10) / AC_SCALE)
    w_a, h_a = (x_hi - x_lo) * AC_SCALE, (y_hi - y_lo) * AC_SCALE
    y_a = BOTTOM + (TOP - BOTTOM - h_a) / 2
    ax_task.set_position([A_X0 / FIG_W, y_a / FIG_H, w_a / FIG_W, h_a / FIG_H])
    ax_task.set_xlim(x_lo, x_hi)
    ax_task.set_ylim(y_lo, y_hi)
    ax_task.set_aspect("equal")
    ax_task.axis("off")
    diag("(a) drawn %.2f x %.2f in" % (w_a, h_a))

    # (c): the withholding rates, under (a)
    order = model_rows(prm)
    ax_c = add_ax(fig, B_X0, y_top_row, B_W, TOP_H)
    arm_panel(ax_c, prm, C_COL, C_SYM, order, key=True,
              xlabel=f"{C_WORD}\n{C_SYM}", marker="o")
    ax_c.xaxis.label.set_linespacing(1.0)
    # 9/24: the threshold c* beside the rate, on the same rows, so a rate is
    # read against its own threshold along one row
    ax_cs = add_ax(fig, B_X0 + B_W + CS_GAP, y_top_row, B_W, TOP_H)
    arm_panel(ax_cs, prm, "c_star_post", "$c^{*}$", order, key=False,
              xlabel="threshold\n$c^{*}$", marker="o")
    ax_cs.xaxis.label.set_linespacing(1.0)
    ax_cs.set_ylim(ax_c.get_ylim())
    ax_cs.patch.set_visible(False)   # (b)'s key runs on over this panel
    ax_cs.set_yticklabels([])

    # (d): the distance from the threshold against the gain, full height
    ax_d = add_ax(fig, D_X0, BOTTOM + D_KEY_H, D_W,
                  y_low_top - BOTTOM - D_KEY_H)
    pts = gain_points(prm, accall)
    off = gain_points_off()
    fit = gain_panel(ax_d, pts, off)

    # (e)-(g): fig_B3's panels less its withholding row. Every model at its
    # main setting (filled), and beside it on the same row, for the six with
    # a reasoning-off run, that run -- striped, as in (d) (9/24). (e) and (f)
    # share their rows, so (f) carries no names.
    b3.ARM_ONLY = "A"                # instruction A only, not pooled (9/24)
    data3 = b3.load_main()
    pairs = b3.load()
    b3.ARM_ONLY = None
    for m in b3.PAIRS:
        data3[m]["off"] = pairs[m]["off"]
    order3 = b3.row_order(data3)
    quant3 = {q[0]: q for q in b3.QUANTITIES}
    strip3 = b3.KEY_STRIP + POOL_STRIP

    def striped(ax, x, y):
        off_mark(ax, x, y, "o", "A", ms=b3.MS + 0.9)   # A: blue stripes

    # (e)-(g) read instruction A alone, so they draw in A's blue (9/24)
    b3.ON_WHISKER, b3.OFF_WHISKER = fs.BLUE, "#4d9cc9"
    ax_e = add_ax(fig, C_X0, y_top_row, C_W_HALF, TOP_H)
    ax_f = add_ax(fig, C_X0 + C_W_HALF + EF_GAP, y_top_row, C_W_HALF, TOP_H)
    # (g) gives (d) some of the lower tier's width: (d) carries its key inside
    ax_g = add_ax(fig, G_X0, BOTTOM, C_X0 + 2 * C_W_HALF + EF_GAP - G_X0,
                  y_low_top - BOTTOM)
    for ax, key in ((ax_e, "a"), (ax_f, "gamma"), (ax_g, "gain")):
        col, how, _, name = quant3[key]
        ax.patch.set_visible(False)
        # (e)-(g) keep the models that can switch reasoning off (9/24)
        rows = [m for m in order3 if m in b3.PAIRS]
        b3.panel(ax, data3, rows, col, how, name,
                 strip=strip3 if ax is not ax_g else 0.0, off_mark=striped,
                 on_style=ON_OPEN)
    ax_f.set_yticklabels([])
    # side by side, each name takes two lines: word over symbol
    ax_e.set_xlabel("internalization\n$\\hat{a}$", labelpad=1, linespacing=1.0)
    ax_f.set_xlabel("net correction\n$\\hat{\\gamma}$", labelpad=1,
                    linespacing=1.0)
    ax_f.tick_params(axis="y", length=2)
    top_row = -b3.ROW_FOOT - strip3
    # key: reasoning on (filled) and off (striped), then the pooling note
    ky = top_row + 0.5
    ax_e.plot([0.03], [ky], transform=ax_e.get_yaxis_transform(), marker="o",
              ls="none", clip_on=False, zorder=4, **ON_OPEN)
    ax_e.annotate("reasoning on", (0.03, ky), xycoords=ax_e.get_yaxis_transform(),
                  xytext=(3, 0), textcoords="offset points", fontsize=fs.FS_SMALL,
                  color=fs.INK2, va="center", ha="left", annotation_clip=False)
    kx_off = 0.03 + 0.78 / C_W_HALF
    off_mark(ax_e, kx_off, ky, "o", "A", transform=ax_e.get_yaxis_transform(),
             ms=b3.MS + 0.9)
    ax_e.annotate("off", (kx_off, ky), xycoords=ax_e.get_yaxis_transform(),
                  xytext=(3, 0), textcoords="offset points", fontsize=fs.FS_SMALL,
                  color=fs.INK2, va="center", ha="left", annotation_clip=False)
    ax_e.text(0.03, top_row + 1.55, "instruction A",
              transform=ax_e.get_yaxis_transform(), fontsize=fs.FS_SMALL,
              color=fs.INK2, va="center", ha="left")

    # lettered in reading order, left half then right half (9/24)
    panels = [(ax_task, "a"), (ax_c, "b"), (ax_cs, "c"), (ax_d, "d"),
              (ax_e, "e"), (ax_f, "f"), (ax_g, "g")]
    for ax, lt in panels:
        box = ax.get_position()
        dx = LETTER_DX if ax is not ax_task else 0.0
        if ax in (ax_c, ax_e, ax_g):
            dx = NAMES_DX
        if ax is ax_d:
            dx = 0.45
        letter(fig, box.x0 * FIG_W - dx, box.y1 * FIG_H + 0.13, lt)
    solid_thin_marks(fig)
    d_model_key(fig, ax_d, order)
    caption(task, prm, accall, pts, off, fit, data3, order3)
    for line in NOTES:
        print(line)
    return fig


if __name__ == "__main__":
    fig = make()
    check_layout(fig)
    fs.save(fig, "F2")
