"""F1: (a) the team and what one agent sees, (b) one agent's round, (c) the
phase diagram.

Layout: 5.5 x 2.22 in, three panels in a row, (a) 1.50 in, (b) 1.70 in,
(c) 1.36 in.

Run from anywhere::

    snakemake --rerun-triggers mtime -c1 figures

Panels (a) and (b) are schematics drawn with matplotlib patches; each sits on
its own axes whose data coordinates are inches, so the drawing is laid out in
the same units as the figure.  They carry no data, so they carry no interval.
Colour is not their only channel: a mark for the right answer is filled and a
mark for a wrong one is open, so (a) survives being printed in greyscale, where
blue and vermillion are 1.3:1 apart (spec section 1).

Panel (c) reads every number it draws:

* ``abm_phase.csv`` -- the simulated share of groups ending right on the
  (c, p) grid.  Group size, grid shape, replicates per cell and the number of
  rounds are read off the file, never typed in here.  Each cell is a proportion
  over replicates, so the plane is an estimate; a heat map has no envelope to
  fill, so spec section 3 is met by stating the per-cell standard error in the
  caption.  The simulation's own 50 % crossing is computed (:func:`crossing`)
  and compared with the theory in the caption, but is NOT drawn: with 1000
  replicates a cell its +/- 2 s.e. interval is thinner than the 1.2 pt boundary
  curve over most of the plane, so drawing it would put a mark in the panel
  that no reader can see, and F1(c) allows nothing in the panel that is not the
  plane, the boundary and the two references;
* ``meanfield_boundary.csv`` -- the boundary p*(c) and the absorbing threshold
  c*, for the standard setting: proportional trigger, ``lam`` = 0.5, q = 3.
  ``lam`` is the library's name for the paper's internalization rate ``a``
  (``cbrm.phase.as_params`` maps ``a`` onto ``Params.lam``), so the caption
  writes it as ``a``, exactly as it writes the file's ``phi`` as ``c``.

Every sentence about the figure lives in ``results/figures/captions/F1.tex``; the panels
carry axis names, tick labels, panel letters, the words of the schematics and
(a)'s legend, and nothing else. The legend names the two marks in the caption's
own words, so a reader who starts at the picture is not sent to the caption to
learn which circle is the statement and which the belief.  Notation follows spec section 8: statement ``y``, belief
``z``, withholding rate ``c``, internalization rate ``a``, dissent ``d``,
critical withholding rate ``c*``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs

BOUNDARY = fs.THEORY_DIR / "meanfield_boundary.csv"
ABM = fs.THEORY_DIR / "abm_phase.csv"
ABM_N = 100      # group size shown in (c); the sweep also stores N = 400

# The standard setting of the mean-field model, named with the csv's own column
# names so the filter below is readable: proportional reconsideration trigger,
# lam = the internalization rate a = 0.5, q = 3 visible neighbours.
TRIGGER, LAM, Q = "fraction", 0.5, 3

LEGEND_BAND = False         # the LaTeX caption carries every word of prose

LW_BOUNDARY = 1.2           # p*(c) over the plane; the caption compares the
                            # simulated crossing's interval against it

FS = fs.FS_NOTE             # 7 pt: the words inside a schematic

# The claim both schematics make is about right answers and wrong ones, so
# blue and vermillion are the only two colours they spend (spec section 1).
RIGHT, WRONG = fs.BLUE, fs.VERM
# ... and colour is never the only channel: right is filled, wrong is open.
EDGE_LW = 1.0
# Arrows, connectors and the frames of the boxes are chrome, not data.
CHROME = fs.INK3
NODE_BOX = dict(boxstyle="round,pad=0.20,rounding_size=0.05", fc=fs.SURFACE,
                ec=CHROME, lw=0.6)


# ------------------------------------------------------------------- helpers
def _extent(artist, fig, ax):
    """Data-coordinate (x0, x1, y0, y1) of a drawn text or of its box."""
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    if hasattr(artist, "get_bbox_patch") and artist.get_bbox_patch() is not None:
        bb = artist.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, x1, y0, y1


def arrow(ax, p0, p1, color, lw=1.2, head=6, zorder=2, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 zorder=zorder, joinstyle="round",
                                 capstyle="round", **kw))


def edge_label(ax, x, y, text, ha="left", va="center", size=None):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size or FS, color=fs.INK2, zorder=6)


def schematic_axes(fig, x0, w, W, H, y0=0.0, h=None):
    """An axes whose data coordinates are inches, ``w`` x ``h`` at ``(x0, y0)``."""
    h = H - y0 if h is None else h
    ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def letter(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="top", fontsize=fs.FS_AXIS,
            fontweight="bold", color=fs.INK, zorder=8)


# ------------------------------------------------------------ (a) the team
# (belief z, statement y) for the five agents; agent 0 is the focal one, and it
# believes the right answer while stating the wrong one -- one withholding.
STATES = [(1, 0), (0, 0), (1, 1), (0, 0), (0, 0)]
SEEN = [1, 2, 4]                 # the three teammates whose statements i reads

R_AGENT = 0.165                  # the disc, small enough that the arrows between
                                 # the agents are the length of a mark and not of
                                 # the gap left over between two large circles
LEG_EC = fs.INK2                 # the legend keys shapes, so it spends no colour


def mark(right: bool) -> dict:
    """Fill/edge for one answer: filled if it is right, open if it is wrong.

    Colour alone would not survive greyscale -- blue #0072b2 and vermillion
    #d55e00 are grey bytes 108 and 129, 1.34:1 apart -- so the second channel of
    spec section 2 carries the same distinction: a right answer is a filled
    disc, a wrong one is a white disc with a 1 pt coloured edge.
    """
    return (dict(fc=RIGHT, ec=RIGHT, lw=EDGE_LW) if right
            else dict(fc=fs.SURFACE, ec=WRONG, lw=EDGE_LW))


def agent(ax, x, y, belief, statement, r, focal=False):
    """Ring = statement y (visible), inner disc = belief z (private)."""
    outer = mark(statement)
    if focal:
        outer["lw"] = 1.8              # agent i is heavier, not another colour
    ax.add_patch(Circle((x, y), r, zorder=4, **outer))
    inner = mark(belief)
    if belief:                                    # keep a filled belief legible
        inner["ec"] = fs.SURFACE                  # inside a filled statement
        inner["lw"] = 0.7
    ax.add_patch(Circle((x, y), r * 0.52, zorder=5, **inner))


def read_legend(fig, ax, x0, y0, w):
    """How to read the marks: the ring, the disc it encloses, and the squares.

    The panel draws two shapes and nothing in it says which is which, so the
    reader had to reach the caption before the drawing meant anything. Leaders
    point at the ring and at the disc inside it; the square of the field above
    stands beside them with its own words under it and no leader, because a mark
    sitting on the phrase that names it needs no line to join the two. The three
    glyphs are drawn in ink: they key the *shape*, while fill and colour go on
    meaning right and wrong, which is the panel's own claim.

    The words are the caption's -- "public statement y", "private belief z", and
    the squares are the statements i reads -- so the same phrase meets the
    reader twice instead of two phrases meaning one thing. Each label breaks
    over two lines: the block is then as wide as its widest word rather than as
    its whole phrase, which is what leaves the square room to stand beside the
    circle instead of on a row of its own under it.
    """
    R, ri, sq = 0.135, 0.068, 0.13
    gx, gy = x0 + R + 0.01, y0 + 0.27
    tx = gx + R + 0.055                  # where the two circle labels start
    ax.add_patch(Circle((gx, gy), R, fc="none", ec=LEG_EC, lw=EDGE_LW, zorder=4))
    ax.add_patch(Circle((gx, gy), ri, fc="none", ec=LEG_EC, lw=EDGE_LW, zorder=5))
    # Two lines each, and far enough apart that the four lines read as two
    # labels and not as one paragraph.
    for dy, rad, text in ((0.145, R, "public\nstatement $y$"),
                          (-0.145, ri, "private\nbelief $z$")):
        s = 0.7071                       # the leader leaves the mark at 45 deg
        ax.plot([gx + rad * s, tx - 0.035], [gy + np.sign(dy) * rad * s, gy + dy],
                color=LEG_EC, lw=0.6, zorder=3, solid_capstyle="round")
        ax.text(tx, gy + dy, text, ha="left", va="center", fontsize=fs.FS_SMALL,
                color=fs.INK2, linespacing=1.15, zorder=6)
    # The square stands top-right of the circle, its words underneath: a mark
    # sitting on the phrase that names it needs no leader. It rides ``lift``
    # above the circle's own top so its second line clears the floor of the
    # panel instead of sitting on it.
    lift = 0.07
    sq_top = gy + R + lift
    t = ax.text(w - 0.03, sq_top - sq - 0.055, "a statement\n$i$ reads",
                ha="right", va="top", ma="center", fontsize=fs.FS_SMALL,
                color=fs.INK2, linespacing=1.15, zorder=6)
    fig.canvas.draw()
    sx0, sx1, _, _ = _extent(t, fig, ax)
    ax.add_patch(FancyBboxPatch(((sx0 + sx1 - sq) / 2, sq_top - sq), sq, sq,
                                boxstyle="square,pad=0", fc="none", ec=LEG_EC,
                                lw=EDGE_LW, zorder=4))


def draw_team(fig, ax, w, h):
    """Five agents, the three statements i reads, the majority they imply, and
    the legend that names the marks.

    The five sit on an ellipse rather than a circle because the panel is narrow
    and tall. ``Rx`` is what the width allows and ``Ry`` is set from the height
    left over once the ``i sees`` field and the legend have taken theirs, which
    is why the numbers here are not round.
    """
    leg_h = 0.54                                  # the legend, on the floor
    read_legend(fig, ax, 0.02, 0.0, w)

    r = R_AGENT
    Rx, Ry = 0.46, 0.40
    bw, bh, by0 = min(1.26, w - 0.10), 0.42, leg_h + 0.06
    cx, cy = w / 2, h - 0.03 - (0.809 * Ry + r)
    angs = np.deg2rad([270, 342, 54, 126, 198])       # agent i at the bottom
    pos = [(cx + Rx * np.cos(t), cy + Ry * np.sin(t)) for t in angs]
    for k, (x, y) in enumerate(pos):
        agent(ax, x, y, *STATES[k], r=r, focal=(k == 0))
    fx, fy = pos[0]
    for k in SEEN:                                     # who i can see
        x, y = pos[k]
        dx, dy = fx - x, fy - y
        d = np.hypot(dx, dy)
        ux, uy = dx / d, dy / d
        arrow(ax, (x + ux * (r + 0.03), y + uy * (r + 0.03)),
              (fx - ux * (r + 0.05), fy - uy * (r + 0.05)), CHROME, lw=1.0, head=7)
    # The one glyph that names the agent the panel is about. A lone italic "i"
    # is mostly dot and stem, so at the 7 pt of the other words it reads smaller
    # than they do; it takes the 8 pt of an axis name, the top of the type scale
    # (spec section 7).
    ax.text(fx - r - 0.07, fy, "$i$", ha="right", va="center",
            fontsize=fs.FS_AXIS, color=fs.INK, zorder=6)

    # what i sees: the three statements it read, and their majority m_i
    bx0 = (w - bw) / 2
    # a neutral area, so FILL and no frame -- spec section 4 wants no boxes
    ax.add_patch(FancyBboxPatch((bx0, by0), bw, bh,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fs.FILL, ec="none", zorder=3))
    ax.text(bx0 + bw / 2, by0 + bh - 0.115, "$i$ sees", ha="center",
            va="center", fontsize=FS, color=fs.INK, zorder=6)
    sq, gap = 0.16, 0.045
    row_y = by0 + 0.07
    run = 3 * sq + 2 * gap + 0.17 + sq + 0.20
    s0 = bx0 + (bw - run) / 2
    for j, k in enumerate(SEEN):
        ax.add_patch(FancyBboxPatch((s0 + j * (sq + gap), row_y), sq, sq,
                                    boxstyle="square,pad=0", zorder=6,
                                    **mark(STATES[k][1])))
    ax.text(s0 + 3 * sq + 2 * gap + 0.085, row_y + sq / 2, r"$\rightarrow$",
            ha="center", va="center", fontsize=FS, color=fs.INK2, zorder=6)
    mx = s0 + 3 * sq + 2 * gap + 0.17
    ax.add_patch(FancyBboxPatch((mx, row_y), sq, sq, boxstyle="square,pad=0",
                                zorder=6, **mark(0)))
    ax.text(mx + sq + 0.05, row_y + sq / 2, "$m_i$", ha="left", va="center",
            fontsize=FS, color=fs.INK, zorder=6)
    arrow(ax, (fx, fy - r - 0.04), (fx, by0 + bh + 0.03), CHROME, lw=1.0, head=7)
    letter(ax, 0.02, h - 0.02, "(a)")


# ------------------------------------------------------------ (b) the round
FS_RATE = fs.FS_AXIS        # the rate symbols carry the model, so they get the
                            # 8 pt of an axis name; "yes"/"no" and the gloss are
                            # flow words and stay at 7 pt


# (b) is the same agent as (a), one round later, so it is drawn in the same
# marks: a ring pair is (statement, belief) and a square is a statement read off
# the panel. The nodes used to be words -- "state m_i", "adopt m_i" -- which
# said in text what the marks of (a) already say in shape, and made the reader
# carry a second notation for the same two quantities. What stays in words is
# what is not a state: the re-derivation, which is an event, and the answer it
# lands on.
B_R_OUT, B_R_IN = 0.155, 0.12
B_SQ = 0.16
B_TOP_Y = 1.92          # the condition the fork turns on, m_i against z_i
B_TOP_X = (0.62, 1.08)
B_LAB_DY = 0.18         # a mark to the symbol under it
B_COL = (0.42, 1.28)    # the withholding path, and the correction path
B_ROW1, B_ROW2 = 1.24, 0.72
B_WORD_Y, B_OUT_Y = 0.86, 0.40   # "reconsiders", and the answer it reaches
B_RATE_DX = 0.08


def pair(ax, x, y, said, held, r_out=B_R_OUT, r_in=B_R_IN):
    """One agent: the ring is what it states, the disc inside what it believes."""
    ax.add_patch(Circle((x, y), r_out, zorder=4, **mark(said)))
    inner = mark(held)
    if held:
        inner["ec"] = fs.SURFACE
        inner["lw"] = 0.7
    ax.add_patch(Circle((x, y), r_in, zorder=5, **inner))


def draw_round(fig, ax, w, h):
    """One agent's round: the withholding path and the correction path.

    The agent of (a) again: it holds the right answer and the majority it can
    see holds the wrong one, which is the only configuration in which either
    rate does anything. Reading down, the round is two forks. The first is the
    public one -- with probability c the agent states the majority rather than
    what it believes, which is the withholding, and its ring turns while its
    disc does not. The second is what the belief then does, and it is a
    different channel on each side: an agent that withheld internalizes what it
    said with probability a, and an agent that spoke re-derives its answer at a
    rate set by the dissent it can see and lands on the right answer with
    probability r. Colour follows the path: vermillion is the answer that is
    wrong, blue the one that is right.
    """
    xm = w / 2
    x_sq, x_disc = B_TOP_X

    # the condition the round turns on: the visible majority against the belief
    ax.add_patch(FancyBboxPatch((x_sq - B_SQ / 2, B_TOP_Y - B_SQ / 2), B_SQ, B_SQ,
                                boxstyle="square,pad=0", zorder=4, **mark(False)))
    ax.add_patch(Circle((x_disc, B_TOP_Y), B_R_IN, zorder=4, **mark(True)))
    ax.text(xm, B_TOP_Y, r"$\neq$", ha="center", va="center", fontsize=FS,
            color=fs.INK, zorder=6)
    for x, lab in zip(B_TOP_X, ("$m_i$", "$z_i$")):
        ax.text(x, B_TOP_Y - B_LAB_DY, lab, ha="center", va="center",
                fontsize=FS, color=fs.INK, zorder=6)

    # the public fork
    y_stub, y_arm = B_TOP_Y - B_LAB_DY - 0.10, B_ROW1 + B_R_OUT + 0.10
    ax.plot([xm, xm], [y_stub, y_arm], color=CHROME, lw=0.8, zorder=2,
            solid_capstyle="round")
    elbow = "angle,angleA=0,angleB=90,rad=0"
    for x, colour, rate in ((B_COL[0], WRONG, "$c$"), (B_COL[1], RIGHT, "$1-c$")):
        arrow(ax, (xm, y_arm), (x, B_ROW1 + B_R_OUT + 0.02), colour, lw=1.3,
              connectionstyle=elbow)
        edge_label(ax, (xm + x) / 2, y_arm + 0.045, rate, ha="center", va="bottom")

    pair(ax, B_COL[0], B_ROW1, False, True)    # states the majority, believes its own
    pair(ax, B_COL[1], B_ROW1, True, True)     # states what it believes

    # withholding path: the belief follows what was said, with probability a
    arrow(ax, (B_COL[0], B_ROW1 - B_R_OUT - 0.02),
          (B_COL[0], B_ROW2 + B_R_OUT + 0.02), WRONG, lw=1.3)
    edge_label(ax, B_COL[0] + B_RATE_DX, (B_ROW1 + B_ROW2) / 2, "$a$")
    pair(ax, B_COL[0], B_ROW2, False, False)
    ax.text(B_COL[0], B_ROW2 - B_R_OUT - B_LAB_DY + 0.05, r"adopts $m_i$",
            ha="center", va="center", fontsize=FS, color=fs.INK, zorder=6)

    # correction path: the dissent it can see buys a re-derivation
    word = ax.text(B_COL[1], B_WORD_Y, "re-derives", ha="center", va="center",
                   fontsize=FS, color=fs.INK, zorder=6)
    fig.canvas.draw()
    _, _, wy0, wy1 = _extent(word, fig, ax)
    arrow(ax, (B_COL[1], B_ROW1 - B_R_OUT - 0.02), (B_COL[1], wy1 + 0.05),
          RIGHT, lw=1.3)
    edge_label(ax, B_COL[1] + B_RATE_DX, (B_ROW1 - B_R_OUT + wy1) / 2,
               r"$\rho\, d_i$")
    arrow(ax, (B_COL[1], wy0 - 0.05), (B_COL[1], B_OUT_Y + B_R_OUT + 0.02),
          RIGHT, lw=1.3)
    edge_label(ax, B_COL[1] + B_RATE_DX, (wy0 + B_OUT_Y + B_R_OUT) / 2, "$r$")
    pair(ax, B_COL[1], B_OUT_Y, True, True)
    ax.text(B_COL[1], B_OUT_Y - B_R_OUT - B_LAB_DY + 0.05, "right answer",
            ha="center", va="center", fontsize=FS, color=fs.INK, zorder=6)

    letter(ax, 0.02, h - 0.02, "(b)")


# ------------------------------------------------------------ (c) phase plane
def load_boundary():
    """p*(c) and c* for the standard setting, straight out of the csv.

    ``phi`` is the file's name for the withholding rate c, ``p_c`` for the
    boundary p*(c) and ``phi_absorbing`` for the threshold c*.
    """
    bd = pd.read_csv(BOUNDARY)
    bd = bd[(bd["trigger"] == TRIGGER) & (bd["lam"] == LAM) & (bd["q"] == Q)]
    if bd.empty:
        raise SystemExit(f"no rows for trigger={TRIGGER}, lam={LAM}, q={Q} in {BOUNDARY}")
    bd = bd.sort_values("phi").reset_index(drop=True)
    c_star = bd["phi_absorbing"].unique()
    if len(c_star) != 1:
        raise SystemExit(f"c* is not constant over the standard setting: {c_star}")
    return bd["phi"].to_numpy(), bd["p_c"].to_numpy(), float(c_star[0])


def load_abm(N=ABM_N):
    """Simulated share of groups ending right on the (c, p) grid, and its shape.

    ``accuracy`` is the share of independent groups whose final expressed
    majority is correct, over ``n_reps`` replicates per cell.
    """
    d = pd.read_csv(ABM)
    d = d[(d["trigger"] == TRIGGER) & (d["N"] == N)]
    if d.empty:
        raise SystemExit(f"no ABM rows for trigger={TRIGGER}, N={N} in {ABM}")
    grid = d.pivot_table(index="p", columns="phi", values="accuracy")
    reps = sorted(d["n_reps"].unique())
    rounds = sorted(d["T"].unique())
    sizes = sorted(d["N"].unique())
    if len(reps) != 1 or len(rounds) != 1 or len(sizes) != 1:
        raise SystemExit(f"the sweep mixes n_reps={reps}, T={rounds} or N={sizes}")
    return (grid.columns.to_numpy(float), grid.index.to_numpy(float),
            grid.to_numpy(float), int(reps[0]), int(rounds[0]), int(sizes[0]))


def crossing(cg, pg, acc, n_reps):
    """Where the simulated share crosses 1/2 down each c column, and its s.e.

    Every cell is a proportion over ``n_reps`` replicates, so the plane is an
    estimate and the crossing read off it is one too (spec section 3). Down a
    column the share falls through 1/2 between two grid rows; linear
    interpolation between them gives p-hat, and the delta method carries the
    two bracketing cells' binomial standard errors through that interpolation:
    with ``t`` the interpolation weight and ``s`` the slope in accuracy per
    unit p, ``var(p-hat) = ((1-t)^2 se_i^2 + t^2 se_{i+1}^2) / s^2``.

    Returns the columns that cross, p-hat, and its standard error.
    """
    cs, phat, se = [], [], []
    for j, cc in enumerate(cg):
        col = acc[:, j]
        hit = np.where((col[:-1] - 0.5) * (col[1:] - 0.5) < 0)[0]
        if hit.size == 0:
            continue
        i = int(hit[-1])
        lo, hi = col[i], col[i + 1]
        t = (0.5 - lo) / (hi - lo)
        dp = pg[i + 1] - pg[i]
        slope = (hi - lo) / dp
        v_lo, v_hi = lo * (1 - lo) / n_reps, hi * (1 - hi) / n_reps
        cs.append(cc)
        phat.append(pg[i] + t * dp)
        se.append(np.sqrt((1 - t) ** 2 * v_lo + t ** 2 * v_hi) / abs(slope))
    return np.array(cs), np.array(phat), np.array(se)


def crossing_stats(cs, phat, se, c, pstar, p_step, lw_p):
    """Everything the caption says about the simulated crossing, off the data.

    The crossing of :func:`crossing` is an estimate of where the simulated
    plane falls through 1/2; ``p*(c)`` is the mean-field boundary at the same c.
    The two are compared column by column, and the comparison is split at the
    columns where the boundary is above zero, because below c* the mean-field
    boundary is identically zero and nothing the simulation puts there can sit
    inside an interval around it.

    ``lw_p`` is the width of the drawn boundary curve -- ``LW_BOUNDARY`` points
    on the rendered panel -- expressed in p, so "narrower than the curve" is a
    measured statement and not an impression.
    ``p_step`` is the grid's own resolution in p, which is what limits the
    crossing below c*.
    """
    target = np.interp(cs, c, pstar)
    keep = target > 0
    lo, hi = phat - 2 * se, phat + 2 * se
    outside = keep & ((target < lo) | (target > hi))
    excess = np.maximum(lo - target, target - hi)
    if not outside.any():                       # nothing to report a miss for
        worst = dict(n_out=0, c_out_lo=float("nan"), c_out_hi=float("nan"),
                     excess=0.0, c_worst=float("nan"))
    else:
        worst = dict(n_out=int(outside.sum()),
                     c_out_lo=float(cs[outside].min()),
                     c_out_hi=float(cs[outside].max()),
                     excess=float(excess[outside].max()),
                     c_worst=float(cs[outside][int(np.argmax(excess[outside]))]))
    return dict(
        n_cols=int(cs.size),
        half=float(np.mean(2 * se)), half_max=float(np.max(2 * se)),
        width_max=float(np.max(4 * se)), p_step=float(p_step),
        n_thin=int(np.sum(4 * se < lw_p)),
        n_keep=int(keep.sum()), n_in=int((keep & ~outside).sum()),
        gap=float(np.mean(np.abs(phat - target)[keep])),
        n_below=int((~keep).sum()),
        p_below_lo=float(phat[~keep].min()), p_below_hi=float(phat[~keep].max()),
        n_below_step=int(np.sum(phat[~keep] < p_step)),
        **worst)


def draw_phase(fig, ax, ax_cb):
    c, pstar, c_star = load_boundary()
    cg, pg, acc, n_reps, T, n_agents = load_abm()

    def edges(v):
        m = 0.5 * (v[1:] + v[:-1])
        return np.concatenate([[v[0] - (m[0] - v[0])], m, [v[-1] + (v[-1] - m[-1])]])

    # Vermillion (every group ends wrong) through white (half do) to the accent
    # blue (all do): the two colours (a) and (b) spend on a wrong answer and a
    # right one, so the plane's low end reads as the adverse pole and not as an
    # empty field. Both ends stop at the accents, so the 1.2 pt black p*(c) over
    # the plane keeps 3.4:1 on blue and 4.7:1 on vermillion.
    cmap = fs.div_wrong_right()
    mesh = ax.pcolormesh(edges(cg), edges(pg), acc, cmap=cmap, vmin=0, vmax=1,
                         shading="flat", rasterized=True, zorder=1)

    # the plane is an estimate; its own 50 % crossing is computed for the
    # caption but not drawn -- see the module docstring and crossing_stats.
    cs, phat, se = crossing(cg, pg, acc, n_reps)

    # theory over the simulation; it is a curve, not an estimate, so no band
    i0 = int(np.argmax(pstar > 0))
    seg = slice(max(i0 - 1, 0), None)
    ax.plot(c[seg], pstar[seg], color=fs.INK, lw=LW_BOUNDARY,
            solid_joinstyle="round", solid_capstyle="butt", zorder=5)
    fs.reference(ax, 0.5, "h", zorder=4.5)
    fs.reference(ax, c_star, "v", zorder=4.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fs.ticks01(ax, "both")
    ax.tick_params(pad=1.5)                  # the gutter is measured, so is this
    ax.set_xlabel("withholding rate $c$", labelpad=1)
    ax.set_ylabel("initial accuracy $p$", labelpad=1)
    fs.panel_label(ax, "(c)", x=-0.20, y=1.09)

    cb = fig.colorbar(mesh, cax=ax_cb, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels([fs.fmt_tick(v) for v in (0, 0.5, 1)])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.6, pad=1.5, labelsize=fs.FS_NOTE,
                      color=fs.AXIS, labelcolor=fs.INK2)
    cb.set_label("share of groups ending right", fontsize=fs.FS_NOTE, labelpad=2,
                 color=fs.INK)

    # the drawn boundary is LW_BOUNDARY points wide; the axes is one unit of p
    # tall, so this is that linewidth measured in p on the rendered panel.
    h_in = ax.get_position().height * fig.get_size_inches()[1]
    lw_p = (LW_BOUNDARY / 72.0) / h_in
    i_above = int(np.argmax(c > c_star))
    return dict(c_star=c_star, n_reps=n_reps, T=T, N=n_agents,
                n_c=len(cg), n_p=len(pg), lw_p=lw_p,
                p_below=float(np.max(pstar[c <= c_star])),
                c_above=float(c[i_above]), p_above=float(pstar[i_above]),
                p_end=float(pstar[-1]),
                acc_half=float(acc[int(np.argmin(np.abs(pg - 0.5))),
                                   int(np.argmin(np.abs(cg - 1.0)))]),
                cross=crossing_stats(cs, phat, se, c, pstar,
                                     float(pg[1] - pg[0]), lw_p))


# ------------------------------------------------------------------- caption
def caption(info):
    c_star, n_reps, T = info["c_star"], info["n_reps"], info["T"]
    n_agents, x = info["N"], info["cross"]
    se = 0.5 / np.sqrt(n_reps)            # the widest binomial s.e. of a cell
    fs.write_caption("F1", [
        r"\textbf{One withheld statement decides whether a team recovers.}",
        r"(a) A team of five. Each agent is drawn as a ring, its public statement"
        r" $y$, around a disc, its private belief $z$; the right answer is blue"
        r" and filled, a wrong one vermillion and open, so fill as well as"
        r" colour separates them in a greyscale print. Agent $i$,"
        r" drawn with the heavier ring, believes the right answer and states the"
        r" wrong one -- one instance of withholding."
        r" Arrows mark the three teammates whose statements $i$ can read, and the"
        r" grey field below holds those three statements and the visible majority"
        r" $m_i$ they imply.",
        r"(b) What $i$ does with the visible majority, in the marks of (a). The"
        r" round turns on the condition at the top: the majority $i$ can see"
        r" holds the wrong answer and $i$ holds the right one, which is the only"
        r" configuration in which either rate does anything; where $m_i$ agrees"
        r" with $z_i$ there is nothing to withhold and the agent simply states"
        r" what it believes. Two forks follow, and they are two different"
        r" channels rather than two steps of one. The first is public:"
        r" vermillion is the \emph{withholding} path, taken with probability"
        r" $c$, where the ring turns to the majority while the disc inside it"
        r" does not -- the agent states $m_i$ and still believes $z_i$ -- and"
        r" blue is the path where it states what it believes. The second is what"
        r" the belief then does. An agent that withheld internalizes what it"
        r" said with probability $a$, and both of its marks end on the wrong"
        r" answer. An agent that spoke re-derives its answer at a rate"
        r" $\rho\,d_i$ set by the fraction $d_i$ of visible dissent it faces,"
        r" and that re-derivation lands on the right answer with probability"
        r" $r$; only that outcome is drawn, as the rate on the arrow already"
        r" says what the other one is. Grey connectors are the flow of the"
        r" round, not quantities.",
        rf"(c) Share of groups whose final expressed majority is correct, from"
        rf" the agent-based model on the $(c, p)$ grid: $N = {n_agents}$ agents,"
        rf" ${info['n_c']} \times {info['n_p']}$ cells, {n_reps} replicates per"
        rf" cell, $T = {T}$ rounds. Every cell is a proportion over its"
        rf" replicates, so the plane is an estimate; a shaded plane has no"
        rf" envelope to draw, so the number is here instead: no cell's standard"
        rf" error exceeds {se:.3f}. The plane runs vermillion where every group"
        rf" ends up holding the wrong answer, through white at half, to blue"
        rf" where every group ends right -- the two colours (a) and (b) spend on"
        rf" a wrong answer and a right one.",
        r"The black curve is the mean-field boundary $p^*(c)$: a group that"
        r" starts with a wrong expressed majority still recovers above it and is"
        r" stuck below it. It solves the mean-field equations for the standard"
        r" setting -- the proportional reconsideration trigger, internalization"
        rf" rate $a = {LAM}$, and each agent seeing the $q = {Q}$ neighbours of"
        rf" (a) -- so it is a curve and not an estimate and carries no interval."
        rf" Up to $c^*$ it never exceeds {info['p_below']:.4f} -- the"
        rf" mean-field team recovers from any $p > 0$ -- one step past it, at"
        rf" $c = {info['c_above']:g}$, it already stands at"
        rf" {info['p_above']:.3f}, and it climbs to {info['p_end']:.3f} at"
        rf" $c = 1$, where nothing is ever expressed against the majority and the"
        rf" plane is just the round-0 vote ({info['acc_half']:.3f} of groups"
        rf" correct at $p = 1/2$).",
        rf"The simulation's own 50\% crossing -- the $p$ at which the share"
        rf" falls through $1/2$, interpolated down each of the {x['n_cols']}"
        rf" columns from the two cells it falls between -- is deliberately"
        rf" \emph{{not}} drawn. With {n_reps} replicates a cell it is pinned to"
        rf" $\pm {x['half']:.3f}$ in $p$ on average and $\pm"
        rf" {x['half_max']:.3f}$ at worst: its $\pm 2$ standard-error envelope"
        rf" is nowhere wider than {x['width_max']:.3f} in $p$, and on"
        rf" {x['n_thin']} of the {x['n_cols']} columns narrower than the"
        rf" {LW_BOUNDARY} pt curve over it, so it would be a band no reader can"
        rf" resolve. Its agreement with the theory is given here instead. Over"
        rf" the {x['n_keep']} columns where $p^*(c)$ is above zero the crossing"
        rf" sits a mean {x['gap']:.3f} in $p$ from the boundary, with $p^*(c)$"
        rf" inside that envelope on {x['n_in']} of them; the other"
        rf" {x['n_out']} are all at the transition, $c = {x['c_out_lo']:g}$ to"
        rf" ${x['c_out_hi']:g}$, and miss by at most {x['excess']:.3f} in $p$"
        rf" at $c = {x['c_worst']:g}$, where the boundary rises almost"
        rf" vertically and the grid's step in $p$ is coarser than the curve it"
        rf" must resolve. Below $c^*$ the comparison degenerates: the boundary"
        rf" is exactly zero, which no envelope around a positive crossing can"
        rf" hold, and a finite team of {n_agents} needs one correct member to"
        rf" have anything to recover from, so on those {x['n_below']} columns"
        rf" the crossing sits between $p = {x['p_below_lo']:.3f}$ and"
        rf" {x['p_below_hi']:.3f} -- {x['n_below_step']} of them inside the"
        rf" grid's first step in $p$, $0$ to ${x['p_step']:g}$.",
        rf"The dashed horizontal is $p = 1/2$, above which the round-0 vote is"
        rf" more likely than not already correct; the dashed vertical is the"
        rf" critical withholding rate $c^* = {c_star:.3f}$, at which the fully"
        rf" falsified consensus becomes stable -- below it the mean-field team"
        rf" recovers from any initial accuracy, above it only from"
        rf" $p > p^*(c)$.",
    ])


# -------------------------------------------------------------------- layout
def fits(fig, ax, gutter, margin):
    """Print what (c)'s y decorations and the colour bar's actually need.

    Spec section 6 asks for a gutter exactly as wide as the tick labels and axis
    name inside it, so the number is measured on the render rather than guessed:
    the left overhang of the y tick labels and the y axis name of ``ax``, and
    the right overhang of the colour bar's own labels.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W = fig.get_size_inches()[0]
    need = {}
    for tag, a in (("left", ax), ("right", fig.axes[-1])):
        bbs = [t.get_window_extent(r) for t in a.get_yticklabels() if t.get_text()]
        bbs.append(a.yaxis.label.get_window_extent(r))
        box = a.get_position()
        need[tag] = (max(box.x0 * W - min(b.x0 for b in bbs) / r.dpi, 0.0)
                     if tag == "left" else
                     max(max(b.x1 for b in bbs) / r.dpi - box.x1 * W, 0.0))
    tight = "" if need["left"] <= gutter and need["right"] <= margin else "  <- clipped"
    print(f"F1: (c) y name + ticks need {need['left']:.3f} in (gutter {gutter:.2f}); "
          f"colour bar labels need {need['right']:.3f} in (margin {margin:.2f})"
          f"{tight}")
    return need


def render():
    fs.apply()
    W, H = fs.WIDTH_IN, 2.22
    # Three in a row (spec section "F1"). (b) gave back the width its node boxes
    # used to pad and (a) took it, because (a) now carries the legend as well as
    # the team; the sum, and (c), are unchanged.
    w_a, w_b, w_c = 1.50, 1.70, 1.36
    # Measured on the render by ``fits``, not guessed: (c)'s y decorations need
    # 0.358 in -- the rotated 8 pt axis name (0.147), the widest 7 pt tick label
    # "0.5" (0.185), the 2 pt tick (0.028) and the 1.5 pt tick pad and 1 pt label
    # pad. Section 6 caps a gutter at 0.30 in, which nothing carrying an 8 pt
    # axis name beside a "0.5" can meet; 0.36 is what actually fits, and at 0.32
    # the axis name ran into (b)'s box. The right margin then holds the colour
    # bar's own labels (0.339 in) inside section 6's 0.45 cap.
    gutter = 0.36
    cb_gap, cb_w = 0.05, 0.10
    x_c = w_a + w_b + gutter
    bottom, top = 0.42, 0.20             # x axis name below, panel letter above
    h_c = H - bottom - top
    margin = W - (x_c + w_c + cb_gap + cb_w)

    fig = plt.figure(figsize=(W, H))
    # (a) and (b) are diagrams with no axis of their own, so they run the full
    # height: (a)'s legend sits on the floor, in the band that (c) spends on its
    # x axis name, instead of costing the figure a row of its own.
    ax_a = schematic_axes(fig, 0.0, w_a, W, H)
    ax_b = schematic_axes(fig, w_a, w_b, W, H)
    ax_c = fig.add_axes([x_c / W, bottom / H, w_c / W, h_c / H])
    ax_cb = fig.add_axes([(x_c + w_c + cb_gap) / W, bottom / H, cb_w / W, h_c / H])

    draw_team(fig, ax_a, w_a, H)
    draw_round(fig, ax_b, w_b, H)
    caption(draw_phase(fig, ax_c, ax_cb))
    fits(fig, ax_c, gutter, margin)
    fs.save(fig, "F1")


if __name__ == "__main__":
    render()
