"""Shared style for the publication figures in ``results/figures``.

Every ``fig_*.py`` script does::

    import figstyle as fs
    fs.apply()
    fig = plt.figure(figsize=(fs.WIDTH_IN, H))   # panels via fig.add_axes, in inches
    ...
    fs.save(fig, "F1")                # -> figs/final/F1.pdf + F1.png (400 dpi),
                                      #    and prints the panel-area fraction
    fs.write_caption("F1", [...])     # -> figs/captions/F1.tex

Rules encoded here:

* **Colour marks the claim; everything else is ink.** A panel spends at most two
  saturated colours, and only on the contrast it exists to make: ``BLUE``
  ``#0072b2`` for the favourable pole (right answer, recovery, correction,
  "discussion helps") and ``VERM`` ``#d55e00`` for the adverse pole (wrong
  answer, trap, internalization, "discussion hurts"). Okabe-Ito blue and
  vermillion survive all three dichromacies, but they separate only weakly in
  greyscale, so colour never carries a distinction alone -- every series is also
  directly labelled. **No green anywhere**, for anything;
* **categories are ink, with one exception** (9/9): a category that a panel
  makes the reader compare mark by mark gets a hue. The three models are that
  exception -- ``MODEL_HUES`` = ``INK`` / ``VERM`` / ``BLUE`` -- because
  ``ink_steps(3)`` put three models x four marks each into a 1.12 in row of
  F2(b) and the row read as one grey mass. Everything else categorical is still
  ``ink_steps``, and a hue never replaces the direct label. Ordinal
  variables (design accuracy ``p``, degree) are one ramp light to dark, every
  step directly labelled: ``seq_blue`` on its own, or ``seq_from(hue, n)`` when
  the ramp sits in a panel that already belongs to one model, so the gradient is
  built on that model's colour instead of contradicting it;
* **a ramp step must be printable**: peers are compared, so every step of both
  ramps clears ``CONTRAST_MIN`` = 2.8:1 on white (``contrast``), and ``apply``
  asserts it. ``INK3`` ``#bdbdbd`` (1.88:1) is not a ramp step at all -- it is
  reserved for reference lines and for context meant to be looked past, so a
  peer series and a ``c*`` rule never share a tone;
* **contrast is a budget**: the emphasised series is darkest, thickest (1.5 pt,
  ``EMPH``) and the only saturated one; true context -- not a peer -- is 0.8 pt
  ink 3 (``CTX``) with open markers. Reference lines (round-0 vote,
  independent-majority contour, ``p = 1/2``) are ink 3, 0.6 pt, dashed and never
  labelled in the panel (``reference``); a rule that belongs to one series, each
  model's own ``c*``, takes that series' colour and a dash pattern of its own.
  Nothing is bold but the panel letter;
* **uncertainty is always drawn, and always the same way**: a filled envelope at
  alpha 0.20 with no edge along a continuous x (``band``), a capless 0.7 pt
  whisker behind the marker at a discrete x (``whisker``), the density itself
  (``FILL`` body, ``INK`` outline) for a posterior. What the interval is goes in
  the caption, never on the figure;
* **nothing in a panel but data**: its letter, axis names, tick labels and a
  direct label per series (``direct_label``, ``spread``). Every sentence --
  glossaries, run parameters, ``n =`` counts, estimates and their intervals --
  moves to ``write_caption``. Spines left and bottom only, no grid, no frames,
  at most three ticks an axis written ``0`` / ``0.5`` / ``1`` (``ticks01``). A
  key naming the MARKS may stand in for direct labels where those cost more than
  they are worth (F2b, F3a, F3f), and is framed only when it sits inside a panel
  whose data it could be mistaken for;
* **type scale 8 / 7 / 6.5 pt**: axis labels and panel letters 8, tick labels
  and direct labels 7, a count that is itself a datum 6.5. Nothing larger than
  the body type beside it, nothing under 6 pt;
* **density is preserved**: the freed space goes to the panels, not to the
  margins. The union of the axes boxes is at least 55 % of the figure area;
  ``density`` measures it and ``save`` prints it on every render;
* output: ``<name>.pdf`` (vector, ``pdf.fonttype`` 42) is what ``main.tex``
  includes, plus ``<name>.png`` at 400 dpi for preview. Caption prose lives in
  ``figs/captions/<name>.tex``, not in a band under the panels.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

WIDTH_IN = 5.5

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------- palette
# Spec section 1, verbatim. Ink is the default; colour is spent on the claim.
INK = "#1a1a1a"          # primary data, the reference series; axis names, letters
INK_MID = "#767676"      # ink 2: a second neutral series
INK2 = "#52514e"         # tick labels, a direct label on a neutral series
INK3 = "#bdbdbd"         # RESERVED: reference lines and true context only (1.88:1)
FILL = "#e6e6e6"         # histogram bodies, posterior densities, neutral areas
BLUE = "#0072b2"         # accent, favourable pole (Okabe-Ito blue)
VERM = "#d55e00"         # accent, adverse pole (Okabe-Ito vermillion)
AXIS = "#c3c2b7"         # spines and tick marks
SURFACE = "#ffffff"

# Deprecated aliases, kept only so the not-yet-rewritten scripts still import.
# The spec's name for the adverse accent is VERM; there is no orange and no
# yellow in rev. 2, and no green at all.
ORANGE = VERM
ORANGE_LINE = VERM
YELLOW = VERM
YELLOW_LINE = VERM
ORANGE_FILL = "#f7d9c2"  # ~35 % vermillion on white -- prefer band(..., VERM)
YELLOW_FILL = ORANGE_FILL
BLUE_FILL = "#cfe2ee"    # ~18 % blue on white -- prefer band(..., BLUE)
GREY = INK3
MUTED = INK_MID
GRID = FILL              # rcParams only; the spec allows no grid

FS_AXIS = 8              # axis label, panel letter (bold)
FS_NOTE = 7              # tick label, direct label
FS_SMALL = 6.5           # a count that is itself a datum

HALO = dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85)

# The two ramps. Both are floored at CONTRAST_MIN against white, because a
# 1 pt line and a filled band below that floor do not survive offset printing:
# rev. 2 dropped #de8f05 at 2.2:1 for exactly this reason, so no ramp step may
# sit there either. INK3 (1.88:1) is deliberately NOT a ramp step; it is the
# reserved reference/context ink, drawn dashed at 0.6 pt where being faint is
# the point. ``apply`` re-checks both ramps on every render.
_INK_RAMP = [INK, "#595959", "#949494"]                     # peers: 17.4 / 7.0 / 3.0
_BLUE_RAMP = ["#4d9cc9", BLUE, "#00446b", "#00283e"]        # ordinal: 3.0 .. 15.3


def _ramp(anchors: list[str], n: int) -> list[str]:
    """``n`` evenly spaced steps of ``anchors``; exact anchors when they divide."""
    if n < 1:
        raise ValueError(f"need at least one step, not {n}")
    if n == 1:
        return [anchors[0]]
    last = len(anchors) - 1
    if n <= len(anchors) and last % (n - 1) == 0:      # hit the anchors exactly
        step = last // (n - 1)
        return [anchors[i * step] for i in range(n)]
    cmap = LinearSegmentedColormap.from_list("ramp", anchors)
    return [to_hex(cmap(t)) for t in np.linspace(0.0, 1.0, n)]


def ink_steps(n: int) -> list[str]:
    """``n`` steps of the neutral ramp, dark -> light: the palette for PEERS.

    For categories the panel needs the reader to compare, every step must be
    legible on its own and not merely present: ``ink_steps(3)``
    is ``#1a1a1a`` (17.4:1 on white), ``#595959`` (7.0:1) and ``#949494``
    (3.0:1), and larger ``n`` interpolates between the same anchors. That is a
    different job from *de-emphasis*: ``INK3`` ``#bdbdbd`` (1.88:1) is reserved
    for reference lines and for context the reader is meant to look past, and is
    deliberately not a step of this ramp -- otherwise a peer series and a ``c*``
    rule share a tone in the same row, as F2(b) did. ``ink_steps(1)`` is the
    primary ink. Every step still needs its own direct label; these are greys,
    and only the label says which category is which. The three models no longer
    come from here -- see ``MODEL_HUES``.
    """
    return _ramp(_INK_RAMP, n)


def seq_blue(n: int) -> list[str]:
    """``n`` ordinal steps of one blue hue, light -> dark, anchored on ``#0072b2``.

    For ORDINAL variables only (design accuracy ``p``, degree): more of them is
    more favourable, so they run light to dark on one hue. The light end stops
    at ``#4d9cc9`` (3.0:1 on white) rather than going paler, so the lowest step
    is still a line and still carries a readable band; ``seq_blue(4)`` and
    ``seq_blue(5)`` both span ``#4d9cc9`` to ``#00283e``. ``seq_blue(1)`` is
    ``BLUE`` itself. Every step is directly labelled -- no colour bar unless the
    shade is a number the reader must read off.
    """
    if n == 1:
        return [BLUE]
    return _ramp(_BLUE_RAMP, n)


# A plane whose low end is the adverse pole, not "nothing here". The two
# accents keep the meaning they carry everywhere else in the paper, and the
# midpoint is white, so the reader sees which side of 1/2 a cell falls on
# without reading the colour bar. Mirrors _BLUE_RAMP's own tints on the
# vermillion side so the two halves are the same weight.
_DIV_RAMP = [VERM, "#e59e66", ORANGE_FILL, SURFACE, BLUE_FILL, "#4d9cc9", BLUE]


def div_wrong_right() -> LinearSegmentedColormap:
    """Vermillion -> white -> blue, for a plane whose 0 is wrong and 1 is right.

    Spec section 1's binary contrast spread over a continuum. A sequential ramp
    off a neutral grey says "nothing here" at its low end, which is wrong for a
    share of groups: 0 does not mean no data, it means every group ended up
    holding the wrong answer, and that is the adverse pole the rest of the paper
    paints vermillion. 1/2 is white so the crossing the panel is about reads off
    the plane itself.
    """
    return LinearSegmentedColormap.from_list("wrong_right", _DIV_RAMP)


# ------------------------------------------------------------------ contrast
# A colour that a printer cannot hold is not a colour. Rev. 2 removed #de8f05
# because 2.2:1 against white vanished as a 1 pt line; the same arithmetic has
# to bind the ramps, or a future edit quietly reintroduces the same failure.
CONTRAST_MIN = 2.8        # floor for any step of ink_steps / seq_blue, on white


def _relative_luminance(colour: str) -> float:
    """WCAG 2.x relative luminance of an sRGB colour (hex, or any name mpl knows)."""
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
           for v in mpl.colors.to_rgb(colour)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(hex_fg: str, hex_bg: str = "#ffffff") -> float:
    """The WCAG contrast ratio between two colours, 1.0 (identical) to 21.0.

    ``contrast("#949494")`` is 3.03: the lightest peer ink. Below roughly 2.8 a
    1 pt line stops surviving offset printing, which is why ``CONTRAST_MIN``
    exists and why ``apply`` refuses to run a ramp that has fallen under it.
    Also useful for checking a band: a fill at alpha ``A`` over white renders at
    ``contrast(blend)``, and ``BAND_ALPHA`` is set so that blend stays visible.
    """
    lo, hi = sorted((_relative_luminance(hex_fg), _relative_luminance(hex_bg)))
    return (hi + 0.05) / (lo + 0.05)


# A ramp on any hue, not just blue: F2(c) runs one ordinal ramp per ROW, and the
# row is a model, so the ramp has to be that model's own colour. Both ends are
# fixed by contrast rather than by a blend factor, because the three model hues
# start at 17.4:1 (ink), 3.9:1 (vermillion) and 5.2:1 (blue) -- one factor would
# put vermillion's light end under the printable floor while leaving ink's dark
# end indistinguishable from its light one.
SEQ_LIGHT, SEQ_DARK = 3.0, 13.0


def _at_contrast(base: str, target: float) -> str:
    """``base`` blended toward white or black until it sits at ``target``:1 on white.

    Which way is decided by where the base already is -- white lowers the ratio,
    black raises it -- so one call gives a ramp's light end and another its dark
    end, whatever hue and whatever lightness it started from.
    """
    here = contrast(base)
    if abs(here - target) < 1e-3:
        return to_hex(to_rgb(base))
    lighten = target < here
    rgb = np.asarray(to_rgb(base))
    toward = 1.0 if lighten else 0.0
    lo, hi = 0.0, 1.0
    for _ in range(48):                       # contrast is monotone in the blend
        t = 0.5 * (lo + hi)
        got = contrast(to_hex(tuple(rgb + (toward - rgb) * t)))
        if (got < target) if lighten else (got > target):
            hi = t
        else:
            lo = t
    return to_hex(tuple(rgb + (toward - rgb) * hi))


def seq_from(base: str, n: int) -> list[str]:
    """``n`` ordinal steps of ``base``'s own hue, light -> dark.

    The generalisation of ``seq_blue`` to the other model hues: same job (an
    ordinal variable, more of it more favourable, every step directly labelled),
    same floor (the light end is pinned at ``SEQ_LIGHT`` = 3.0:1, above
    ``CONTRAST_MIN``), but anchored on whichever colour the row already wears.
    ``seq_from(BLUE, 4)`` is within a step or two of ``seq_blue(4)``.
    """
    if n == 1:
        return [to_hex(to_rgb(base))]
    # Every step is its own tint or shade of the base, spaced evenly in contrast
    # rather than evenly along a line in RGB. Two reasons. Interpolating straight
    # from the light blend to the dark one desaturates the middle -- blue's mid
    # step came out #357aa2, a slate, beside a row name in #0072b2 -- and the row
    # name is the one thing the ramp has to look like. And the three bases sit at
    # 17.4:1, 3.9:1 and 5.2:1, so a ramp laid out in blend units would put its
    # steps in a different place in each row; laid out in contrast, every row
    # climbs 3.0 -> 13.0 alike and the same p reads as the same lightness in all
    # three.
    return [_at_contrast(base, t) for t in
            np.geomspace(SEQ_LIGHT, SEQ_DARK, n)]


def _check_ramps(floor: float = CONTRAST_MIN) -> None:
    """Fail loudly if any ramp step has drifted below the printable floor.

    Checks the anchors and every ``n`` a figure actually asks for, so a step
    that only appears through interpolation is caught too.
    """
    for name, steps in (("ink_steps", _INK_RAMP), ("seq_blue", _BLUE_RAMP)):
        got = list(steps)
        for n in range(1, 9):
            got += ink_steps(n) if name == "ink_steps" else seq_blue(n)
        worst = min(got, key=contrast)
        assert contrast(worst) >= floor, (
            f"{name} step {worst} is {contrast(worst):.2f}:1 on white, under the "
            f"{floor}:1 floor -- it will not survive printing as a 1 pt line "
            f"(see CONTRAST_MIN; INK3 is the reserved reference ink, not a step)")
    got = list(MODEL_HUES)
    for base in MODEL_HUES:
        for n in range(1, 7):
            got += seq_from(base, n)
    worst = min(got, key=contrast)
    assert contrast(worst) >= floor, (
        f"model hue or its ramp step {worst} is {contrast(worst):.2f}:1 on "
        f"white, under the {floor}:1 floor")
    # One hue per GROUP, not per model: colour says whether the model reasons,
    # and identity inside a group is the marker and the dash.
    assert set(MODEL_COLOR) == set(MODELS), "one colour per model"
    assert set(MODEL_MARKER) == set(MODELS), "one marker per model"
    assert len({MODEL_MARKER[m] for m in MODELS}) == len(MODELS), "markers repeat"
    assert set(MODEL_HUES) == set(MODEL_COLOR.values()), "MODEL_HUES is the hues in use"


# ------------------------------------------------------------------ rcParams
def apply() -> None:
    _check_ramps()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        # The paper sets math in Computer Modern: the style file loads
        # `times`, which changes the roman text font and leaves math alone
        # (pdffonts on the built PDF: NimbusRomNo9L for text, CMMI/CMR/CMSY for
        # math). "cm" is matplotlib's copy of those faces, so a symbol in a
        # panel and the same symbol in the caption are the same glyph.
        "mathtext.fontset": "cm",
        "font.size": FS_NOTE,
        "axes.titlesize": FS_AXIS,
        "axes.titleweight": "normal",
        "axes.titlecolor": INK,
        "axes.labelsize": FS_AXIS,
        "axes.labelcolor": INK,
        "xtick.labelsize": FS_NOTE,
        "ytick.labelsize": FS_NOTE,
        "legend.fontsize": FS_NOTE,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.8,
        "patch.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


# --------------------------------------------------------------------- ticks
def fmt_tick(value: float) -> str:
    """``0``, ``0.5``, ``1`` -- the spec's one tick spelling. Never ``.5``."""
    return f"{float(value):g}"


def ticks01(ax, axis: str = "y", values=(0, 0.5, 1)) -> None:
    """Put ``values`` on ``axis`` ("x", "y" or "both") and label them the one way.

    Three ticks an axis at most -- the two ends and one middle value -- written
    ``0`` / ``0.5`` / ``1``, so no panel ever prints ``.5`` or ``0.50``.
    """
    labels = [fmt_tick(v) for v in values]
    for which in (("x", "y") if axis == "both" else (axis,)):
        if which == "x":
            ax.set_xticks(list(values))
            ax.set_xticklabels(labels)
        elif which == "y":
            ax.set_yticks(list(values))
            ax.set_yticklabels(labels)
        else:
            raise ValueError(f"axis must be 'x', 'y' or 'both', not {axis!r}")


# --------------------------------------------------------------- uncertainty
# Spec section 3: every estimate shows its interval, drawn the same way every
# time. "The interval is small" is a reason to draw it, not to omit it.
BAND_ALPHA = 0.20
WHISKER_LW = 0.7


def band(ax, x, lo, hi, color, alpha: float = BAND_ALPHA, zorder: float = 1.2, **kw):
    """A filled uncertainty envelope along a continuous ``x``: no edge, alpha 0.20.

    Alpha 0.20, not 0.15: on the lightest peer ink the band has to stay a band
    once the ramp is floored at ``CONTRAST_MIN``, and three of them still read
    as three where they overlap.

    ``x``, ``lo``, ``hi`` are array-likes (or scalars) of equal length. Where two
    envelopes overlap, draw the emphasised one last. A single point has no
    envelope to fill, so it falls through to :func:`whisker` -- the interval is
    drawn either way.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lo = np.broadcast_to(np.atleast_1d(np.asarray(lo, dtype=float)), x.shape)
    hi = np.broadcast_to(np.atleast_1d(np.asarray(hi, dtype=float)), x.shape)
    if x.size < 2:
        return whisker(ax, x, lo, hi, color, **kw)
    return ax.fill_between(x, lo, hi, facecolor=color, alpha=alpha, edgecolor="none",
                           linewidth=0.0, zorder=zorder, **kw)


def whisker(ax, x, lo, hi, color, lw: float = WHISKER_LW, zorder: float = 1.2,
            orient: str = "v", **kw):
    """A capless interval at a discrete ``x``, drawn behind the marker.

    ``x``, ``lo``, ``hi`` are array-likes or scalars (one point is fine). Markers
    default to ``zorder`` 2, so the default ``zorder`` 1.2 puts the interval
    under them; the marker, not the whisker, carries the estimate. ``orient``
    ``"h"`` turns it on its side for a forest plot, where ``x`` is the row and
    ``lo``/``hi`` are the interval along the value axis.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lo = np.broadcast_to(np.atleast_1d(np.asarray(lo, dtype=float)), x.shape)
    hi = np.broadcast_to(np.atleast_1d(np.asarray(hi, dtype=float)), x.shape)
    if orient.lower() not in ("h", "v", "horizontal", "vertical"):
        raise ValueError(f"orient must be horizontal or vertical, not {orient!r}")
    draw = ax.hlines if orient.lower() in ("h", "horizontal") else ax.vlines
    coll = draw(x, lo, hi, colors=color, linewidths=lw, zorder=zorder, **kw)
    coll.set_capstyle("butt")
    return coll


# ------------------------------------------------------------------ emphasis
# Spec section 2: contrast is a budget. Emphasising everything emphasises
# nothing, so exactly one series per claim wears EMPH and the rest wear CTX.
EMPH = dict(lw=1.5, color=INK, solid_capstyle="round", zorder=3)
CTX = dict(lw=0.8, color=INK3, solid_capstyle="round", zorder=2)
MARKER_EMPH = dict(ms=3.0, markeredgewidth=0.6)          # filled: face = colour
MARKER_CTX = dict(ms=3.0, markerfacecolor=SURFACE, markeredgewidth=0.6)


def emphasis(artist=None, on: bool = True, **kw) -> dict:
    """The line style for an emphasised (1.5 pt, dark) or a context (0.8 pt, ink 3) series.

    Three ways to call it, all returning the style dict::

        ax.plot(x, y, **fs.emphasis(True, color=fs.BLUE))   # style a new line
        style = fs.emphasis(on=False)                       # just the dict
        fs.emphasis(line, on=False)                         # restyle in place

    Passing a Line2D (or a list of them) as ``artist`` applies the style to it as
    well as returning it. ``kw`` overrides any key, e.g. ``color``.
    """
    if isinstance(artist, bool):                 # emphasis(False) reads naturally
        artist, on = None, artist
    style = dict(EMPH if on else CTX)
    style.update(kw)
    if artist is not None:
        for line in (artist if isinstance(artist, (list, tuple)) else [artist]):
            line.set(**style)
    return style


def marker_style(on: bool, color: str = INK, **kw) -> dict:
    """Filled markers on the emphasised series, open (white face) on a context one."""
    style = dict(MARKER_EMPH if on else MARKER_CTX)
    style["color"] = color
    if not on:
        style["markeredgecolor"] = color
    style.update(kw)
    return style


def reference(ax, value, orient: str = "h", lo=None, hi=None, color: str = INK3,
              lw: float = 0.6, dashes=(3, 2), zorder: float = 0.5, **kw):
    """An ink-3, 0.6 pt dashed reference line -- never labelled in the panel.

    Ink 3 is the default and the rule for anything shared by the whole panel
    (the round-0 vote, the independent-majority contour, ``p = 1/2``). A rule
    that belongs to ONE series -- each model's own ``c*`` in F2(b) -- takes that
    series' colour instead, because three ink-3 rules in one row say nothing
    about which model each belongs to; give it a dash pattern of its own so it
    is not read as a second line of that series. ``orient`` is ``"h"``/``"y"``
    for a horizontal rule or
    ``"v"``/``"x"`` for a vertical one. ``value`` may be a scalar or a sequence.
    With ``lo``/``hi`` given the rule is short, spanning ``[lo, hi]`` in DATA
    coordinates of the other axis; without them it spans the panel. What the
    reference means goes in the caption.
    """
    horizontal = orient.lower() in ("h", "y", "horizontal", "axhline")
    if orient.lower() not in ("h", "y", "horizontal", "axhline",
                              "v", "x", "vertical", "axvline"):
        raise ValueError(f"orient must be horizontal or vertical, not {orient!r}")
    values = np.atleast_1d(np.asarray(value, dtype=float))
    style = dict(color=color, lw=lw, zorder=zorder, **kw)
    if lo is None or hi is None:
        out = []
        for v in values:
            line = (ax.axhline if horizontal else ax.axvline)(float(v), **style)
            line.set_dashes(list(dashes))
            out.append(line)
        return out[0] if len(out) == 1 else out
    coll = (ax.hlines if horizontal else ax.vlines)(
        values, lo, hi, colors=color, linewidths=lw, zorder=zorder,
        linestyles=(0, tuple(dashes)), **kw)
    return coll


# ------------------------------------------------------------------- caption
FS_CAP = 7.5              # deprecated: the legend band at the foot of a figure


def write_caption(name: str, lines: list[str]) -> Path:
    """Write the caption body to ``figs/captions/<name>.tex`` for ``\\input``.

    ``lines`` are the sentences of one caption paragraph; they are joined with
    single spaces, so a script may wrap them however it likes. The file holds
    the body only -- ``main.tex`` supplies ``\\caption{}`` and the label.

    Everything the panel may not say lives here: what is plotted, on what data,
    how many runs, what the interval is, what the colours mean, and what the
    reader should see. Long is fine.
    """
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    body = " ".join(line.strip() for line in lines if line and line.strip())
    out = CAPTION_DIR / f"{name}.tex"
    out.write_text(body + "\n", encoding="utf-8")
    return out


def caption_band(fig, lines, y_top: float, x0: float = 0.03, lh: float = 0.128):
    """Deprecated for publication output: use :func:`write_caption` instead.

    The band was written for drafts in which an image travels with no caption
    of its own. The paper carries captions in LaTeX, and the band costs
    1.0-1.6 in of every figure to print prose at 7.5 pt, so no figure in
    ``figs/final`` may keep it. It stays here only so the scripts that have not
    been rewritten yet still import.
    """
    W, H = fig.get_size_inches()
    fig.add_artist(mpl.lines.Line2D(
        [x0 / W, 1 - 0.03 / W], [y_top / H] * 2, color=AXIS, lw=0.6,
        transform=fig.transFigure, figure=fig))
    y = y_top - 0.155
    for line in lines:
        fig.text(x0 / W, y / H, line, ha="left", va="top", fontsize=FS_CAP,
                 color=INK2, linespacing=1.15)
        y -= lh
    return y


def figure(height: float, width: float = WIDTH_IN, **kw):
    return plt.subplots(figsize=(width, height), **kw)


# -------------------------------------------------------------------- density
DENSITY_MIN = 0.55        # spec section 6: panels take at least 55 % of the figure


def density(fig) -> float:
    """Union area of the figure's axes boxes, as a fraction of the figure area.

    Overlapping boxes count once. A hand-drawn schematic on one big axes counts
    as its axes box, exactly like a data panel. The spec's floor is
    ``DENSITY_MIN``; ``save`` prints the achieved value on every render so it can
    be checked without re-measuring by hand.
    """
    boxes = [ax.get_position() for ax in fig.axes if ax.get_visible()]
    rects = [(b.x0, b.y0, b.x1, b.y1) for b in boxes if b.width > 0 and b.height > 0]
    if not rects:
        return 0.0
    xs = sorted({v for r in rects for v in (r[0], r[2])})
    ys = sorted({v for r in rects for v in (r[1], r[3])})
    area = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        xm = 0.5 * (x0 + x1)
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            ym = 0.5 * (y0 + y1)
            if any(rx0 <= xm <= rx1 and ry0 <= ym <= ry1 for rx0, ry0, rx1, ry1 in rects):
                area += (x1 - x0) * (y1 - y0)
    return area  # axes positions are figure fractions, so the figure area is 1


def save(fig, name: str) -> list[Path]:
    """Write ``figs/final/<name>.png`` (400 dpi) and ``<name>.pdf`` (vector).

    The canvas is written at the size the script set: figure heights are fixed
    by the spec, so nothing here may grow the box past the layout. The panel
    density is printed beside the paths, flagged when it is under the spec's
    floor.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{name}.{ext}"
        # No creation date in the PDF, so two renders of the same data are
        # byte-comparable.
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(out, dpi=400, metadata=meta)
        outs.append(out)
    frac = density(fig)
    flag = "" if frac >= DENSITY_MIN else f"  <- under the {DENSITY_MIN:.0%} floor"
    print(f"{name}: panels {frac:.1%} of the figure{flag}  "
          f"[{', '.join(str(p) for p in outs)}]")
    return outs


def panel_label(ax, letter: str, x: float = -0.14, y: float = 1.0) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_AXIS, fontweight="bold",
            ha="left", va="top", color=INK)


MODEL_KEY_H = 0.20       # inches the one-row model key takes over a figure


def model_key_strip(fig, models, y_in: float, extra=(), sep: float = 0.12) -> None:
    """Name every model's shape once, in one centred row at height ``y_in``.

    F2 and F3 carry the model on the marker shape in their scatter panels,
    where no row names it (9/24: the reader had no key for the shapes).
    ``extra`` appends (label, marker style dict) entries after a wider gap,
    for a figure whose instruction fill is not keyed in any panel.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches()
    sample, gap, sep_extra = 0.07, 0.03, 0.26                # inches
    # the star is drawn half again as large in the panels (solid_thin_marks)
    entries = [(MODEL_SHORT[m], dict(marker=MODEL_MARKER[m], mew=0.6, mfc=INK,
                                     mec=INK,
                                     ms=3.6 * (1.6 if MODEL_MARKER[m] == "*" else 1)))
               for m in models]
    entries += list(extra)
    widths = []
    for name, _ in entries:
        t = fig.text(0, 0, name, fontsize=FS_NOTE)
        widths.append(t.get_window_extent(r).width / fig.dpi)
        t.remove()
    seps = [sep_extra if i == len(models) else sep for i in range(1, len(entries))]
    total = sum(sample + gap + w for w in widths) + sum(seps)
    x = (fw - total) / 2
    for i, ((name, style), w) in enumerate(zip(entries, widths)):
        fig.add_artist(mpl.lines.Line2D([(x + sample / 2) / fw], [y_in / fh],
                                        transform=fig.transFigure, ls="none",
                                        **style))
        fig.text((x + sample + gap) / fw, y_in / fh, name, fontsize=FS_NOTE,
                 color=INK2, ha="left", va="center")
        x += sample + gap + w + (seps[i] if i < len(seps) else 0)


def direct_label(ax, x, y, text: str, dx: float = 4, dy: float = 0, ha: str = "left",
                 va: str = "center", color: str = INK2, fontsize: float = FS_NOTE) -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                color=color, fontsize=fontsize, annotation_clip=False)


def spread(values, min_gap: float, lo: float, hi: float) -> list[float]:
    """Push a sorted list of label positions apart to at least ``min_gap``.

    Returns new positions in the same order as ``values`` (which need not be
    sorted), kept inside ``[lo, hi]``.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < min_gap:
            pos[i] = pos[i - 1] + min_gap
    overflow = pos[-1] - hi
    if overflow > 0:
        pos = [p - overflow for p in pos]
    for i in range(len(pos) - 2, -1, -1):
        if pos[i + 1] - pos[i] < min_gap:
            pos[i] = pos[i + 1] - min_gap
    if pos[0] < lo:
        shift = lo - pos[0]
        pos = [p + shift for p in pos]
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


# --------------------------------------------------------------- paper data
# Paths and the model table come from the figure context, a JSON file the
# workflow writes from config/config.yaml (rule `figure_context`) and points to
# with the FIG_CONTEXT environment variable. Nothing about models -- which ones,
# their order, markers, run directories -- is written in this file.
import json as _json
import os as _os

_CTX_PATH = Path(_os.environ.get("FIG_CONTEXT", "results/figures/context.json"))
if not _CTX_PATH.exists():
    raise SystemExit(f"figure context {_CTX_PATH} not found; run the workflow "
                     "(snakemake figures) or set FIG_CONTEXT")
FIGCTX = _json.loads(_CTX_PATH.read_text())

OUT_DIR = Path(FIGCTX["out_dir"])
CAPTION_DIR = Path(FIGCTX["caption_dir"])
EST = Path(FIGCTX["estimates_dir"])          # released or recomputed estimates
TRANSCRIPTS = Path(FIGCTX["transcripts_dir"])
ITEMS_DIR = Path(FIGCTX["items_dir"])
SWEEP = Path(FIGCTX["sweep"])
DECOMP_DIR = Path(FIGCTX["decomposition_dir"])
THEORY_DIR = Path(FIGCTX["theory_dir"])      # abm_phase.csv, meanfield_boundary.csv
MODEL_COMPARISON = Path(FIGCTX["model_comparison"])

# The main-text models, in display order (non-reasoning first). The keys are
# the ids the transcripts carry, because the csv files a figure joins on
# (decomposition_fitted.csv, sweep.csv) name a model that way. Every model is
# drawn at one reasoning setting, the one its main arm was run at; a second arm
# of the same model id is a treatment, not a replicate, and is excluded wherever
# the sweep is read by run (ABLATION_SUFFIX).
_M = FIGCTX["models"]
MODELS = [m["name"] for m in _M]
REASONS = {m["name"]: bool(m["reasoning"]) for m in _M}
MODEL_ID = {m["name"]: m["api_id"] for m in _M}
MODEL_HP_RUN = {m["name"]: m["hp_run"] for m in _M}
MODEL_DIR = {m["name"]: EST / "runs" / m["hp_run"] for m in _M}
DATA = EST / "runs"   # run-level estimates: DATA / <run> / cbrm_params.csv
MODEL_EVENTS_DIR = {m["name"]: TRANSCRIPTS / m["hp_run"] for m in _M}
MODEL_MARKER = {m["name"]: m["marker"] for m in _M}
MODEL_DASH = {m["name"]: tuple(m["dash"]) if m.get("dash") else (None, None) for m in _M}
MODEL_SHORT = {m["name"]: m["short"] for m in _M}
MODEL_ABLATION = {m["name"]: m["ablation"] for m in _M if m.get("ablation")}
# F2 is greyscale: a mark's shape is its model and its fill the instruction arm.
REASON_COLOR = {True: INK, False: INK}
MODEL_COLOR = {m: INK for m in MODELS}
MODEL_HUES = [INK]
MODEL_DISPLAY = list(MODELS)

# The paper's three instructions: A (answer honestly), B (no instruction), C
# (the strongest cohesion instruction). The data files carry the raw keys the
# runs were launched under; `arms_only` keeps those rows and renames them on
# read, so nothing downstream sees a raw key.
ARM_NAME = dict(FIGCTX["arm_names"])
RAW_ARMS = list(FIGCTX["arms"])                  # raw keys, for event-level readers
ARMS = [ARM_NAME[k] for k in RAW_ARMS]           # names as printed
CONDITIONS = list(FIGCTX["conditions"])


def arms_only(d, col: str = "condition"):
    """Keep the rows of the paper's instructions and name them as printed."""
    d = d[d[col].isin(ARM_NAME)].copy()
    d[col] = d[col].map(ARM_NAME)
    return d

PROBE = FIGCTX["probe"]
ABLATION_SUFFIX = tuple(FIGCTX["ablation_suffix"])
C_STAR_STD = float(FIGCTX["c_star_std"])  # standard setting gamma = 0.6, a = 0.5
BENCHMARKS = FIGCTX["benchmarks"]
F3_PANELS = list(FIGCTX["f3_panels"])
F3_ARMS = [ARM_NAME[k] for k in FIGCTX["f3_arms"]]   # the two arms F3 draws, as printed
