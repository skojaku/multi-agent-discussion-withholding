"""F3: where the gain goes, and what moves the rate it goes with.

Three panels in one row (spec 2026-09-08 rev. 2, 5.5 x 2.05 in):

  (a) what discussion bought -- the collective accuracy after discussion minus
      the round-0 vote of the same agents on the same items -- against the
      distance from the threshold, c-hat minus that setting's own c*, one mark
      per setting over every setting the meter sweep measures;
  (b) withholding measured per (model, benchmark);
  (c) the model and instruction effects, on the logit scale.

9/14, fourth pass: the figure is one row of three and every panel now crosses
benchmarks.  The two hidden profile panels that were (a) and (b) -- collective
accuracy against c-hat on one model and one task, and the disclosure of decisive
evidence -- are Appendix Figure C1, which draws them beside the accuracy split
they end in.  In their place is what was the standalone F5, re-drawn against the
margin instead of against P(c-hat > c*): the claim of 4.2 is that the DISTANCE
from the threshold orders the gain, and the margin is that distance, on the same
axis the reader has already met in F2(b).  F5 itself is gone; this is it.

Every number is read from the csv files at run time; every sentence is in
``figs/captions/F3.tex``.

Run:
  uv run snakemake --rerun-triggers mtime -c1 figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

DECOMP = fs.DECOMP_DIR
SWEEP = fs.SWEEP
# sweep.csv's own model column carries the API id ("openai/gpt-4o-mini"), not
# the short key fs.MODEL_MARKER etc. are keyed on ("gpt-4o-mini") -- the
# inverse of fs.MODEL_ID.
ID_TO_MODEL = {v: k for k, v in fs.MODEL_ID.items()}
# Shape is the model (as in F2(e)); fill is the instruction, the same two
# instructions (a)-(c) already filter to. Mirrors fig_F2.py's ARM_FACE/
# mark_style, just for the two-arm A/C slice this figure draws.
# 9/19: A is blue wherever an instruction is a mark -- F2(c-f) and here -- and
# B and C stay ink, filled at B and open at C. Two panels that spelled the
# same arm two ways ("filled means A" here against "filled means B" in F2)
# would have cost the reader more than the colour does.
INSTR_FACE = {"A": fs.BLUE, "C": fs.SURFACE}
# 9/24: A carries a white edge and is a touch larger, as in F2(c).
INSTR_EDGE = {"A": fs.SURFACE, "C": fs.INK}
INSTR_MS = {"A": 4.0, "C": 3.2}
# Six families share the rows of (b), so each name is set in the panel's own
# name column rather than on a tick; the full names are in the caption.
FAMILY_SHORT = {k: b["short"] for k, b in fs.BENCHMARKS.items()}
FAMILY_LONG = {k: b["long"] for k, b in fs.BENCHMARKS.items()}

TICK_PAD = 0.5          # points between a tick mark and its label; see finish()

# The models this figure keys: those of fs.MODELS with at least one cell in
# the sweep slice (a-c) read or in the decomposition (b) and (d) read. Set in
# main() once the data are loaded. qwen3.8 joined fs.MODELS on 9/16 with its
# hidden-profile run only, and a key that names a model no panel draws would
# send the reader looking for a mark that is not there.
DRAWN_MODELS: list[str] = list(fs.MODELS)

NOTES: list[str] = []


def note(msg: str) -> None:
    NOTES.append(msg)


def nice_grid_bounds(lo_data: float, hi_data: float,
                      max_pad_ratio: float = 0.2) -> tuple[float, float, float]:
    """The coarsest 1/2/5 x 10^n grid step whose floor/ceil bounds don't pad
    the data range by more than ``max_pad_ratio``.

    A fixed grid (e.g. always round to 0.1) reads fine on a panel whose data
    happens to span several of that grid's steps, but on one that doesn't --
    MuSiQue's gain tops out at 0.067, so rounding to 0.1 alone puts the data
    in the bottom two-thirds of the axis, a 71 % pad -- it wastes most of the
    panel on white space. Tried coarse to fine so the answer stays as round
    as the actual spread of the data allows.
    """
    data_range = hi_data - lo_data
    if data_range <= 0:
        data_range = abs(hi_data) if hi_data != 0 else 1.0
    for exponent in range(3, -6, -1):
        for base in (5, 2, 1):
            step = base * (10.0 ** exponent)
            lo = np.floor(lo_data / step) * step
            hi = np.ceil(hi_data / step) * step
            if (hi - lo) - data_range <= max_pad_ratio * data_range:
                return round(lo, 10), round(hi, 10), step
    return lo_data, hi_data, data_range


def safe_mid(lo: float, hi: float, step: float) -> float:
    """The cleanest-looking midpoint tick between ``lo`` and ``hi``.

    Tries the true arithmetic midpoint snapped at ``step``'s own resolution
    and at two finer ones, keeps whichever candidates don't collide with
    ``lo``/``hi`` (a risk at the coarsest resolution: 0.2 and 0.8 a step of
    0.2 apart round their exact midpoint, 0.5, to 0.4 under banker's
    rounding), and returns the one that prints shortest. That means an exact
    midpoint that already lands on the grid (0.5) wins outright, and one that
    doesn't (0.035 between 0 and 0.07) yields to a nearby value that reads
    cleaner (0.04) instead of being forced onto the coarse grid (0.4/5 -> a
    value indistinguishable from the axis ends) or printed at full precision.
    """
    raw_mid = (lo + hi) / 2
    candidates = []
    for resolution in (step / 4, step / 2, step):
        grid = 1.0 / resolution
        mid = round(raw_mid * grid) / grid
        if lo < mid < hi:
            candidates.append(mid)
    return min(candidates, key=lambda m: len(f"{m:g}")) if candidates else raw_mid


# -------------------------------------------------------------------- panel a
# The sweep names a setting's family after the top folder of its run, so the
# seven hidden profile folders -- the topology sweep, N = 25, the repeats, the
# interventions, the model sweep, gemma -- are seven families and one benchmark,
# and the Silo pilot is Silo. The panel does not colour by benchmark, but the
# caption says how many there are, and counting folders would let it claim
# fifteen benchmarks where the reader would find seven.
# Silo-Bench and AssetOpsBench were dropped on 2026-09-14, on cost: they are the
# most expensive families per measured cell and the models added that day were
# never run on them, so a panel that kept them would report four models on two
# families and two models on two more. Their transcripts stay on disk; the keys
# stay here, commented, because the rows exist and are simply not read. The same
# four families are what the decomposition (b) and (c) draw is fit on, so the
# three panels now cover one set of benchmarks rather than two.
# Run-directory prefix -> the benchmark name the panels are titled by.
BENCH = {p: b["figure_name"] for b in fs.BENCHMARKS.values()
         for p in b["figure_prefixes"]}
# The elicitation every cell in this figure has to share. Section 3.1 asks for
# the private answer in a call of its own; a cell measured the other way puts a
# different quantity on the same axis (a moves by about 0.5 between the two), and
# the runs made before the flag existed record no probe at all.
PROBE = fs.PROBE
# A run whose folder ends in one of these is a reasoning ablation: the same
# model id run at a different thinking setting. What that setting IS varies and
# the folder name does not say it -- events.meta.json does: `off` for luna and
# qwen, `low` for gemini (whose main arm is the provider default, so its pair is
# default against low effort, not on against off), `high` for glm-5.3-flash
# against a main arm at `low`. Either way it is a treatment and not a replicate,
# so a panel that marks the model by shape would otherwise draw it as extra
# cells of that model -- two marks of the same shape at the same arm, one of
# them a setting the text never mentions.
# The run name is the only key available here: sweep.csv carries no reasoning or
# effort column, so a cell cannot be matched back to its setting from the sweep
# alone. decomposition.py cuts the same runs the same way.
ABLATION_SUFFIX = fs.ABLATION_SUFFIX

# Bin edges on the margin, with zero as an edge: the claim is about the sign, so
# no bin may average a team that can still recover together with one that
# cannot. Two bins below the threshold and three above, which is where the 177
# settings are -- being past it is the ordinary case in what we ran.
BINS = (-1.001, -0.15, 0.0, 0.2, 0.5, 1.001)
N_BOOT = 5000
SEED = 20260914


def load_settings(off: bool = False) -> pd.DataFrame:
    """Every sweep cell whose threshold is identified and whose gain exists.

    The cut the sweep script itself reports on, over the benchmarks BENCH maps,
    the one elicitation PROBE names and the models the figure has a mark for.
    Nothing else is dropped -- no instruction, no topology -- because "every
    setting" is what the panel claims.

    The model cut is not cosmetic. A cell the panel cannot draw would still
    enter the regression line, the bin means and the caption's correlation, so
    the reader would be shown a line fitted to points that are not there (9/19:
    a new model (gemini-3.8-flash) entered in the sweep, and nothing in the paper keys
    it -- four undrawn cells per benchmark flattened MuSiQue's fit). Drawing it
    instead is an editorial decision about which models the paper reports, and
    belongs in fs.MODELS, not here.
    """
    d = fs.arms_only(pd.read_csv(SWEEP))
    ok = d[d["measurable"] & d["margin"].notna()
           & d["interaction_gain"].notna()].copy()
    if ok.empty:
        raise SystemExit(f"{SWEEP} has no measurable cells")
    probe = ok["probe"].fillna("unknown").astype(str)
    dropped_probe = int((probe != PROBE).sum())
    ok = ok[probe == PROBE].copy()
    out = sorted(set(ok["family"]) - set(BENCH))
    dropped_fam = int(ok["family"].isin(out).sum())
    ok = ok[~ok["family"].isin(out)].copy()
    if ok.empty:
        raise SystemExit(f"{SWEEP} has no {PROBE}-probe cells in {sorted(set(BENCH.values()))}")
    unkeyed = sorted(set(ok["model"]) - set(ID_TO_MODEL))
    dropped_model = int(ok["model"].isin(unkeyed).sum())
    ok = ok[~ok["model"].isin(unkeyed)].copy()
    leaf = ok["run"].str.rsplit("/", n=1).str[-1]
    if off:
        # 9/24: the reasoning-off arms alone, drawn striped and left out of
        # the fit, as F2(d) draws them
        ok = ok[leaf.str.endswith("_nothink")].copy()
        ok["benchmark"] = ok["family"].map(lambda f: BENCH.get(f, f))
        return ok
    abl = leaf.str.endswith(ABLATION_SUFFIX)
    dropped_abl = int(abl.sum())
    abl_runs = sorted(set(ok.loc[abl, "run"]))
    ok = ok[~abl].copy()
    note("(a) %d cells dropped: %d not probed %s, %d in families this figure does"
         " not read (%s), %d run by models the figure does not draw (%s),"
         " %d reasoning ablations (%s)"
         % (dropped_probe + dropped_fam + dropped_model + dropped_abl,
            dropped_probe, PROBE, dropped_fam, ", ".join(out) or "none",
            dropped_model, ", ".join(unkeyed) or "none",
            dropped_abl, ", ".join(abl_runs) or "none"))
    # A cell whose model solved no item before the discussion and none after
    # it has a gain of zero by construction: there was nothing to gain, so the
    # cell says nothing about whether the threshold orders the gain. Dropped
    # here rather than drawn at y = 0 (9/16: mixtral on MuSiQue, three cells).
    floor = ((ok["vote_accuracy"].fillna(0) == 0)
             & (ok["final_accuracy"].fillna(0) == 0))
    if floor.any():
        note("(a-c) %d floor cells dropped (accuracy 0 before and after): %s"
             % (int(floor.sum()),
                ", ".join(sorted(ok.loc[floor, "run"] + "/" + ok.loc[floor, "condition"]))))
        ok = ok[~floor].copy()
    ok["benchmark"] = ok["family"].map(lambda f: BENCH.get(f, f))
    return ok


# 9/24: the fit that admits the reasoning-off arms, pooled into their base
# model (decomposition.py --reasoning). Split into ``<model>@off`` levels the
# two arms sat on top of each other, so (d) draws the pool; the split drawing
# below stays for a fit that carries @off levels. Later on 9/24 the fit moved
# to one stage, on the per-task counts (decomposition.py --one-stage); (a)-(c)
# keep the per-setting rates.
FIT_TAG = "_crossed_onestage"
OFF_TAG = "@off"


def load_effects() -> pd.DataFrame:
    """The crossed fit's effects, with the instruction levels named A/B/C.

    B is the fit's baseline and has no coefficient, so the instruction rows are
    A and C; any other instruction level the file carries is dropped.
    """
    eff = pd.read_csv(DECOMP / f"decomposition_effects{FIT_TAG}.csv")
    kap = eff.term == "kappa"
    eff = pd.concat([eff[~kap], fs.arms_only(eff[kap], "level")])
    return eff[~((eff.term == "kappa") & (eff.level == "B"))].copy()


def boot_mean_ci(y: np.ndarray, rng) -> tuple[float, float, float]:
    """Mean of ``y`` with a 95 % percentile bootstrap interval around it."""
    draws = rng.choice(y, size=(N_BOOT, len(y)), replace=True).mean(axis=1)
    return (float(y.mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def binned(d: pd.DataFrame) -> pd.DataFrame:
    """The mean gain of each bin of the margin, drawn at the bin's median margin."""
    rng = np.random.default_rng(SEED)
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        g = d[(d.margin > lo) & (d.margin <= hi)]
        if g.empty:
            continue
        mean, cl, ch = boot_mean_ci(g.interaction_gain.to_numpy(), rng)
        rows.append(dict(lo=lo, hi=hi, n=len(g), x=float(g.margin.median()),
                         gain=mean, lo_ci=cl, hi_ci=ch))
    return pd.DataFrame(rows)


def model_shape_legend(ax, models, x_frac: float, y_frac: float,
                        row_h: float = 0.11) -> None:
    """Model shapes only, stacked from ``(x_frac, y_frac)`` in axes fractions.

    Drawn once, in (a) -- the first panel read -- rather than repeated in
    each: the shape's meaning (which model) doesn't change panel to panel, so
    it only needs saying once, and not every model this key names actually
    has a point in (a) (MedEInst is the one benchmark with all four). The
    instruction fill is explained in the caption instead of a second in-panel
    key, since (d) already names A/C as text.
    """
    y = y_frac
    for m in models:
        ax.plot([x_frac], [y], transform=ax.transAxes, marker=fs.MODEL_MARKER[m],
                ms=3.2, mew=0.6, mfc=fs.INK, mec=fs.INK, ls="none", zorder=6,
                clip_on=False)
        ax.text(x_frac + 0.09, y, fs.MODEL_SHORT[m], transform=ax.transAxes,
                fontsize=fs.FS_SMALL, color=fs.INK2, ha="left", va="center",
                zorder=6)
        y -= row_h


# (a)-(c): degree of the drawn fit. Linear (9/24): unlike F2(d), a quadratic
# does not beat the line on any benchmark (AIC and leave-one-out MSE equal or
# worse), and on MuSiQue it turns up at the right end on a few settings.
FIT_DEG = 1
BAND_BOOT = 2000         # bootstrap draws over settings for the fit's band


def fit_compare(x: np.ndarray, y: np.ndarray) -> dict:
    """AIC and leave-one-out MSE of the linear and the quadratic fit."""
    out = {}
    n = x.size
    for deg in (1, 2):
        rss = float(((y - np.polyval(np.polyfit(x, y, deg), x)) ** 2).sum())
        aic = n * np.log(rss / n) + 2 * (deg + 1)
        loo = np.mean([(y[i] - np.polyval(np.polyfit(np.delete(x, i),
                                                     np.delete(y, i), deg), x[i])) ** 2
                       for i in range(n)])
        out[deg] = (float(aic), float(loo))
    return out


def panel_gain(ax, d: pd.DataFrame, off: pd.DataFrame, benchmark: str,
               title: str = None) -> dict:
    """The gain against the distance from the threshold, drawn as F2(d) but
    with a linear fit.

    9/24: no whisker per setting and no bin means -- the marks are the point
    estimates, shape the model and fill the instruction (fig_F2.mark_style),
    and the uncertainty is the fit's: a line with a 95 % band from 2000
    bootstrap resamples of the settings, as seaborn's regplot draws it. The
    reasoning-off settings are striped (fig_F2.off_mark) and, since later on
    9/24, enter the fit with the reasoning-on ones. The band does not set the
    y range; the points do.
    """
    import fig_F2 as f2
    d = d[d.benchmark == benchmark].copy()
    o = off[off.benchmark == benchmark].copy()
    for t in (d, o):
        t["model_key"] = t["model"].map(ID_TO_MODEL)
    fs.reference(ax, 0.0, "h")
    fs.reference(ax, 0.0, "v")
    for r in d.itertuples():
        ax.plot([r.margin], [r.interaction_gain], marker=fs.MODEL_MARKER[r.model_key],
                ls="none", zorder=3.2, **f2.mark_style(r.condition))
    for r in o.itertuples():
        f2.off_mark(ax, r.margin, r.interaction_gain,
                    fs.MODEL_MARKER[r.model_key], r.condition)

    # every drawn setting, reasoning on and off (9/24)
    x = np.r_[d.margin.to_numpy(float), o.margin.to_numpy(float)]
    y = np.r_[d.interaction_gain.to_numpy(float),
              o.interaction_gain.to_numpy(float)]
    coef = np.polyfit(x, y, FIT_DEG)
    xx = np.linspace(x.min(), x.max(), 100)
    rng = np.random.default_rng(SEED)
    boot = np.empty((BAND_BOOT, xx.size))
    for k in range(BAND_BOOT):
        i = rng.integers(0, x.size, x.size)
        boot[k] = np.polyval(np.polyfit(x[i], y[i], FIT_DEG), xx)
    band_lo, band_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ax.fill_between(xx, band_lo, band_hi, color=fs.VERM, alpha=0.18, lw=0,
                    zorder=2.0)
    # the line over every mark, so no setting hides it (9/24)
    ax.plot(xx, np.polyval(coef, xx), "-", color=fs.VERM, lw=1.2,
            solid_capstyle="round", zorder=4.0)
    cmp_ = fit_compare(x, y)
    note("(a-c) %s: %d settings fitted (%d of them reasoning off); linear AIC"
         " %.1f LOO-MSE %.5f, quadratic AIC %.1f LOO-MSE %.5f"
         % (benchmark, x.size, len(o), cmp_[1][0], cmp_[1][1], cmp_[2][0], cmp_[2][1]))

    # ranges from the points, on and off; the band may run off (as F2(d))
    xa, ya = x, y
    x_tick_lo = float(np.floor(xa.min() * 20) / 20)
    x_tick_hi = float(np.ceil(xa.max() * 20) / 20)
    x_lo, x_hi = x_tick_lo - 0.03, x_tick_hi + 0.03
    y_tick_lo, y_tick_hi, y_step = nice_grid_bounds(float(ya.min()), float(ya.max()))
    y_lo, y_hi = y_tick_lo - 0.06 * y_step, y_tick_hi + 0.06 * y_step
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.spines["bottom"].set_bounds(x_lo, x_hi)
    ax.spines["left"].set_bounds(y_lo, y_hi)
    fs.ticks01(ax, "x", (x_tick_lo, safe_mid(x_tick_lo, x_tick_hi, 0.05), x_tick_hi))
    fs.ticks01(ax, "y", (y_tick_lo, safe_mid(y_tick_lo, y_tick_hi, y_step), y_tick_hi))
    ax.set_xlabel("$\\hat{c} - c^{*}$", labelpad=0)
    # every panel names its own y: the three do not share a range
    # one line: only (b) carries it, and it may run past (b) into the gaps
    ax.set_ylabel("gain from discussion", labelpad=1)
    if title:
        # inside, top right, which no benchmark's settings reach (9/24)
        ax.text(0.97, 0.97, title, transform=ax.transAxes, ha="right",
                va="top", fontsize=fs.FS_AXIS, color=fs.INK, zorder=6)
    finish(ax)
    return dict(n=x.size, n_off=len(o), cmp=cmp_)


def arm_key_strip(fig, y_in: float) -> None:
    """The foot's first row: the instruction fills, then the striped mark of a
    reasoning-off setting -- what (a)-(c) draw that the model row does not key."""
    import fig_F2 as f2
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches()
    # the marks ride on the first panel in inch coordinates, unclipped, so the
    # foot adds no axes box for fs.density to count
    ax = fig.axes[0]
    inch = fig.dpi_scale_trans
    items = [("Instruction", None)] + [(a, a) for a in fs.ARMS] + [("reasoning off", "off")]
    widths = []
    for name, _ in items:
        t = fig.text(0, 0, name, fontsize=fs.FS_NOTE)
        widths.append(t.get_window_extent(r).width / fig.dpi)
        t.remove()
    sample, gap, sep = 0.07, 0.03, 0.08
    total = sum(w + (0 if k is None else sample + gap) for (_, k), w in zip(items, widths)) \
        + sep * (len(items) - 1) + 0.08
    x = (fw - total) / 2
    for (name, k), w in zip(items, widths):
        if k == "off":
            x += 0.08                         # a wider gap before the stripe
            f2.off_mark(ax, x + sample / 2, y_in, "o", "C", transform=inch)
        elif k is not None:
            ax.plot([x + sample / 2], [y_in], marker="o", ls="none",
                    transform=inch, clip_on=False, **f2.mark_style(k))
        if k is not None:
            x += sample + gap
        fig.text(x / fw, y_in / fh, name, fontsize=fs.FS_NOTE, color=fs.INK2,
                 ha="left", va="center")
        x += w + sep


# ------------------------------------------------------- the decomposition
DODGE_F = 0.46          # full width of the dodge when a family has three models

# (b)'s key, in inches.  Three leadered labels used to name the models where
# their points were, and once the panel was turned on its side each leader had to
# cross a row of somebody else's whiskers to reach its point.  A key in the one
# corner no data reaches costs the same ink and crosses nothing.  It is FRAMED,
# against the spec's "no legend boxes": three coloured marks loose in a panel of
# coloured marks are three more points, and only a frame says otherwise.
KEY_PAD = 0.045         # inside the frame
KEY_SAMPLE = 0.11       # the whisker sample, as long as a short interval
KEY_GAP = 0.035         # sample to word
KEY_ROW = 0.105         # row pitch; a 7 pt line is 0.097


def name_column(ax, width_in: float, lo: float, hi: float, names) -> float:
    """Open a column inside ``ax`` to the left of ``lo``, wide enough for ``names``.

    The forest-plot idiom this figure uses in (b) and (c): the rows are named in
    a column of 7 pt text inside the panel rather than as y tick labels, because
    ``v4-flash`` is 0.39 in wide and the gutter it would sit in is capped at
    0.30 in.  Returns the x the names start at; the caller draws them there and
    bounds the bottom spine to ``[lo, hi]``, so the column reads as a column and
    not as empty plot.
    """
    names_in = max(text_width(ax, n) for n in names) + 0.06
    span = (hi - lo) / (1 - names_in / width_in)
    x_name = hi - span
    ax.set_xlim(x_name, hi)
    ax.spines["bottom"].set_bounds(lo, hi)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return x_name


def model_key(ax, models, width_in: float, height_in: float) -> None:
    """Name the models once, in a framed key in the top right of ``ax``.

    Each entry is the mark the panel actually draws -- a filled dot over a
    capless whisker in the model's colour -- so the key is read off the panel
    rather than learned from it. Laid out in inches from the measured words,
    then converted to axes fractions, because the panel's own x is a rate and
    its y is a row index and neither is a length.
    """
    names = [fs.MODEL_SHORT[m] for m in models]
    w = 2 * KEY_PAD + KEY_SAMPLE + KEY_GAP + max(text_width(ax, n) for n in names)
    h = 2 * KEY_PAD + KEY_ROW * (len(models) - 1) + 0.075
    x1, y1 = 0.995, 0.995                       # flush to the panel's top right
    x0, y0 = x1 - w / width_in, y1 - h / height_in
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes,
                               facecolor=fs.SURFACE, edgecolor=fs.AXIS, lw=0.6,
                               zorder=6))
    xs = x0 + KEY_PAD / width_in                # left edge of the sample
    for i, m in enumerate(models):
        y = y1 - (KEY_PAD + 0.0375 + i * KEY_ROW) / height_in
        colour = fs.MODEL_COLOR[m]
        line = plt.Line2D([xs, xs + KEY_SAMPLE / width_in], [y, y],
                          transform=ax.transAxes, color=colour,
                          lw=fs.WHISKER_LW, solid_capstyle="butt", zorder=7)
        ax.add_artist(line)
        ax.plot([xs + 0.5 * KEY_SAMPLE / width_in], [y], transform=ax.transAxes,
                marker=fs.MODEL_MARKER[m], linestyle="none", zorder=8,
                **fs.marker_style(True, colour))
        ax.text(xs + (KEY_SAMPLE + KEY_GAP) / width_in, y, fs.MODEL_SHORT[m],
                transform=ax.transAxes, fontsize=fs.FS_NOTE, color=colour,
                ha="left", va="center", zorder=8)


def panel_benchmarks(ax) -> None:
    """Measured withholding per (model, benchmark): the benchmark moves the models apart.

    The rates are the per-setting posteriors of the no-instruction condition B, pooled by
    inverse-variance weighting where a pair ran more than once -- not a fitted
    quantity.  The crossed fit that used to supply them carried a benchmark term
    and a model-benchmark interaction estimated from six benchmarks and three
    models, and nothing in the paper quoted either (9/12); reading the rates
    directly says the same thing about the ordering and says it of the data.

    Points only, never joined: the axis of benchmarks is an unordered set and
    a line between two of them would claim a trend that is not there (9/7
    comment).

    Turned on its side on 9/9 to match (c), which sat under it then and stands
    beside it now: both are one row per category with a value and its interval
    running across, so both are read the same way, the six family names are set
    horizontally in a name column instead of squeezed into six x ticks, and the
    two panels never ask the eye to change axes between them.

    The dodge covers only the models a family actually has.  A fixed three-slot
    dodge put the single gpt-5.6-luna point of assetops and musique at the slot
    the eye reads as the third model, so two families appeared to report a model
    that never ran on them; centred, one point sits on its family's row.
    """
    d = pd.read_csv(DECOMP / "decomposition_fitted.csv")
    order = d.groupby("family")["c_fit_mean"].mean().sort_values().index.tolist()
    ypos = {f: i for i, f in enumerate(order)}
    present = {f: [m for m in fs.MODELS if not d[(d.model == m) & (d.family == f)].empty]
               for f in order}
    off = {}
    for f, models in present.items():
        k = len(models)
        for i, m in enumerate(models):
            off[(f, m)] = 0.0 if k == 1 else (i / (k - 1) - 0.5) * DODGE_F
    note("(b) models per benchmark: " + "  ".join(
        "%s %s" % (FAMILY_SHORT.get(f, f), "/".join(fs.MODEL_SHORT[m] for m in ms))
        for f, ms in present.items()))

    names = [FAMILY_SHORT.get(f, f) for f in order]
    width_in = ax.get_position().width * FIG_W
    x_name = name_column(ax, width_in, 0.0, 1.0, names)
    span = 1.0 - x_name
    ax.set_ylim(len(order) - 1 + 0.7, -0.7)      # first row at the top, as in (c)

    for m in fs.MODELS:
        g = d[d.model == m]
        if g.empty:
            continue
        y = np.array([ypos[f] + off[(f, m)] for f in g.family], dtype=float)
        colour = fs.MODEL_COLOR[m]
        fs.whisker(ax, y, g.c_fit_lo.to_numpy(), g.c_fit_hi.to_numpy(), colour,
                   orient="h")
        # The SHAPE is the model. Colour says whether the model reasons before it
        # answers, and two models share each hue since 9/15, so a row of dots in
        # one hue would be two models drawn as one series.
        ax.plot(g.c_fit_mean.to_numpy(), y, linestyle="none",
                marker=fs.MODEL_MARKER[m], zorder=4,
                **fs.marker_style(True, colour))
        note("(b) %-18s " % m + "  ".join(
            "%s %.2f [%.2f, %.2f]" % (FAMILY_SHORT.get(f, f), v, a, b)
            for f, v, a, b in zip(g.family, g.c_fit_mean, g.c_fit_lo, g.c_fit_hi)))
    model_key(ax, DRAWN_MODELS, width_in, ax.get_position().height * FIG_H)
    for f, name in zip(order, names):
        ax.text(x_name, ypos[f], name, fontsize=fs.FS_NOTE, color=fs.INK2,
                ha="left", va="center")
    fs.ticks01(ax, "x")
    ax.set_xlabel("withholding rate", labelpad=0)
    ax.xaxis.label.set_x((0.5 - x_name) / span)   # on the axis line, not the box
    # The row axis has no scale, so it is named where a y axis name would go.
    ax.text(-0.075, 0.5, "benchmark", transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=fs.FS_AXIS, color=fs.INK)
    finish(ax)
    ax.spines["left"].set_visible(False)


def text_width(ax, s: str, fontsize: float = fs.FS_NOTE) -> float:
    """Width of ``s`` in inches, measured in the figure's own font."""
    t = ax.text(0, 0, s, fontsize=fontsize)
    w = t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
    t.remove()
    return w / ax.figure.dpi


def panel_effects(ax) -> dict:
    """The withholding rate split into a benchmark, a model and an instruction part.

    Transposed on review: the terms are ROWS, grouped under three
    block names (benchmark, model, instruction), and the effect on the logit
    scale runs along x. One fit, over every benchmark at once (``--crossed``):

        logit c = mu + beta[benchmark] + alpha[model] + kappa[instruction]
                  + delta[model x benchmark] + e

    The three blocks are beta, alpha and kappa.  ``delta`` is fitted and not
    drawn: what it says -- that which model withholds more changes with the
    benchmark -- is the variation in (a)-(c).

    A model's row draws its effect in that model's own SHAPE, so the model
    block doubles as the shape key for (a)-(c); an instruction's row draws it
    in that instruction's fill (A blue, C open), as (a)-(c) do.
    """
    eff_filt = load_effects()
    shares = pd.read_csv(DECOMP / f"decomposition_shares{FIT_TAG}.csv").set_index("component")

    rows = []            # (y, label, mean, lo, hi, marker style)
    extra = []           # reasoning-off arms: (y, mean, lo, hi, style), unlabelled
    headers = []         # (y, block name)
    y = 0.0
    for term, label in (("beta", "benchmark"), ("alpha", "model"),
                        ("kappa", "instruction")):
        g = eff_filt[eff_filt.term == term]
        headers.append((y, label))
        y += 1.0
        if term == "alpha":
            # one row per base model, ordered by its main arm; the off arm of
            # the same model shares that row, open, as in (e)
            on = g[~g.level.str.endswith(OFF_TAG)].sort_values("mean")
            off = g[g.level.str.endswith(OFF_TAG)].set_index("level")
            for lvl, m, lo, hi in zip(on.level, on["mean"], on.lo, on.hi):
                mk = fs.MODEL_MARKER.get(lvl, "o")
                ms = 3.6                # the star grows in solid_thin_marks
                rows.append((y, fs.MODEL_SHORT.get(lvl, lvl), m, lo, hi,
                             dict(marker=mk, ms=ms, mew=0.6, mfc=fs.INK, mec=fs.INK)))
                if lvl + OFF_TAG in off.index:
                    o = off.loc[lvl + OFF_TAG]
                    extra.append((y, o["mean"], o.lo, o.hi,
                                  dict(marker=mk, ms=ms, mew=0.6, mfc=fs.SURFACE,
                                       mec=fs.INK)))
                y += 1.0
            y += 0.35
            continue
        g = g.sort_values("mean")
        for lvl, m, lo, hi in zip(g.level, g["mean"], g.lo, g.hi):
            if term == "beta":
                name, style = FAMILY_SHORT.get(lvl, lvl), dict(
                    marker="o", **fs.marker_style(True, fs.INK))
            else:
                name, style = lvl, dict(marker="o", ms=INSTR_MS.get(lvl, 3.2) + 0.4,
                                        mew=0.5,
                                        mfc=INSTR_FACE.get(lvl, fs.INK),
                                        mec=INSTR_EDGE.get(lvl, fs.INK))
            rows.append((y, name, m, lo, hi, style))
            y += 1.0
        y += 0.35                     # air between two blocks

    fs.reference(ax, 0.0, "v")
    for yy, name, m, lo, hi, style in rows:
        fs.whisker(ax, [yy], [lo], [hi], fs.INK, orient="h", zorder=3)
        ax.plot([m], [yy], linestyle="none", zorder=4, **style)
        note("(d) %-8s %+.2f [%+.2f, %+.2f]" % (name, m, lo, hi))
    for yy, m, lo, hi, style in extra:
        fs.whisker(ax, [yy], [lo], [hi], fs.INK_MID, orient="h", zorder=2.9)
        ax.plot([m], [yy], linestyle="none", zorder=4.1, **style)
        note("(d) off@%-4.1f %+.2f [%+.2f, %+.2f]" % (yy, m, lo, hi))
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=fs.FS_NOTE, color=fs.INK2)
    y_last = rows[-1][0]
    ax.set_ylim(y_last + 0.7, -0.7)
    # the left spine in one piece per block, so no block name crosses it (9/24)
    ax.spines["left"].set_visible(False)
    starts = [h[0] for h in headers] + [np.inf]
    for h0, h1 in zip(starts[:-1], starts[1:]):
        ys = [r[0] for r in rows if h0 < r[0] < h1]
        if ys:
            ax.plot([0, 0], [min(ys), max(ys)], transform=ax.get_yaxis_transform(),
                    color=fs.AXIS, lw=0.6, solid_capstyle="butt", clip_on=False,
                    zorder=1)
    # the block names head their rows, flush with the row names' left edge
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    left_px = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels())
    x_head = ax.transAxes.inverted().transform((left_px, 0))[0]
    for yy, label in headers:
        ax.text(x_head, yy, label, transform=ax.get_yaxis_transform(),
                fontsize=fs.FS_AXIS, color=fs.INK,
                ha="left", va="center")

    lim = 1.06 * max(abs(eff_filt[eff_filt.term.isin(["beta", "alpha", "kappa"])].lo.min()),
                     abs(eff_filt[eff_filt.term.isin(["beta", "alpha", "kappa"])].hi.max()))
    ax.set_xlim(-lim, lim)
    fs.ticks01(ax, "x", (-2, 0, 2))
    ax.set_xlabel("effect on logit $c$", labelpad=0)
    finish(ax)
    note("(d) variance shares " + "  ".join(
        "%s %.0f%%" % (k, 100 * shares.loc[k, "mean"]) for k in shares.index))
    return {k: 100 * shares.loc[k, "mean"] for k in shares.index}


def finish(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    # A gutter is capped at 0.30 in and has to hold a tick label (0.15), a 2 pt
    # tick (0.03) and a rotated axis name (0.11).  What is left for the two pads
    # is 0.01 in, so both are pulled in to it.
    ax.tick_params(pad=TICK_PAD)
    for axis in (ax.xaxis, ax.yaxis):
        axis.labelpad = 0


# -------------------------------------------------------------------- figure
# 9/24 review, second pass: landscape, as F2. The three benchmark panels
# (a)-(c) run along the top and share one y name; under them (e), what a model
# tables, and (f), the evidence pooling/utilization plane (appendix B4's panel,
# drawn by fig_F2.du_panel). (d), the decomposition, takes the right column at
# full height. x, y, w, h in inches.
ARM_KEY_H = 0.15        # the foot's second row: instruction fills, reasoning off
# 9/24 layout: three columns. (a)-(c), the three benchmarks, stacked on the
# left; (d), the decomposition, in the middle at the columns' full height;
# (e) over (f), the evidence panels, on the right. The keys sit at the foot.
KEY_Y0 = 0.08            # the foot's model row; the instruction row sits over it
_BOT = KEY_Y0 + ARM_KEY_H + 0.44             # the two key rows, then x names
_S_H, _S_GAP = 0.55, 0.38                    # a scatter; its tick labels + letter
_TOP = _BOT + 3 * _S_H + 2 * _S_GAP          # the ceiling of every column
FIG_W, FIG_H = fs.WIDTH_IN, _TOP + 0.28
_EF_GAP = 0.64                               # (e)'s two-line x name + (f)'s letter
_EF_H = (_TOP - _BOT - _EF_GAP) / 2
_D_H = _TOP - _BOT       # (d) takes the column's full height (9/24)
BOXES = {"a": (0.60, _BOT + 2 * (_S_H + _S_GAP), 1.30, _S_H),
         "b": (0.60, _BOT + _S_H + _S_GAP, 1.30, _S_H),
         "c": (0.60, _BOT, 1.30, _S_H),
         "d": (2.66, _TOP - _D_H, 0.92, _D_H),
         "e": (4.30, _BOT + _EF_H + _EF_GAP, 1.12, _EF_H),
         "f": (4.30, _BOT, 1.10, _EF_H)}
LETTER_DX = {"a": -0.02, "b": -0.02, "c": -0.02, "d": 0.62, "e": 0.62,
             "f": 0.62}
F_STRIP = 1.0            # (f)'s rows open one row over the top for its note


# ------------------------------------------------------ (e) tabled evidence
# 9/24: both reasoning settings of a model on that model's row -- filled is
# reasoning on (the run every other panel draws), open is off -- so the fill
# can no longer say which side the evidence backs. Colour says it instead:
# ink for the correct answer, vermillion for the agent's own answer.
TAB_COLOUR = {"truth": fs.INK, "own": fs.VERM}
TAB_NAME = {"truth": "correct answer", "own": "own answer"}
TAB_STRIP = 1.5          # rows opened over the top row: a one-line key


def panel_tabled(ax) -> None:
    """Share of what a model tables that backs the correct / its own answer."""
    import evidence as f4
    # 9/24: the main setting only; the reasoning-off arms are B3's
    order, _, share_on = f4.tabled_summary()
    for i, m in enumerate(order):
        for key in ("truth", "own"):
            v, lo, hi = share_on[m][key]
            c = TAB_COLOUR[key]
            fs.whisker(ax, [i], [lo], [hi], c, orient="h", zorder=2.4)
            ax.plot([v], [i], marker="o", ms=3.4, ls="none", zorder=3.2,
                    mfc=c, mec=c, mew=0.8)
            note("(e) %-9s %-5s %.3f [%.3f, %.3f]"
                 % (fs.MODEL_SHORT[m], key, v, lo, hi))
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([fs.MODEL_SHORT[m] for m in order])
    ax.set_ylim(len(order) - 1 + 0.6, -0.6 - TAB_STRIP)
    ax.spines["left"].set_bounds(0, len(order) - 1)
    ax.set_xlim(0, 1.06)
    fs.ticks01(ax, "x")
    ax.set_xlabel("share of\ntabled evidence", labelpad=1, linespacing=1.0)
    # key, two lines: the two colours
    trans = ax.get_yaxis_transform()
    top = -0.6 - TAB_STRIP
    # what the rows pool over, as (f) names its instruction (9/24)
    # 9/25: the pooling note is left to the caption, which says it
    # 9/25: F3 is 23 % shorter, so the two colours share one row
    rows = [(top + 0.75, [(0.02, TAB_COLOUR["truth"], True, "correct"),
                         (0.55, TAB_COLOUR["own"], True, "own answer")])]
    for y, entries in rows:
        for x, c, filled, word in entries:
            ax.plot([x], [y], transform=trans, marker="o", ms=3.4, ls="none",
                    mfc=c if filled else fs.SURFACE, mec=c, mew=0.8,
                    clip_on=False)
            ax.annotate(word, (x, y), xycoords=trans, xytext=(3, 0),
                        textcoords="offset points", fontsize=fs.FS_SMALL,
                        color=fs.INK2, va="center")
    finish(ax)


# ------------------------------------------------- (f) pooling, utilization
DU_GAP = 0.26                               # inches between the two columns


def panel_du_rows(fig, box) -> tuple:
    """Pooling and utilization rates at instruction A, one row per model on
    (e)'s rows; B4 draws all three instructions with the same function (9/24).
    """
    import fig_B4 as b4
    x, y, w, h = box
    w_c = (w - DU_GAP) / 2
    ax_d = fig.add_axes([x / FIG_W, y / FIG_H, w_c / FIG_W, h / FIG_H])
    ax_u = fig.add_axes([(x + w_c + DU_GAP) / FIG_W, y / FIG_H, w_c / FIG_W,
                         h / FIG_H])
    # at A alone utilization spans 0.88-1, so its axis is cut to that
    b4.du_rows(ax_d, ax_u, arms=("A",), strip=F_STRIP,
               xlabel_size=fs.FS_NOTE, d_ticks=(0.2, 0.6),
               u_lim=(0.76, 1.02), u_ticks=(0.8, 1),
               labels=("pooling", "utilization"))   # "rate" would collide
    for ax in (ax_d, ax_u):
        finish(ax)
    ax_d.text(0.0, -0.6 - F_STRIP + 0.45, "instruction A",
              transform=ax_d.get_yaxis_transform(), fontsize=fs.FS_SMALL,
              color=fs.INK2, va="center", ha="left")
    return ax_d, ax_u


# Fonts bumped one step above the spec's global 8/7 pt (9/15): the panels
# shrank when the figure was made compact, and at the smaller size the default
# scale reads faint next to the rest of the paper. Applied only in this
# script's own process, so F1/F2/etc. are untouched.
fs.FS_AXIS = 9
fs.FS_NOTE = 8


def main() -> None:
    global DRAWN_MODELS
    fs.apply()
    settings = load_settings()

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def box(key):
        x, y, w, h = BOXES[key]
        return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])

    ax = {k: box(k) for k in BOXES}

    # Upper row: three scatter plots by benchmark. The model shape key goes
    # in (a): one panel needs it, and (a) is read first.
    benchmarks = [fs.BENCHMARKS[k]["figure_name"] for k in fs.F3_PANELS]
    bench_names = [fs.BENCHMARKS[k]["short"] for k in fs.F3_PANELS]
    fitted = pd.read_csv(DECOMP / "decomposition_fitted.csv")
    seen = (set(settings.loc[settings["benchmark"].isin(benchmarks), "model"]
                .map(ID_TO_MODEL)) | set(fitted["model"]))
    DRAWN_MODELS = [m for m in fs.MODELS if m in seen]
    if len(DRAWN_MODELS) < len(fs.MODELS):
        note("models in fs.MODELS with no cell in this figure, left out of its"
             " keys: %s" % ", ".join(m for m in fs.MODELS if m not in seen))
    settings_off = load_settings(off=True)
    for k, (bname, label) in enumerate(zip(benchmarks, bench_names)):
        panel_gain(ax[chr(97 + k)], settings, settings_off, benchmark=bname,
                   title=label)

    # Lower row: effects plot
    shares = panel_effects(ax["d"])

    # Lower row: what a model tables, and where the evidence goes
    import fig_F2 as f2
    panel_tabled(ax["e"])
    ax["f"].remove()
    panel_du_rows(fig, BOXES["f"])
    f2.solid_thin_marks(fig)
    # (a)-(c) share both quantities, so the middle one names y for the column
    # and the bottom one names x; each keeps its own scale (9/24)
    for k in "ac":
        ax[k].set_ylabel("")
    for k in "ab":
        ax[k].set_xlabel("")

    for k, (bx, by, bw, bh) in BOXES.items():
        fig.text((bx - LETTER_DX[k]) / FIG_W, (by + bh + 0.06) / FIG_H, f"({k})",
                 ha="left", va="bottom", fontsize=fs.FS_AXIS, fontweight="bold",
                 color=fs.INK)

    # the model shapes and the instruction / reasoning-off marks of (a)-(c)
    fs.model_key_strip(fig, DRAWN_MODELS, KEY_Y0, sep=0.09)
    arm_key_strip(fig, KEY_Y0 + ARM_KEY_H)
    caption(settings, None, shares)
    for out in fs.save(fig, "F3"):
        print(out)
    print()
    for line in NOTES:
        print(line)


def caption(d, b, shares) -> None:
    """Everything the panels may not say (spec section 5), read off the data."""
    # Read effects data for (d), through the same loader the panel uses, so
    # the caption's counts match what the panel actually draws.
    eff = load_effects()
    n_cells = len(pd.read_csv(DECOMP / f"decomposition_cells{FIT_TAG}.csv"))
    eff = eff[eff.term.isin(["beta", "alpha", "kappa"])]
    n_eff = len(eff)
    n_zero = int(((eff.lo <= 0) & (eff.hi >= 0)).sum())
    widest = eff.loc[(eff.hi - eff.lo).idxmax()]
    spelled = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight",
               "nine", "ten")
    count = spelled[n_eff] if n_eff < len(spelled) else str(n_eff)
    said = lambda k: spelled[k].capitalize() if k < len(spelled) else str(k)
    cover = ("All %s intervals cover zero" % count if n_zero == n_eff
             else "%s of the %s intervals cover zero" % (said(n_zero), count))
    n_beta_g = int((eff.term == "beta").sum())
    n_models_g = int((eff.term == "alpha").sum())
    n_kappa_g = int((eff.term == "kappa").sum())
    # (a)-(c) fit every drawn setting, reasoning on and off (9/24)
    on_, off_ = load_settings(), load_settings(off=True)
    fit_n = "; ".join(
        "%s %d, %d of them off" % (FAMILY_LONG.get(k, k),
                                  int((on_.benchmark == b).sum() + (off_.benchmark == b).sum()),
                                  int((off_.benchmark == b).sum()))
        for k, b in ((k, fs.BENCHMARKS[k]["figure_name"]) for k in fs.F3_PANELS))

    fs.write_caption("F3", [
        "Throughout, $\\hat{c}$ is the counting estimator of X1, conditioned on the",
        "answer an agent holds entering the round, and $c^{*}$ is the critical",
        "withholding rate of that same setting, $c^{*} = \\gamma / (\\gamma + a)$.",
        "(a-c) Gain from discussion against the distance from the threshold $\\hat{c} - c^{*}$,",
        "one panel per benchmark: HiddenBench, MuSiQue, and MedEInst.  One mark is one",
        "setting -- a (run, instruction) cell -- drawn at its point estimate, with no",
        "whisker.  Marker shape is the model (the key at the foot) and the fill the",
        "instruction, blue at A, grey at B and open at C, as in Figure 2(d).  Striped",
        "marks are the same models run with reasoning off.  The orange line is a",
        "linear fit over every setting drawn, reasoning on and off (%s), and the" % fit_n,
        "band its 95\\%% interval from %d bootstrap resamples of those settings." % BAND_BOOT,
        "(d) The same rates split into a benchmark part, a model part and an",
        "instruction part, on the logit scale, from one fit over every benchmark at",
        "once: logit $c = \\mu$ + benchmark + model + instruction + (model $\\times$",
        "benchmark) + cell, over the %d cells of the standard protocol -- a complete" % n_cells,
        "graph, three rounds, four or five agents -- and the instructions A, B and C.",
        "The rows are the %s benchmarks, the %s models and the %s instructions the fit"
        % tuple(spelled[k] if k < len(spelled) else str(k)
                for k in (n_beta_g, n_models_g, n_kappa_g)),
        "resolves against B, grouped in three blocks; a model's effect is drawn in",
        "that model's shape, and an instruction's in its fill.  The model $\\times$",
        "benchmark term is fitted but not drawn, because what it says -- that which",
        "model withholds more changes with the benchmark -- is the variation in (a-c).",
        "%s -- the widest is %s at [%+.2f, %+.2f] -- so the figure separates no single"
        % (cover, fs.MODEL_SHORT.get(widest.level, FAMILY_LONG.get(widest.level, widest.level)),
           widest.lo, widest.hi),
        "benchmark, model or instruction from the average at this sample size; what the",
        "fit does resolve is the variance split.  The benchmark takes %.0f \\%% of the"
        % shares["task"],
        "variance and its interaction with the model another %.0f \\%%, the instruction"
        % shares["interaction"],
        "%.0f \\%%, the model %.0f \\%% and the cell %.0f \\%%."
        % (shares["instruction"], shares["model"], shares["residual"]),
        "Intervals are 95 \\% throughout: a percentile bootstrap in (a-c), and posterior",
        "credible intervals in (d).",
    ])


if __name__ == "__main__":
    main()
