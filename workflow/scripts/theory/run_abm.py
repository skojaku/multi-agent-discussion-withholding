"""Agent-based phase plane: the simulated grid of Fig. 1 (c).

N binary agents holding a private belief and a public statement, run to a
steady state on the annealed q-panel that matches the mean-field
neighbourhood, on a 41 x 41 grid of (withholding rate c = phi, initial accuracy
p), for N = 100 (drawn) and N = 400. The mean-field boundary from
``run_meanfield.py`` is overlaid on this grid in Fig. 1 (c).

The grid is independent per (N, phi) column -- the seed depends on phi only --
so it can be computed in chunks and concatenated in order; the result is the
same table as one pass.

Usage:
    python run_abm.py OUT.csv                      # the whole grid (~100 min)
    python run_abm.py OUT.csv --N 100 --chunk 0 --n-chunks 8
    python run_abm.py OUT.csv --merge PART.csv ...   # concatenate chunks in order
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbrm_theory import Params, simulate  # noqa: E402

BASE = Params(lam=0.5, rho=1.0, r=0.8, q=3, trigger="fraction")
SEED = 20260831
NS = (100, 400)
T = 120
N_P = 41
N_PHI = 41
N_REPS = 1000


def phase_rows(N: int, phis, n_reps: int = N_REPS, T: int = T, n_p: int = N_P) -> list:
    ps = np.linspace(0.0, 1.0, n_p)
    rows = []
    for phi in phis:
        prm = replace(BASE, phi=float(phi))
        for p in ps:
            out = simulate(N, float(p), prm, "panel", n_rounds=T,
                           n_reps=n_reps, seed=SEED + int(round(1000 * phi)))
            rows.append(dict(trigger=prm.trigger, N=N, T=T, phi=float(phi),
                             p=float(p), **{k: v for k, v in out.items()}))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--N", type=int, default=None, help="one group size only")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--merge", nargs="+", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.merge:
        pd.concat([pd.read_csv(p) for p in a.merge], ignore_index=True).to_csv(out, index=False)
        return
    phis = np.linspace(0.0, 1.0, N_PHI)
    mine = np.array_split(phis, a.n_chunks)[a.chunk]
    rows = []
    for N in ([a.N] if a.N else NS):
        rows += phase_rows(N, mine)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), "cells")


if __name__ == "__main__":
    main()
