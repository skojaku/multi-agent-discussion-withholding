"""Mean-field phase boundary: the black curve of Fig. 1 (c).

meanfield_boundary.csv     p_c(phi), the separatrix crossing of the symmetric
                           initial condition, plus the two analytic landmarks
                           phi_saddle (the jump) and phi_absorbing (c*).

Deterministic; about a minute. ``phi`` is the withholding rate c and ``lam``
the internalization rate a in the paper's notation.

Usage: python run_meanfield.py OUT.csv [n_phi]
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbrm_theory import (  # noqa: E402
    Params, basin_boundary, phi_absorbing, phi_saddle,
)

# The reference parameterisation. rho = 1 and r = 0.8 say "an agent that engages
# with visible disagreement re-derives its answer, and re-deriving under
# challenge is better than its first pass (0.8 against a p that is below 0.5 on
# the items that matter)". lam = 0.5 is a deliberately mid-range guess at how
# often an agent that has publicly conceded actually updates.
BASE = Params(lam=0.5, rho=1.0, r=0.8, q=3, trigger="fraction")
LAMS = (0.25, 0.5, 1.0)
TRIGGERS = ("fraction", "any")


def boundary_table(n_phi: int = 101) -> pd.DataFrame:
    phis = np.linspace(0.0, 1.0, n_phi)
    rows = []
    for trigger in TRIGGERS:
        for lam in LAMS:
            prm0 = replace(BASE, lam=lam, trigger=trigger)
            sn, ab = phi_saddle(prm0), phi_absorbing(prm0)
            for phi in phis:
                prm = replace(prm0, phi=float(phi))
                rows.append(dict(trigger=trigger, lam=lam, q=prm.q, phi=float(phi),
                                 p_c=basin_boundary(prm), phi_saddle=sn,
                                 phi_absorbing=ab))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    n_phi = int(sys.argv[2]) if len(sys.argv) > 2 else 101
    out.parent.mkdir(parents=True, exist_ok=True)
    boundary_table(n_phi).to_csv(out, index=False)
    print("wrote", out)
