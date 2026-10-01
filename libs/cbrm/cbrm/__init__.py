"""cbrm -- measure concealment, internalisation and repair in agent transcripts.

A meter, not an optimiser. Point it at an events log and it returns where the
team sits on the CBRM phase diagram, which parameter is the brake, and -- the
part that matters most in practice -- whether the transcript could support the
answer at all.

    import cbrm
    t = cbrm.load_events("data/run/events.jsonl")
    print(cbrm.diagnose(t, condition="C4"))

The input schema is described in the repository README (Data).
"""

from .bayes import posterior, summary as posterior_summary
from .latent import fit_latent, sequences
from .estimators import (Estimate, base_noise, by_dissent, by_round,
                         cluster_bootstrap,
                         counts, c_star, gap_series, interval, rates,
                         trigger_check)
from .evidence import (EvidenceAdapter, FactTallyAdapter, evidence_frame,
                       evidence_rates, limiting_factor)
from .io import (PROJECTIONS, Transcript, decisions, error_concentration,
                 load_events, load_items, p_hat, round0)
from .phase import cheapest_lever, levers, p_star, predict_rounds, verdict
from .report import Diagnosis, diagnose, diagnose_all

__version__ = "0.1"

__all__ = [
    "Diagnosis", "Estimate", "EvidenceAdapter", "FactTallyAdapter", "PROJECTIONS",
    "Transcript", "error_concentration",
    "base_noise", "by_dissent", "by_round", "cheapest_lever",
    "cluster_bootstrap", "counts",
    "c_star", "decisions", "diagnose", "diagnose_all", "evidence_frame",
    "evidence_rates", "gap_series", "interval", "levers", "limiting_factor",
    "fit_latent", "load_events", "load_items", "p_hat", "p_star",
    "posterior", "posterior_summary", "predict_rounds", "rates", "sequences",
    "trigger_check",
    "round0", "verdict",
]
