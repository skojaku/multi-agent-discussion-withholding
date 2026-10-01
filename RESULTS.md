# Paper item → script → rule → inputs → output

Every figure and quantitative claim of the paper, the script, rule and inputs
that produce it, and what the repository does not reproduce. Everything needed
is in this repository and the data archive. Rules
are in `workflow/rules/`; `data/` is the data archive (README, *Where things are*);
`results/` is what the workflow writes. Every run is on the separate-probe
protocol (the private answer asked in its own call, Appendix B.2).

## Figures (entry point `figures`, rule `figure`)

| Paper item | Script(s) | Inputs (under the estimates tree unless noted) | Output | Note |
|---|---|---|---|---|
| Fig. 1 (§2) | `figures/fig_F1.py`, `figures/figstyle.py` | `theory/abm_phase.csv` (N = 100 rows), `theory/meanfield_boundary.csv` | `results/figures/F1.pdf` | |
| Fig. 2 (§4) | `figures/fig_F2.py` (+ `fig_B3.py` for panels e–g, `evidence.py`) | per model: `runs/<run>/cbrm_params.csv`, `hp_accuracy.csv`, and the same for its reasoning-off run; `sweep.csv`; `data/items/hidden_profile.jsonl` | `results/figures/F2.pdf` | `evidence.py` holds the evidence-level estimates F2, F3 and B4 share. |
| Fig. 3 (§4) | `figures/fig_F3.py` (+ `fig_F2.py`, `fig_B4.py`, `evidence.py`) | `sweep.csv`; `decomposition/decomposition_{effects,shares,cells}_crossed_onestage.csv`; `decomposition/decomposition_fitted.csv`; the Fig. 2 inputs and `model_comparison.csv`; `data/transcripts/<main run>/events.jsonl` (what each model tables) | `results/figures/F3.pdf` | |
| Fig. B3 (Appendix B) | `figures/fig_B3.py` | `sweep.csv` rows of each model's main and reasoning-off run | `results/figures/B3.pdf` | |
| Fig. B4 (Appendix B) | `figures/fig_B4.py` | per model `cbrm_params.csv` (`ev_D`, `ev_U`), `model_comparison.csv`, transcripts | `results/figures/B4.pdf` | |

## Estimates (entry point `estimates`; theory: `theory`)

| Paper item | Script(s) | Rule | Output (under `results/estimates/`) |
|---|---|---|---|
| B.2 posteriors of c, a, ρ, r, γ, c\*, P(c > c\*) per setting | `analysis/analyze_cbrm.py` → `libs/cbrm` | `cbrm_params` | `runs/<run>/cbrm_params.csv` |
| Accuracy and evidence tables of a hidden-profile run | `analysis/analyze_hidden_profile.py` | `hp_tables` | `runs/<run>/hp_{accuracy,params,manipulation,phi}.csv` |
| Per-model table (tally accuracy; orders Fig. 3 e) | `analysis/compare_models.py` | `compare_models` | `model_comparison.csv` |
| Every (run, instruction) setting, re-estimated | `analysis/sweep.py` | `sweep` | `sweep.csv` |
| Per-task withholding counts (input of the one-stage fit) | `analysis/task_counts.py` | `task_counts` | `task_counts.csv` |
| B.4 / Fig. 3 d: one-stage decomposition over four benchmarks (binomial likelihood, Pólya–Gamma Gibbs, 4 × 210,000 sweeps, reasoning-off runs pooled into their model) | `analysis/decomposition.py --one-stage --reasoning` | `decomposition_onestage` | `decomposition/decomposition_{cells,effects,shares,fitted}_crossed_onestage.csv`, and a sampler log with the split R-hat |
| Decomposition on the hidden profile only (Fig. 3 reads its per-benchmark measured rates) | `analysis/decomposition.py` | `decomposition_single` | `decomposition/decomposition_*.csv` |
| Two-stage crossed decomposition (not in the paper; a cross-check) | `analysis/decomposition.py --crossed` | `decomposition_crossed` | `decomposition/decomposition_*_crossed.csv` |
| Fig. 1 c: simulated phase plane and mean-field boundary | `theory/run_abm.py`, `theory/run_meanfield.py` | `abm_phase`, `meanfield_boundary` | `theory/*.csv` |
| Every run's settings as recorded | — | `run_settings` | `run_settings.csv` |
| Transcript audit (gates everything) | `analysis/audit_gate.py` → `audit_transcripts.py` | `audit_transcripts` | `results/audit/transcripts.txt` |

## Claims (`results/numbers.json`)

The paper's main text quotes no point estimates; its quantitative statements
are directional (a rate rises with conformity, gains fall with the
margin, an interval covers zero). `config/paper_numbers.yaml` turns each into a
check with the sentence it comes from; `workflow/scripts/numbers/quote_numbers.py`
evaluates them and records the values behind each verdict (Spearman ρ with its
bootstrap interval, variance shares, effects, per-model shifts).

## Not reproduced here

| Statement | Where | Why |
|---|---|---|
| "a task-level cluster bootstrap gives intervals approximately 1.5 times wider" | B.2 | Computed in a separate estimator-validation experiment, not from the paper's transcripts. |
| A.2: the closed form agrees with the numerical solution to four decimals; the transition width scales as N^−1/2 | A.2 | Theory checks that are not part of the workflow. |
