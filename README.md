# Multi-agent discussion gains less when dissent is withheld — reproduction

Paper: [arXiv:2609.38324](https://arxiv.org/abs/2609.38324). If you use this code or data, please cite:

```bibtex
@misc{mansuri2026withholding,
  title         = {Multi-agent discussion gains less when dissent is withheld},
  author        = {Mansuri, Chand Sahil and Wang, Xin and Li, Mengying and Acton, Bryan and Eckardt, Rory and Patel, Dhaval and Kojaku, Sadamori},
  year          = {2026},
  eprint        = {2609.38324},
  archivePrefix = {arXiv},
  primaryClass  = {physics.soc-ph},
  url           = {https://arxiv.org/abs/2609.38324}
}
```

This repository regenerates every figure and checks every quantitative claim of
the paper *"Multi-agent discussion gains less when dissent is withheld"* from the conversation logs of its experiments.

In one paragraph: teams of LLM agents discuss a task over three rounds. In each
round an agent states an answer in public and, in a separate call, gives its
private answer. An agent **withholds** when it states the majority's answer
while privately disagreeing. The paper's model predicts that discussion beats a
majority vote only while the withholding rate `c` stays below a critical rate
`c* = γ / (γ + a)`, where `a` is how often a withheld answer becomes the agent's
own belief (internalization) and `γ` the net rate at which disagreement moves
agents to the correct answer (net correction). The code here estimates `c`, `a`,
`γ` and `c*` from the logs with a Bayesian method, splits the withholding rate
into benchmark, LLM and instruction parts, and draws the figures.

`RESULTS.md` lists, for every figure and claim, the script, rule, inputs and
output, what the repository does not reproduce, and where the paper and its
data disagree.

## Contents

- [Quick start](#quick-start)
- [What each command reproduces](#what-each-command-reproduces)
- [Time and resources](#time-and-resources)
- [The three entry points](#the-three-entry-points)
- [Data](#data)
- [Rerunning the LLM experiments](#rerunning-the-llm-experiments)
- [Common problems](#common-problems)
- [Terms](#terms)
- [Layout](#layout)

## Quick start

Requirements: Linux or macOS, Python 3.12, [uv](https://docs.astral.sh/uv/),
and about 400 MB of disk for the data. Every Python dependency is pinned in
`uv.lock`.

```bash
git clone <this repository> withholding-debate && cd withholding-debate
uv sync                                   # creates .venv from uv.lock
# put the data archive's data/ directory at ./data (see Data)
uv run snakemake --rerun-triggers mtime -c4 figures
```

After a few minutes, `results/figures/*.pdf` holds the figures and
`results/numbers.json` every checked claim with the value computed here.

Always pass `--rerun-triggers mtime` (see [Common problems](#common-problems)).

## What each command reproduces

| Paper item | Output | Snakemake target (`uv run snakemake --rerun-triggers mtime -c4 <target>`) |
|---|---|---|
| Fig. 1 (model schematic and phase plane) | `results/figures/F1.pdf` | `results/figures/F1.pdf` |
| Fig. 2 (hidden profile task: withholding, threshold, gain, reasoning switch) | `results/figures/F2.pdf` | `results/figures/F2.pdf` |
| Fig. 3 (other benchmarks, decomposition, evidence sharing) | `results/figures/F3.pdf` | `results/figures/F3.pdf` |
| Fig. B3 (reasoning on against off, four quantities) | `results/figures/B3.pdf` | `results/figures/B3.pdf` |
| Fig. B4 (evidence pooling and utilization per instruction) | `results/figures/B4.pdf` | `results/figures/B4.pdf` |
| Every quantitative claim of §4 and B.4 | `results/numbers.json`, `results/numbers.csv` | `results/numbers.json` |
| All of the above | | `figures` |
| Estimates of B.2 (c, a, ρ, r, γ, c*, P(c > c*)) per setting | `results/estimates/runs/<run>/cbrm_params.csv` | `estimates` |
| Decomposition of B.4 / Fig. 3 d | `results/estimates/decomposition/decomposition_*_crossed_onestage.csv` | `estimates` |
| Theory tables behind Fig. 1 c | `results/estimates/theory/*.csv` | `theory` |
| Rerun of every LLM discussion | `results/transcripts/transcripts/<run>/events.jsonl` | `transcripts` |

`config/paper_numbers.yaml` lists every claim `numbers.json` checks, with the
sentence of the paper it comes from.

## Time and resources

Measured on one Linux workstation (x86-64). No GPU is needed except to serve
the one self-hosted model when rerunning the LLM experiments.

| Target | Wall time | CPU | Needs |
|---|---|---|---|
| `figures` | under a minute to 3 min | 1–6 cores | the data archive |
| `estimates` | about 30 min; the one-stage decomposition (4 chains × 210,000 Gibbs sweeps) takes most of it | 4–12 cores | the data archive |
| `theory` (+ `verify_theory`) | about 40 min on 14 cores (100 CPU-minutes for the agent-based phase plane) | many cores help | nothing |
| `transcripts` | days; see [Cost and time](#cost-and-time-estimate) | — | API keys and money |

## The three entry points

### (a) `figures`: estimates → figures and claims (minutes, no API keys)

```bash
uv run snakemake --rerun-triggers mtime -c4 figures
```

Reads the estimates released with the paper (`data/estimates/`), draws the
figures into `results/figures/` (PDF, 400-dpi PNG, and a caption file in
`results/figures/captions/`), and evaluates every claim into
`results/numbers.json`. The transcript audit runs first (see
[Common problems](#common-problems)).

The figures use DejaVu Sans, which ships with matplotlib, so they do not depend
on the fonts installed on the machine; matplotlib is pinned to 3.11.2, the
version most of the paper's figures were rendered with.

### (b) `estimates`: transcripts → estimates (about 30 min, no API keys)

```bash
uv run snakemake --rerun-triggers mtime -c8 estimates
uv run snakemake --rerun-triggers mtime -c14 theory verify_theory    # optional, ~100 CPU-minutes
```

Re-estimates everything from the transcripts in `data/transcripts/`: the
per-setting posteriors (`cbrm_params.csv`, B.2), the accuracy tables, the sweep
over every (run, instruction) setting (`sweep.csv`), the per-task withholding
counts (`task_counts.csv`), the three decomposition fits (hidden profile only;
four benchmarks fitted to per-setting rates; and the one-stage fit to per-task
counts that the paper reports), and a table of each
run's settings as recorded in its `events.meta.json`
(`results/estimates/run_settings.csv`). Outputs go to `results/estimates/` in
the same layout as `data/estimates/`, and `results/verification/estimates.csv`
compares each recomputed file with the released one, cell by cell. To draw the figures from the recomputed estimates instead, run
`uv run snakemake --rerun-triggers mtime -c4 figures --config estimates_source=recomputed`
(or copy `config/config.template.yaml` to `config/config.yaml` and set the key
there). This also makes Fig. 1 read the recomputed theory tables, so it
runs `theory` (about 100 CPU-minutes) unless those exist already.

Randomness: every estimator is seeded (bootstrap `seed=0`; decomposition chains
seeds 0–3; figure bootstraps fixed per script), so reruns give identical files.

### (c) `transcripts`: rerun the LLM discussions (API keys, costs money)

```bash
uv run snakemake --rerun-triggers mtime -c1 -n -p transcripts   # dry run: prints every command
uv run snakemake --rerun-triggers mtime -c1 results/transcripts/transcripts/hp_models/luna/events.jsonl
```

One job per run in `config/transcripts.yaml`, each with the settings its paper
transcript was produced with. New transcripts go to `results/transcripts/`,
never overwrite `data/`, are write-protected once finished, and are audited.
To estimate from them, copy them over `data/transcripts/` (or point `data_dir`
at a copy) and run (b).

## Data

The data is not in git. It will be deposited on Zenodo, and its DOI will be
added here. Unpack the archive at the repository root so that these paths
exist:

```
data/transcripts/<run>/events.jsonl        one line per agent and round: task_id, agent, round,
                                           branch (the instruction; r0 for the independent round 0),
                                           private, public, share (the finding put on the table), raw reply
data/transcripts/<run>/events.meta.json    the run's settings: model, endpoint, reasoning effort,
                                           token budget, probe, instructions, temperature, seed
data/items/*.jsonl                         the task files the runs used
data/estimates/...                         released estimates (same layout as results/estimates/)
data/MANIFEST.json                         sha256 of every file
```

`config/data_manifest.yaml` lists the runs whose transcripts are in the archive.
Machine- and account-specific fields were removed from `events.meta.json`;
transcripts are unchanged.

### Excluded run

One run on MedEInst, `medeinst/base5_qwen` (Qwen3.8-27B), is not in the data,
the figures or the fit. 179 of its 5,400 discussion rows (3.3 %) carry no public
answer: 155 replies are empty and 24 stop before the PUBLIC line, so no parser
can recover them. That exceeds the 2 % threshold of the transcript audit (see
[Common problems](#common-problems)): the withholding rate of such a run would
be computed over a silently shrunken set of opportunities to dissent.

### Licenses

Code: MIT (`LICENSE`). Data and transcripts: CC BY 4.0 (`LICENSE-DATA`). The
task files are built from HiddenBench, MuSiQue and MedEInst, which keep their
own licenses.

## Rerunning the LLM experiments

### Endpoints

| `endpoint` | Models | Needs |
|---|---|---|
| `openrouter` | gpt-4o-mini, gpt-5.6-luna, glm-5.3-flash, nemotron-3-super-120b-a12b, ling-3.0-flash, seed-2.0-mini | `OPENROUTER_API_KEY` |
| `vertex` | gemini-3.8-flash, through Vertex AI's OpenAI-compatible surface (location `global`) | `gcloud auth application-default login`; `VERTEX_PROJECT` or `GOOGLE_CLOUD_PROJECT`. The access token is refreshed every 50 minutes. |
| `openai-compatible` | mixtral-8x22b-instruct, served by ollama behind an OpenAI-compatible chat gateway | `OPENAI_COMPATIBLE_URL` (the chat-completions URL) and `OPENAI_COMPATIBLE_API_KEY` |
| `local` | Qwen3.8-27B (8-bit) with a DFlash2 draft, on llama.cpp | `LOCAL_BASE_URL`: one or more comma-separated `/v1/chat/completions` URLs |

### Settings of every run

`config/transcripts.yaml` is the record of every run: model id, endpoint,
reasoning effort, completion budget (`max_tokens`, raised automatically up to
`max_budget` when a reply is cut off), probe, instructions, temperature, seed
where one was set, and the number of parallel workers. It is built from each
run's `events.meta.json`, with the instructions read from the transcript
itself, because a meta file is rewritten by every invocation on the same
transcript and records only the last one (two runs record `conditions: PS`
although their transcripts hold A, B, C and more). `results/estimates/run_settings.csv`
tabulates it and names the fields a run's meta file left unset.

What is not recorded, and so not pinned: the OpenRouter provider a call was
routed to (no run pinned one; routing was OpenRouter's default), top_p (never
set; provider default), the reasoning setting where it is "provider default",
and a seed for most runs (only the replicate runs set one; temperature is 0.7
throughout). Fields missing from older meta files (`endpoint`,
`reasoning_effort`, `max_budget`, `provider`) mean the harness default: endpoint
`openrouter`, reasoning and routing at the provider default, `max_budget` 16384.

Reasoning setting per run (`reasoning_effort` in `events.meta.json`):

| Model | Main run (every figure) | Second run (Fig. 2 e–g, B3) |
|---|---|---|
| gpt-4o-mini, mixtral-8x22b | no reasoning | — |
| glm-5.3-flash | `low` | `high` (`*_highthink`; not drawn, not in the fit) |
| gpt-5.6-luna, ling-3.0-flash, nemotron-3-super-120b-a12b, seed-2.0-mini | provider default | `off` (`*_nothink`) |
| Qwen3.8-27B | default (thinking on) | `off` (`*_nothink`) |
| gemini-3.8-flash | provider default | `low`, the lowest its API accepts (`*_nothink`) |

A `*_nothink` suffix therefore means "the second run", not always "reasoning
off". Fig. 2 e–g, Fig. 3 a–c and Fig. B3 draw these runs beside the main runs,
and the decomposition (Fig. 3 d, B.4) pools each model's reasoning-off runs
with its main runs, as the paper states (`decomposition_pool_reasoning_off:
true` in the config). The `*_highthink` run of glm-5.3-flash is in no figure
and no fit.

### Self-hosted models

- **Qwen3.8-27B**: unsloth's `Qwen3.8-27B-UD-Q8_K_L.gguf` with the draft
  `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, served by llama.cpp's `llama-server` with
  `--spec-type draft-dflash --spec-draft-n-max 7 -ngl 999 -ngld 999
  --ctx-size 65536 -np 8 --no-kv-unified --flash-attn on --jinja`, on two to
  four data-centre GPUs per instance. The llama.cpp build was not recorded.
- **mixtral-8x22b-instruct**: the ollama tag `mixtral:8x22b-instruct` (ollama's
  default quantization for that tag), behind an OpenAI-compatible chat gateway. The ollama
  version was not recorded.

### Cost and time (estimate)

The paper's transcripts hold about 670,000 model calls: 253,000 on the hidden
profile task, 69,000 on HiddenBench, 202,000 on MedEInst and 148,000 on
MuSiQue. Prompts are roughly 600–900 input tokens (hidden profile, HiddenBench,
MedEInst) and 2,000–2,500 (MuSiQue), so a full rerun sends on the order of
0.7 billion input tokens, plus the hidden reasoning tokens of the reasoning
models, which can exceed the visible reply many times. Token counts and prices
were not logged per call, so no exact cost is given; at hosted prices of
US$0.1–0.5 per million input tokens the input side alone is on the order of
US$100–400, and reasoning tokens can multiply that. A single run (5,400–12,600
calls) took hours at 8–16 parallel workers; a self-hosted or rate-limited
endpoint managed 8–40 rows per minute. Rerun one run
first.

### pi

The debate harness calls chat-completions endpoints directly and does not use
the pi coding-agent CLI. The Snakefile nevertheless sets
`PI_CODING_AGENT_DIR=config/pi_config` for every job, so that any pi-based step
reads this experiment's own `config/pi_config/models.json` (sampling
parameters as `samplingParams` on the model entries) and never the shared
`~/.pi/agent/models.json`.

## Common problems

- **Use `--rerun-triggers mtime`.** With Snakemake's default triggers, editing
  a script or a parameter marks finished outputs stale and deletes them before
  re-running them; for a transcript, that deletes a paid run.
- **Resuming a transcript.** A partial `events.jsonl` resumes on
  `(task_id, agent, round, branch)` when the runner is called again on the same
  file. Snakemake will either refuse (incomplete output) or delete the file
  (`--forcerun`); take the command from `snakemake -n -p <target>` and run it
  directly instead.
- **The audit stops the workflow.** Every estimate, figure and claim depends on
  the transcript audit, which fails any run with more than 2 % of its public
  answers unparsed (its rates would be computed over a silently shrunken
  denominator). Every run in the archive passes; the one that did not is
  excluded ([Excluded run](#excluded-run)). A run that fails but must be kept
  can be listed in `audit_known_failures`, and is then named in
  `results/audit/transcripts.txt`.
- **Missing input files.** `figures` needs the data archive at `data/`. The
  error names the first missing file.
- **A figure differs from the paper's PDF by a few pixels.** Fig. 1 was
  rendered with matplotlib 3.10.6 and the others with 3.11.2 (the pinned
  version); the difference is anti-aliasing.
- **`figures` fails at `check_numbers`.** A claim of the paper does not hold on
  the estimates used; `results/numbers.json` says which and why.
- **Rerunning the debates.** Pin the provider when a model id is served by
  several OpenRouter providers, and set the completion budget and the timeout
  together.

## Terms

The paper's terms, and where the code or the data spell them differently:

| Paper | Meaning | In code and data files |
|---|---|---|
| agent | one LLM instance in the team | `agent` (a few variable names say `seat`) |
| statement, public answer | the answer an agent states to the team | `public` |
| private answer, belief | the answer the agent gives in a separate call no other agent sees | `private` |
| withhold, withholding rate `c` | state the majority's answer while privately disagreeing | `conceal`, `n_conceal`, `c`, `c_post` |
| opportunity to dissent | a round where the agent's previous private answer differs from the visible majority | `n_contra` |
| internalization rate `a` | a withheld answer becomes the private answer | `a`, `a_post` |
| net correction rate `γ` | reconsideration toward the correct answer minus away from it | `gamma`, `gamma_post` |
| critical withholding rate `c*` | `γ / (γ + a)` | `c_star`, `c_star_post` |
| setting | one (model, benchmark, instruction) | one row of `sweep.csv`, `(run, condition)` |
| instructions A, B, C | answer honestly / no instruction / value cohesion | `A0`, `C0`, `C4` |
| shared evidence | the findings agents have put on the table so far | `share`, `{board}` in the prompts |
| gain from discussion | collective accuracy after discussion minus the round-0 majority vote | `interaction_gain` |

## Layout

```
config/
  config.template.yaml    models, benchmarks, instructions, fit settings (copy to config.yaml to override)
  data_manifest.yaml      the runs whose transcripts the paper uses
  transcripts.yaml        per-run settings for rerunning the debates
  paper_numbers.yaml      every checked claim and the sentence it comes from
workflow/
  Snakefile               entry points: figures, estimates, theory, transcripts
  rules/                  audit, theory, estimates, figures, numbers, transcripts
  scripts/harness/        debate runners and task-file builders (entry point c)
  scripts/analysis/       estimators, sweep, task counts, decomposition, audit, table diff
  scripts/theory/         mean-field boundary and agent-based simulations
  scripts/figures/        one script per figure, shared style (figstyle.py)
  scripts/numbers/        quote_numbers.py (claims → results/numbers.json)
libs/cbrm/                the estimator library of Appendix B.2
scripts/                  make_transcripts_config.py
LICENSE, LICENSE-DATA     MIT for the code, CC BY 4.0 for the data
RESULTS.md                paper item → script → rule → inputs → output; gaps; mismatches
```

Models, benchmarks and instructions are configuration: no script carries a list
of models, colours or counts. To add a model, add its runs to
`config/data_manifest.yaml` and `config/transcripts.yaml` and an entry under
`models` in the config.
