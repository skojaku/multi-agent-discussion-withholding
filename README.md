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

Teams of LLM agents discuss a task over three rounds. In each round an agent
states an answer in public and, in a separate call, gives its private answer;
it **withholds** when it states the majority's answer while privately
disagreeing. This repository regenerates every figure and checks every
quantitative claim of the paper from the logs of those discussions: it
estimates the withholding rate `c`, the internalization rate `a`, the net
correction rate `γ` and the critical rate `c* = γ / (γ + a)` (Appendix B.2).

## Where things are

| What | Where |
|---|---|
| Logs: every agent's public and private answer, per round | `data/transcripts/<run>/events.jsonl` in the data archive (Zenodo, DOI to be added here) |
| Settings of each run | `data/transcripts/<run>/events.meta.json`, `config/transcripts.yaml` |
| Task files | `data/items/*.jsonl` |
| Released estimates | `data/estimates/` |
| Figures (PDF, 400-dpi PNG, captions) | `results/figures/{F1,F2,F3,B3,B4}.pdf`, written by the workflow |
| Checked claims, with the value computed here | `results/numbers.json`, listed in `config/paper_numbers.yaml` |
| Each paper item → script → rule → inputs → output | `RESULTS.md` |
| Workflow and scripts; estimator library (Appendix B.2) | `workflow/` (`Snakefile`, `rules/`, `scripts/`); `libs/cbrm/` |
| Configuration: models, benchmarks, fit settings; runs in the archive | `config/config.template.yaml` (copy to `config.yaml` to override); `config/data_manifest.yaml` |

`data/` is not in git. Unpack the archive at the repository root. Its layout:

```
data/transcripts/<run>/events.jsonl      one line per agent and round: task_id, agent, round,
                                         branch (the instruction; r0 for the independent round 0),
                                         private, public, share (the finding put on the table), raw reply
data/transcripts/<run>/events.meta.json  model, endpoint, reasoning effort, token budget, probe,
                                         instructions, temperature, seed
data/items/*.jsonl                       task files
data/estimates/...                       released estimates (same layout as results/estimates/)
data/MANIFEST.json                       sha256 of every file
```

`config/data_manifest.yaml` lists the runs in the archive. One MedEInst run
(`base5_qwen`) is excluded from the data, figures and fit, because 3.3 % of its
rows have no public answer, above the 2 % limit of the transcript audit. Code
is MIT (`LICENSE`), data CC BY 4.0 (`LICENSE-DATA`); the task files come from
HiddenBench, MuSiQue and MedEInst, which keep their own licenses.

## Reproduce

Requirements: Linux or macOS, Python 3.12, [uv](https://docs.astral.sh/uv/),
about 400 MB for the data. All dependencies are pinned in `uv.lock`.

```bash
git clone https://github.com/skojaku/multi-agent-discussion-withholding && cd multi-agent-discussion-withholding
uv sync
# put the archive's data/ directory at ./data
uv run snakemake --rerun-triggers mtime -c4 figures
```

After a few minutes `results/figures/` holds the figures and
`results/numbers.json` the checked claims. **Always pass
`--rerun-triggers mtime`**: with Snakemake's default triggers, editing a script
marks finished outputs stale and deletes them, and for a transcript that
deletes a paid run.

| Target (`uv run snakemake --rerun-triggers mtime -c4 <target>`) | Produces | Time | Needs |
|---|---|---|---|
| `figures` (or one file, e.g. `results/figures/F2.pdf`) | Figs. 1, 2, 3, B3, B4 and `numbers.json`, from the released estimates | about 3 min | data archive |
| `estimates` | per-setting posteriors, sweep, decomposition, run settings, and `results/verification/estimates.csv`, which compares each file with the released one | about 30 min, 4–12 cores | data archive |
| `theory verify_theory` | the phase-plane tables behind Fig. 1 c, compared with the released ones | about 100 CPU-minutes | nothing |
| `transcripts` | a rerun of every LLM discussion (below) | days, API keys, money | keys |

Every estimator is seeded, so reruns give identical files. To draw the figures
from recomputed estimates, add `--config estimates_source=recomputed`. A figure
that differs from the paper's PDF by a few pixels is anti-aliasing (Fig. 1 was
rendered with matplotlib 3.10.6, the others with the pinned 3.11.2).

Every estimate, figure and claim depends on a transcript audit
(`results/audit/transcripts.txt`) that fails any run with more than 2 % of its
public answers unparsed, since its rates would be computed over a silently
shrunken denominator. If `figures` fails at `check_numbers`, a claim of the
paper does not hold on the estimates used; `results/numbers.json` says which.

## Rerunning the LLM experiments

```bash
uv run snakemake --rerun-triggers mtime -c1 -n -p transcripts   # dry run: prints every command
uv run snakemake --rerun-triggers mtime -c1 results/transcripts/transcripts/hp_models/luna/events.jsonl
```

One job per run in `config/transcripts.yaml`, which records each run's model
id, endpoint, reasoning effort, completion budget, probe, instructions,
temperature, seed (where set) and parallel workers. New transcripts go to
`results/transcripts/`, never overwrite `data/`, and are write-protected and
audited when finished. To estimate from them, copy them over
`data/transcripts/` and run `estimates`. A partial `events.jsonl` resumes on
`(task_id, agent, round, branch)` when its runner is called again; take the
command from the dry run and run it directly, because Snakemake would delete
the partial file.

| `endpoint` | Models | Needs |
|---|---|---|
| `openrouter` | gpt-4o-mini, gpt-5.6-luna, glm-5.3-flash, nemotron-3-super-120b-a12b, ling-3.0-flash, seed-2.0-mini | `OPENROUTER_API_KEY` |
| `vertex` | gemini-3.8-flash (Vertex AI, OpenAI-compatible, location `global`) | `gcloud auth application-default login`, `VERTEX_PROJECT` or `GOOGLE_CLOUD_PROJECT` |
| `openai-compatible` | mixtral-8x22b-instruct (ollama tag `mixtral:8x22b-instruct`) | `OPENAI_COMPATIBLE_URL`, `OPENAI_COMPATIBLE_API_KEY` |
| `local` | Qwen3.8-27B, `Qwen3.8-27B-UD-Q8_K_L.gguf` with draft `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, llama.cpp `llama-server --spec-type draft-dflash --spec-draft-n-max 7 -ngl 999 -ngld 999 --ctx-size 65536 -np 8 --no-kv-unified --flash-attn on --jinja` | `LOCAL_BASE_URL` (comma-separated `/v1/chat/completions` URLs) |

A model's main run feeds every figure. Its second run (`*_nothink`) feeds
Fig. 2 e–g, 3 a–c and B3 and is pooled with the main run in the decomposition;
it means reasoning off, except for gemini-3.8-flash (`low`, the lowest its API
accepts). The `*_highthink` run of glm-5.3-flash is in no figure or fit. Not
recorded, and so not pinned: the OpenRouter provider a call was routed to,
`top_p` (provider default), most seeds (temperature is 0.7 throughout), and the
llama.cpp and ollama versions. A field missing from an older meta file means the
harness default (`openrouter`, provider-default reasoning, `max_budget` 16384).

Cost: the paper's transcripts hold about 670,000 model calls, roughly 0.7
billion input tokens plus hidden reasoning tokens, so the input side alone is
about US$100–400 at US$0.1–0.5 per million tokens. One run takes hours at 8–16
parallel workers. Prices were not logged. Rerun one run first.

## Terms

| Paper | In code and data files |
|---|---|
| agent; statement (public answer); private answer | `agent` (a few variables say `seat`); `public`; `private` |
| withhold, withholding rate `c` (opportunities to dissent: `n_contra`) | `conceal`, `n_conceal`, `c`, `c_post` |
| internalization `a`; net correction `γ`; critical rate `c*` | `a`, `gamma`, `c_star` (`_post` for posterior means) |
| setting (one model, benchmark and instruction) | one row of `sweep.csv`, `(run, condition)` |
| instructions A, B, C (answer honestly; none; value cohesion) | `A0`, `C0`, `C4` |
| shared evidence; gain from discussion | `share`, `{board}` in prompts; `interaction_gain` |
