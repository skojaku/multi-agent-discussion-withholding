# Entry point (b): transcripts -> estimates. No API keys.
#
# Everything is written under results/estimates/ in the same layout as the
# released estimates in data/estimates/, so `estimates_source: recomputed` makes
# the figures read these instead, and rule `verify_estimates` diffs the two.

ITEMS_MAP = {p: b["items"] for b in config["benchmarks"].values() if b.get("items")
             for p in set(b["decomposition_prefixes"] + b["figure_prefixes"])}
_DECOMP_KEYS = ("models", "benchmarks", "arms", "reference_arm", "probe",
                 "ablation_suffix", "reasoning_off_suffix", "decomposition_fit_family",
                 "decomposition_min_contra")
write_if_changed(DECOMP_CONFIG, yaml.safe_dump({k: config[k] for k in _DECOMP_KEYS
                                                if k in config}, sort_keys=True))
CONFIG_INPUT = DECOMP_CONFIG

RUN_DIR = j(EST_RECOMPUTED, "runs", "{run}")
CBRM_PARAMS = j(RUN_DIR, "cbrm_params.csv")
HP_TABLES = {k: j(RUN_DIR, f"hp_{k}.csv") for k in ("params", "accuracy", "manipulation", "phi")}
MODEL_COMPARISON = j(EST_RECOMPUTED, "model_comparison.csv")
SWEEP = j(EST_RECOMPUTED, "sweep.csv")
DECOMP_DIR = j(EST_RECOMPUTED, "decomposition")
DECOMP_FILES = ["cells", "effects", "shares", "fitted"]
DECOMP_SINGLE = [j(DECOMP_DIR, f"decomposition_{k}.csv") for k in DECOMP_FILES]
DECOMP_CROSSED = [j(DECOMP_DIR, f"decomposition_{k}_crossed.csv") for k in DECOMP_FILES]
DECOMP_ONESTAGE = [j(DECOMP_DIR, f"decomposition_{k}_crossed_onestage.csv") for k in DECOMP_FILES]
TASK_COUNTS = j(EST_RECOMPUTED, "task_counts.csv")
RUN_SETTINGS = j(EST_RECOMPUTED, "run_settings.csv")

def _items_arg(run):
    it = items_for(run)
    return f"--items {it}" if it else ""


# ---------------------------------------------------------------- per run
rule cbrm_params:
    """B.2: posteriors of c, a, rho, r, gamma, c* and P(c > c*), per condition."""
    input:
        events=EVENTS,
        items_file=lambda w: [items_for(w.run)] if items_for(w.run) else [],
        audit=AUDIT_REPORTS,
    output:
        params=CBRM_PARAMS,
    params:
        items_arg=lambda w: _items_arg(w.run),
    shell:
        "python {SCRIPTS}/analysis/analyze_cbrm.py --events {input.events}"
        " {params.items_arg} --out-params {output.params} > /dev/null"


rule hp_tables:
    """Accuracy, pooling and manipulation-check tables of one hidden-profile run."""
    input:
        events=EVENTS,
        items_file=j(ITEMS, "hidden_profile.jsonl"),
        audit=AUDIT_REPORTS,
    output:
        **HP_TABLES,
    shell:
        "python {SCRIPTS}/analysis/analyze_hidden_profile.py --events {input.events}"
        " --items {input.items_file} --out-params {output.params}"
        " --out-accuracy {output.accuracy} --out-manipulation {output.manipulation}"
        " --out-phi {output.phi} > /dev/null"


# ---------------------------------------------------------------- tables
rule compare_models:
    """Per-model table of the main hidden-profile runs (Fig. 2 g orders by it)."""
    input:
        items_file=j(ITEMS, "hidden_profile.jsonl"),
        tables=[HP_TABLES[k].format(run=r) for r in HP_RUNS
                for k in ("params", "accuracy", "manipulation")],
        events=[EVENTS.format(run=r) for r in HP_RUNS],
    output:
        table=MODEL_COMPARISON,
    params:
        labels=[m["name"] for m in MODELS],
        dirs=[RUN_DIR.format(run=r) for r in HP_RUNS],
        events_dirs=[j(TRANSCRIPTS, r) for r in HP_RUNS],
    script:
        "../scripts/analysis/compare_models.py"


rule sweep:
    """Every (run, condition) cell of the separate-probe sweep, re-estimated."""
    input:
        events=expand(EVENTS, run=MAIN_RUNS),
        items_files=sorted({j(ITEMS, v) for v in ITEMS_MAP.values()}),
        audit=AUDIT_REPORTS,
    output:
        sweep=SWEEP,
    params:
        transcripts_dir=TRANSCRIPTS,
        items_dir=ITEMS,
        items_map=ITEMS_MAP,
        n_boot=config["sweep_n_boot"],
    script:
        "../scripts/analysis/sweep.py"


rule decomposition_single:
    """B.4, eq. decomp: model + instruction on the hidden profile benchmark."""
    input:
        sweep=SWEEP,
        config=CONFIG_INPUT,
    output:
        DECOMP_SINGLE,
    params:
        seed=config["decomposition_seed"],
    shell:
        "python {SCRIPTS}/analysis/decomposition.py --sweep {input.sweep}"
        " --config {input.config} --outdir {DECOMP_DIR} --seed {params.seed} > /dev/null"


rule decomposition_crossed:
    """B.4, eq. decomp-crossed, two-stage: fitted to logit(c_post) with a known
    standard error, over the four benchmarks; main arms only."""
    input:
        sweep=SWEEP,
        config=CONFIG_INPUT,
    output:
        DECOMP_CROSSED,
    params:
        seed=config["decomposition_seed"],
    shell:
        "python {SCRIPTS}/analysis/decomposition.py --sweep {input.sweep} --crossed"
        " --config {input.config} --outdir {DECOMP_DIR} --seed {params.seed} > /dev/null"


rule task_counts:
    """Per-task withholding counts of every sweep cell; they sum to the sweep's."""
    input:
        sweep=SWEEP,
        events=expand(EVENTS, run=MAIN_RUNS),
        audit=AUDIT_REPORTS,
    output:
        table=TASK_COUNTS,
    shell:
        "python {SCRIPTS}/analysis/task_counts.py {input.sweep} {output.table}"
        " --data-root {TRANSCRIPTS} > /dev/null"


rule decomposition_onestage:
    """B.4 and Fig. 3 (d): the crossed decomposition fitted to the per-task
    counts (binomial likelihood, task-level term per benchmark, Polya-Gamma
    Gibbs, four chains). Reasoning-off arms pooled into their model when
    `decomposition_pool_reasoning_off` is set, as in the paper."""
    input:
        sweep=SWEEP,
        task_counts=TASK_COUNTS,
        config=CONFIG_INPUT,
    output:
        DECOMP_ONESTAGE,
    params:
        seed=config["decomposition_seed"],
        chains=config["decomposition_chains"],
        reasoning="--reasoning" if config.get("decomposition_pool_reasoning_off") else "",
    threads: 4
    shell:
        "python {SCRIPTS}/analysis/decomposition.py --sweep {input.sweep} --one-stage"
        " {params.reasoning} --task-counts {input.task_counts} --tag _crossed_onestage"
        " --config {input.config} --outdir {DECOMP_DIR} --seed {params.seed}"
        " --chains {params.chains} > {DECOMP_DIR}/decomposition_onestage.log"


# ---------------------------------------------------------------- run settings
rule run_settings:
    """Every run's settings: config/transcripts.yaml (built from each run's
    events.meta.json, with the instructions read from the transcript itself)
    is the record; the columns say which fields the meta file left unset, in
    which case the harness default applied (endpoint openrouter, provider
    routing and reasoning at the provider default, max_budget 16384)."""
    input:
        runs="config/transcripts.yaml",
        meta=[j(TRANSCRIPTS, r, "events.meta.json") for r in MAIN_RUNS],
    output:
        table=RUN_SETTINGS,
    run:
        import json
        import pandas as pd
        spec = yaml.safe_load(open(input.runs))["transcripts"]
        keys = ["model", "endpoint", "provider", "reasoning_effort", "max_tokens",
                "max_budget", "timeout", "probe", "conditions", "intervention",
                "topology", "agents", "rounds", "temperature", "seed", "workers",
                "runner", "items"]
        rows = []
        for r in MAIN_RUNS:
            m = json.load(open(j(TRANSCRIPTS, r, "events.meta.json")))
            unset = [k for k in ("endpoint", "provider", "reasoning_effort", "max_budget")
                     if m.get(k) in (None, "")]
            rows.append(dict(run=r, **{k: spec[r].get(k) for k in keys},
                             meta_unset=",".join(unset)))
        pd.DataFrame(rows).to_csv(output.table, index=False)


# ---------------------------------------------------------------- verification
# What (b) must reproduce: every recomputed file against its released copy.
# Hidden-profile second-reasoning runs: F2 (e-g) reads their estimates.
ABLATION_RUNS = [m["ablation"]["run"] for m in MODELS if m.get("ablation")]
VERIFY_PAIRS = (
    [(CBRM_PARAMS.format(run=r), j(EST_RELEASED, "runs", r, "cbrm_params.csv"), "condition")
     for r in HP_RUNS]
    + [(HP_TABLES["accuracy"].format(run=r), j(EST_RELEASED, "runs", r, "hp_accuracy.csv"),
        "condition,k_informed") for r in HP_RUNS]
    + [(CBRM_PARAMS.format(run=r), j(EST_RELEASED, "runs", r, "cbrm_params.csv"), "condition")
       for r in ABLATION_RUNS]
    + [(HP_TABLES["accuracy"].format(run=r), j(EST_RELEASED, "runs", r, "hp_accuracy.csv"),
        "condition,k_informed") for r in ABLATION_RUNS]
    + [(MODEL_COMPARISON, j(EST_RELEASED, "model_comparison.csv"), "model,condition"),
       (SWEEP, j(EST_RELEASED, "sweep.csv"), "run,condition")]
    + [(f, j(EST_RELEASED, "decomposition", os.path.basename(f)), "-") for f in
       DECOMP_SINGLE + DECOMP_CROSSED + DECOMP_ONESTAGE]
    + [(TASK_COUNTS, j(EST_RELEASED, "task_counts.csv"), "run,condition,task_id")]
)
VERIFY_THEORY = [(ABM_PHASE, j(EST_RELEASED, "theory", "abm_phase.csv"), "-"),
                 (MEANFIELD_BOUNDARY, j(EST_RELEASED, "theory", "meanfield_boundary.csv"), "-")]
VERIFY_REPORT = j(RES_DIR, "verification", "estimates.csv")
VERIFY_THEORY_REPORT = j(RES_DIR, "verification", "theory.csv")


rule verify_estimates:
    """Diff every recomputed estimate against the released one (rows matched
    on the key columns; the released file may cover more rows)."""
    input:
        new=[a for a, _, _ in VERIFY_PAIRS],
        old=[b for _, b, _ in VERIFY_PAIRS],
    output:
        report=VERIFY_REPORT,
    params:
        key_cols=[k for _, _, k in VERIFY_PAIRS],
    shell:
        "python {SCRIPTS}/analysis/diff_tables.py --out {output.report}"
        " --new {input.new} --old {input.old} --keys {params.key_cols:q}"


use rule verify_estimates as verify_theory with:
    input:
        new=[a for a, _, _ in VERIFY_THEORY],
        old=[b for _, b, _ in VERIFY_THEORY],
    output:
        report=VERIFY_THEORY_REPORT,
    params:
        key_cols=[k for _, _, k in VERIFY_THEORY],


rule estimates:
    """Entry point (b). The ABM phase plane (~100 CPU-minutes) is its own target,
    `theory`, and is verified by `verify_theory`."""
    input:
        [a for a, _, _ in VERIFY_PAIRS],
        RUN_SETTINGS,
        VERIFY_REPORT,
