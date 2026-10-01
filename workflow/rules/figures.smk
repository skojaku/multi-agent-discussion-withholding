# Entry point (a): estimates -> figures (+ numbers, rules/numbers.smk).
#
# The figure scripts read every path and the model table from a JSON context
# written here from the config, so they carry no model list of their own.

FIG_DIR = j(RES_DIR, "figures")
FIGURES = ["F1", "F2", "F3", "B3", "B4"]
FIG_PDF = j(FIG_DIR, "{fig}.pdf")
FIG_LOG = j(FIG_DIR, "logs", "{fig}.txt")

# What each figure reads, inside an estimates tree `root`.
def figure_inputs(root, fig):
    runs = lambda f: [j(root, "runs", m["hp_run"], f) for m in MODELS]
    ablations = lambda f: [j(root, "runs", m["ablation"]["run"], f) for m in MODELS
                           if m.get("ablation") and m["ablation"]["kind"] == "switch"]
    decomp = lambda tag, ks: [j(root, "decomposition", f"decomposition_{k}{tag}.csv") for k in ks]
    evidence = (runs("cbrm_params.csv") + runs("hp_accuracy.csv")
                + [j(root, "model_comparison.csv")]
                + [j(TRANSCRIPTS, m["hp_run"], "events.jsonl") for m in MODELS])
    f2 = (evidence + ablations("cbrm_params.csv") + ablations("hp_accuracy.csv")
          + [j(root, "sweep.csv"), j(ITEMS, "hidden_profile.jsonl")])
    return {
        "F1": [j(root, "theory", "abm_phase.csv"), j(root, "theory", "meanfield_boundary.csv")],
        "F2": f2,
        "F3": f2 + decomp("_crossed_onestage", ("effects", "shares", "cells"))
              + decomp("", ("fitted",)),
        "B3": [j(root, "sweep.csv")],
        "B4": evidence,
    }[fig]


def context(root):
    """The paths and model table figstyle.py reads."""
    keys = ("arms", "arm_names", "conditions", "probe", "ablation_suffix",
            "c_star_std", "benchmarks", "f3_panels", "f3_arms", "models",
            "reasoning_off_suffix")
    return dict(
        out_dir=FIG_DIR,
        caption_dir=j(FIG_DIR, "captions"),
        estimates_dir=root,
        transcripts_dir=TRANSCRIPTS,
        items_dir=ITEMS,
        sweep=j(root, "sweep.csv"),
        decomposition_dir=j(root, "decomposition"),
        theory_dir=j(root, "theory"),
        model_comparison=j(root, "model_comparison.csv"),
        **{k: config[k] for k in keys},
    )


# Written when the workflow is parsed, from the config as given (including
# --config overrides), and only rewritten when the content changes.
import json as _json
write_if_changed(FIGURE_CONTEXT, _json.dumps(context(EST), indent=1) + "\n")
write_if_changed(ESTIMATES_CONTEXT, _json.dumps(context(EST_RECOMPUTED), indent=1) + "\n")


rule figure:
    input:
        script=f"{SCRIPTS}/figures/fig_{{fig}}.py",
        style=f"{SCRIPTS}/figures/figstyle.py",
        helpers=[f"{SCRIPTS}/figures/{h}.py" for h in ("evidence", "fig_F2", "fig_B3", "fig_B4")],
        context=FIGURE_CONTEXT,
        data=lambda w: figure_inputs(EST, w.fig),
        audit=AUDIT_REPORTS,
    output:
        pdf=FIG_PDF,
        png=j(FIG_DIR, "{fig}.png"),
        caption=j(FIG_DIR, "captions", "{fig}.tex"),
        log=FIG_LOG,
    wildcard_constraints:
        fig="|".join(FIGURES),
    shell:
        "FIG_CONTEXT={input.context} MPLBACKEND=Agg python {input.script} > {output.log} 2>&1"
        " || (cat {output.log}; exit 1)"


rule figures:
    """Entry point (a): every figure and results/numbers.json."""
    input:
        expand(FIG_PDF, fig=FIGURES),
        NUMBERS_JSON,
        j(RES_DIR, "numbers.ok"),
