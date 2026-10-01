# Every number the paper quotes, beside the paper's value (config/paper_numbers.yaml).


rule numbers:
    input:
        spec="config/paper_numbers.yaml",
        script=f"{SCRIPTS}/numbers/quote_numbers.py",
        context=FIGURE_CONTEXT,
        data=[j(EST, "sweep.csv"), j(EST, "decomposition", "decomposition_fitted.csv")]
        + [j(EST, "decomposition", f"decomposition_{k}_crossed_onestage.csv")
           for k in ("effects", "shares", "cells", "fitted")]
        + [j(EST, "runs", m["hp_run"], f) for m in MODELS
           for f in ("cbrm_params.csv", "hp_accuracy.csv")],
        audit=AUDIT_REPORTS,
    output:
        json=NUMBERS_JSON,
        csv=NUMBERS_CSV,
    log:
        j(RES_DIR, "numbers.log"),
    shell:
        "FIG_CONTEXT={input.context} MPLBACKEND=Agg python {input.script}"
        " {input.spec} {output.json} {output.csv} > {log} 2>&1 || (cat {log}; exit 1);"
        " tail -1 {log}"


rule check_numbers:
    """Fail the workflow if any checked claim does not hold (numbers.json is
    kept either way, so the mismatch can be read)."""
    input:
        json=NUMBERS_JSON,
    output:
        ok=j(RES_DIR, "numbers.ok"),
    run:
        import json
        recs = json.load(open(input.json))["numbers"]
        bad = [r["key"] for r in recs if r["match"] is False]
        if bad:
            raise ValueError("claims that do not hold: " + ", ".join(bad)
                             + " (see results/numbers.json)")
        open(output.ok, "w").write("%d claims checked, all hold\n"
                                   % sum(r["match"] is not None for r in recs))
