# The transcript audit: every run whose share of unparsed
# PUBLIC answers exceeds 2 % fails the workflow, so no number is quoted from a
# transcript whose denominators silently shrank. Every rule that reads a
# transcript or quotes a number depends on these reports.

AUDIT_MAIN = j(RES_DIR, "audit", "transcripts.txt")
AUDIT_REPORTS = [AUDIT_MAIN] if config.get("audit", True) else []


rule audit_transcripts:
    input:
        events=expand(EVENTS, run=MAIN_RUNS),
    output:
        report=AUDIT_MAIN,
    params:
        root=TRANSCRIPTS,
        allowed=config.get("audit_known_failures", []),
    shell:
        "python {SCRIPTS}/analysis/audit_gate.py {params.root} {output.report} {params.allowed}"
