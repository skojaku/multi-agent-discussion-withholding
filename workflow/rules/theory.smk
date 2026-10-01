# Theory: the simulated phase plane and mean-field boundary of Fig. 1 (c). No model calls; deterministic (seeded).

THEORY_OUT = j(EST_RECOMPUTED, "theory")
ABM_PHASE = j(THEORY_OUT, "abm_phase.csv")
ABM_PHASE_CHUNK = j(RES_DIR, "tmp", "abm_phase", "N{N}_chunk{chunk}.csv")
MEANFIELD_BOUNDARY = j(THEORY_OUT, "meanfield_boundary.csv")
ABM_N = [str(n) for n in config["abm_phase_N"]]
ABM_CHUNKS = int(config["abm_phase_chunks"])


rule abm_phase_chunk:
    output:
        part=temp(ABM_PHASE_CHUNK),
    params:
        n_chunks=ABM_CHUNKS,
    shell:
        "python {SCRIPTS}/theory/run_abm.py {output.part} --N {wildcards.N}"
        " --chunk {wildcards.chunk} --n-chunks {params.n_chunks}"


rule abm_phase:
    input:
        # In the order the one-pass table has them: N, then withholding rate.
        parts=[ABM_PHASE_CHUNK.format(N=n, chunk=c) for n in ABM_N
               for c in range(ABM_CHUNKS)],
    output:
        table=ABM_PHASE,
    shell:
        "python {SCRIPTS}/theory/run_abm.py {output.table} --merge {input.parts}"


rule meanfield_boundary:
    output:
        table=MEANFIELD_BOUNDARY,
    shell:
        "python {SCRIPTS}/theory/run_meanfield.py {output.table}"


rule theory:
    input:
        ABM_PHASE, MEANFIELD_BOUNDARY,
