# Entry point (c): rerun the LLM debates. Needs API keys, and costs money.
#
# One job per run in config/transcripts.yaml, each with the settings its paper
# transcript was produced with (taken from that run's events.meta.json). New
# transcripts go to results/transcripts/<tree>/<run>/events.jsonl and never
# touch data/. Outputs are protected (read-only) once written.
#
# ALWAYS run with `--rerun-triggers mtime`: under the default triggers a change
# to a script or a param deletes a finished transcript and re-buys it. A partial
# transcript resumes on (task_id, agent, round, branch) when the runner is
# called again on the same file; see README, Rerunning the LLM experiments, for how to
# do that without Snakemake deleting it.
#
# Endpoints (the `endpoint` field of an entry):
#   openrouter  OPENROUTER_API_KEY. gpt-4o-mini, gpt-5.6-luna, glm-5.3-flash.
#   vertex      Vertex AI OpenAI-compatible surface, location `global`, for
#               gemini-3.8-flash. Credentials: `gcloud auth application-default
#               login`; project: VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT.
#   openai-compatible  an OpenAI-compatible chat gateway in front of ollama, for
#               mixtral-8x22b-instruct. OPENAI_COMPATIBLE_URL (the chat-completions
#               URL) and OPENAI_COMPATIBLE_API_KEY.
#   local       a llama.cpp server (OpenAI-compatible) for Qwen3.8-27B with a
#               DFlash2 draft. LOCAL_BASE_URL: one or more comma-separated
#               /v1/chat/completions URLs.

TRANSCRIPT_RUNS = yaml.safe_load(open("config/transcripts.yaml"))
NEW_TRANSCRIPTS = j(RES_DIR, "transcripts")
NEW_EVENTS = j(NEW_TRANSCRIPTS, "{tree}", "{run}", "events.jsonl")
NEW_TARGETS = [NEW_EVENTS.format(tree=t, run=r)
               for t, runs in TRANSCRIPT_RUNS.items() for r in runs]

FLAGS = ("model", "agents", "rounds", "topology", "k", "conditions", "intervention",
         "probe", "provider", "endpoint", "base_url", "reasoning_effort", "timeout",
         "max_budget", "temperature", "seed", "max_tokens", "workers")
# llm_debate predates the per-run intervention, seed and token flags.
NOT_IN = {"llm_debate": {"intervention", "seed", "max_tokens"},
          "hiddenbench_debate": {"agents"}, "rag_debate": {"agents"}}


def debate_command(tree, run, events):
    e = dict(TRANSCRIPT_RUNS[tree][run])
    runner = e.pop("runner")
    items = e.pop("items").replace("data/items", ITEMS, 1)
    args = [f"python {SCRIPTS}/harness/{runner}.py", f"--items {items}",
            f"--events {events}"]
    for k in FLAGS:
        if k in e and k not in NOT_IN.get(runner, set()):
            v = os.path.expandvars(str(e[k]))
            args.append(f"--{k.replace('_', '-')} '{v}'")
    return " ".join(args)


rule debate:
    input:
        items_file=lambda w: TRANSCRIPT_RUNS[w.tree][w.run]["items"].replace("data/items", ITEMS, 1),
    output:
        events=protected(NEW_EVENTS),
    wildcard_constraints:
        tree="transcripts",
    params:
        cmd=lambda w, output: debate_command(w.tree, w.run, output.events),
    shell:
        "{params.cmd}"


rule audit_new_transcripts:
    input:
        events=NEW_TARGETS,
    output:
        report=j(RES_DIR, "audit", "new_transcripts.txt"),
    params:
        root=NEW_TRANSCRIPTS,
    shell:
        "python {SCRIPTS}/analysis/audit_gate.py {params.root} {output.report}"


rule transcripts:
    """Entry point (c): every paper transcript, rerun. Then point `data_dir` (or
    copy the new trees into data/transcripts*) and run `estimates`."""
    input:
        NEW_TARGETS,
        j(RES_DIR, "audit", "new_transcripts.txt"),
