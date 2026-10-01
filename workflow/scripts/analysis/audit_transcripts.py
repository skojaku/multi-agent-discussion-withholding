"""Is this transcript measurable, or did the harness spend the money for nothing?

A debate row carries a PRIVATE and a PUBLIC answer, and concealment is the rate
at which they differ. A row whose PUBLIC line was never parsed is not a row where
the agent agreed with itself -- it is a row with no measurement in it, and every
rate computed over the run is computed over a denominator that quietly shrank.
Three ways that happened here, each of which finished at full cost with nothing
on the way that said so:

  whitespace accepted   `chat()` tested `if text:`, and " " passes it. A
                        reasoning model that spends its whole budget on hidden
                        tokens returns "" from some OpenRouter providers and " "
                        from others, so a quarter of deepseek-v4-flash-0731's
                        rows had no answer lines. Fixed in llm_debate.Client; the
                        check here is what would have caught it.
  wrong runner          MuSiQue answers are free-form spans and its items carry
                        `choices: []`. Run through `hiddenbench_debate.py`, whose
                        parser matches an option in the reply, all 6,000 rows
                        came back with private=None and public=None.
  budget too small      a model truncated mid-reply writes PRIVATE and stops, so
                        PUBLIC is missing while `raw` is non-empty. Around 0.2%
                        for gpt-5.6-luna at 256 tokens, which is the noise floor;
                        anything above about 2% is a broken run, not noise.

Run it over a data root before running the analysis, and over a single run before
quoting a number from it:

    python3 workflow/scripts/audit_transcripts.py data
    python3 workflow/scripts/audit_transcripts.py data/hp_models/luna --verbose

Exit status is 1 if any run is over the threshold, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROBE_SEP = "\n---\n"
# Above this share of unparsed PUBLIC lines a run is broken rather than noisy.
# The observed split is stark: every healthy run in this experiment sits under
# 1%, and every broken one was over 10%.
THRESHOLD = 0.02


def audit_one(path: Path) -> dict | None:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    debate = [r for r in rows if r.get("round", 0) >= 1]
    if not debate:
        return None

    def empty(key):
        return sum(not str(r.get(key) or "").strip() for r in debate)

    # The debate reply is everything before the separate probe's answer, so a
    # blank one means the debate call itself returned nothing usable.
    blank_raw = sum(1 for r in debate
                    if not (r.get("raw") or "").split(PROBE_SEP)[0].strip())
    blank_probe = sum(1 for r in debate
                      if PROBE_SEP in (r.get("raw") or "")
                      and not r["raw"].split(PROBE_SEP, 1)[1].strip())
    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    models = sorted({str(r.get("model")) for r in debate if r.get("model")})
    return dict(rows=len(rows), debate=len(debate),
                private_empty=empty("private"), public_empty=empty("public"),
                blank_raw=blank_raw, blank_probe=blank_probe,
                probe=meta.get("probe", "unknown"),
                provider=meta.get("provider") or "-",
                max_tokens=meta.get("max_tokens", "?"),
                model="|".join(m.split("/")[-1] for m in models))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a data directory, or one run directory")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="share of unparsed PUBLIC lines that fails a run")
    ap.add_argument("--verbose", action="store_true",
                    help="print a few of the offending raw replies")
    args = ap.parse_args(argv)

    root = Path(args.root)
    events = ([root / "events.jsonl"] if (root / "events.jsonl").exists()
              else sorted(root.glob("**/events.jsonl")))
    events = [p for p in events if "external" not in p.parts and "items" not in p.parts]
    if not events:
        raise SystemExit(f"no events.jsonl under {root}")

    print(f"{'run':44s} {'debate':>7} {'pub_empty':>10} {'%':>6} "
          f"{'blank':>6} {'probe':>9} {'mt':>5}  model / provider")
    failed = []
    for p in events:
        a = audit_one(p)
        if a is None:
            continue
        try:
            name = str(p.parent.relative_to(root))
        except ValueError:
            name = p.parent.name
        share = a["public_empty"] / a["debate"]
        mark = ""
        if share > args.threshold:
            failed.append((name, share))
            mark = "  <<< BROKEN"
        print(f"{name:44s} {a['debate']:7d} {a['public_empty']:10d} "
              f"{100 * share:5.1f}% {a['blank_raw']:6d} {a['probe']:>9s} "
              f"{str(a['max_tokens']):>5s}  {a['model']} / {a['provider']}{mark}")
        if args.verbose and a["public_empty"]:
            shown = 0
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("round", 0) >= 1 and not str(r.get("public") or "").strip():
                    print(f"      {(r.get('raw') or '')!r:.150}")
                    shown += 1
                    if shown >= 3:
                        break

    if failed:
        print(f"\n{len(failed)} run(s) over {100 * args.threshold:.0f}% unparsed "
              f"PUBLIC. Their withholding rates are computed over a denominator "
              f"that silently shrank; do not quote them.")
        for name, share in failed:
            print(f"  {name}  {100 * share:.1f}%")
        return 1
    print("\nevery run under threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
