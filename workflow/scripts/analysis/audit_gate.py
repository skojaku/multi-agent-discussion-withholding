"""Run audit_transcripts.py over a tree and turn its verdict into a gate.

The audit fails any run whose share of unparsed PUBLIC answers exceeds 2 %.
This gate fails the workflow on such a run unless it is listed in the config
under `audit_known_failures` -- runs the paper itself draws on although they
fail the audit. Those pass the gate, and the report says so in a line of its
own, so the exception is on the page rather than silent.

Usage: python audit_gate.py ROOT REPORT [KNOWN_FAILURE ...]
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(root: str, report: str, allowed: list[str]) -> int:
    r = subprocess.run([sys.executable, str(HERE / "audit_transcripts.py"), root],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    broken = re.findall(r"^(\S+)\s.*<<< BROKEN$", r.stdout, flags=re.M)
    unexpected = [b for b in broken if b not in allowed]
    lines = [out.rstrip(), ""]
    for b in broken:
        if b in allowed:
            lines.append(f"KNOWN FAILURE (allowed by config audit_known_failures): {b}"
                         " -- the paper uses this run; its rates are over a shrunken denominator.")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    if r.returncode not in (0, 1) or unexpected:
        print(out)
        print("audit failed:", ", ".join(unexpected) or f"exit {r.returncode}")
        return 1
    Path(report).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
