#!/usr/bin/env python3
"""End-to-end pipeline for one page: transcribe with each agent, then adjudicate.

A single command runs the whole flow in sequence:

    transcribe (claude, tiled)  ->  transcribe (gemini, tiled)  ->  adjudicate

so one call produces a finished, cross-model-reconciled page instead of running
three commands by hand. Each step is the existing standalone script, invoked as
a subprocess; run them individually if you need finer control. adjudicate reads
the two per-agent files directly, so reconcile.py is not part of this chain (it
remains a fast, model-free "how much do they disagree" check).

Usage:
    GEMINI_API_KEY=... python src/pipeline.py --corpus corpora/us_census_1850 \
        --reel populationschedu0604unix --frame 23
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, label):
    print(f"\n[pipeline] === {label} ===", file=sys.stderr, flush=True)
    result = subprocess.run(cmd)  # inherit stdout/stderr so child progress streams
    if result.returncode != 0:
        sys.exit(f"[pipeline] step failed: {label} (exit {result.returncode}); aborting.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full per-page pipeline (transcribe x agents -> adjudicate).")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--reel", required=True)
    ap.add_argument("--frame", required=True, type=int)
    ap.add_argument("--mode", default="tiled", choices=["whole", "tiled"])
    ap.add_argument("--agents", nargs="+", default=["claude", "gemini"])
    args = ap.parse_args()

    py = sys.executable
    base = ["--corpus", args.corpus, "--reel", args.reel, "--frame", str(args.frame)]

    for agent in args.agents:
        run([py, str(HERE / "transcribe.py"), *base, "--agent", agent, "--mode", args.mode],
            f"transcribe: {agent} ({args.mode})")

    run([py, str(HERE / "adjudicate.py"), *base, "--agents", *args.agents],
        f"adjudicate: {' + '.join(args.agents)}")

    out = (Path(args.corpus).resolve() / "output" / "rows" / args.reel
           / f"{args.reel}_{args.frame:04d}.adjudicated.json")
    print(f"\n[pipeline] === done -> {out} ===", file=sys.stderr)


if __name__ == "__main__":
    main()
