#!/usr/bin/env python3
"""Per-page transcription pipeline (Scriptorium engine, steps 2-4).

One page image in -> structured JSON rows out. Model-agnostic:

  Step 2  convert a JP2 source frame to JPG (cached under the corpus data/pages)
  Step 3  hand (image, prompt, schema) to the chosen backend (--agent)
  Step 4  save the returned JSON under output/rows, namespaced by agent

The model-specific code lives in backends.py; this file never needs to change
to add a new model. Everything census-specific (prompt, columns, schema) loads
from the corpus config dir, so the same pipeline runs any corpus under corpora/.

Usage:
    python src/transcribe.py --corpus corpora/us_census_1850 \
        --reel populationschedu0604unix --frame 23 --agent claude
    python src/transcribe.py ... --agent gemini

Note: for batch runs, launch detached / in a separate terminal. (The claude
backend uses --strict-mcp-config so it won't disturb channel plugins, but a
long inline run can still tie up an interactive session.)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import REGISTRY  # noqa: E402


def convert_frame(reel_dir: Path, pages_dir: Path, frame: int) -> Path:
    """Step 2: JP2 -> JPG via macOS `sips`, cached. Returns the JPG path."""
    stem = f"{reel_dir.name}_{frame:04d}"
    jp2 = reel_dir / f"{stem}.jp2"
    if not jp2.exists():
        sys.exit(f"source frame not found: {jp2}")
    pages_dir.mkdir(parents=True, exist_ok=True)
    jpg = pages_dir / f"{stem}.jpg"
    if not jpg.exists():
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(jp2), "--out", str(jpg)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return jpg


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe one census page to JSON.")
    ap.add_argument("--corpus", required=True, type=Path,
                    help="corpus dir, e.g. corpora/us_census_1850")
    ap.add_argument("--reel", required=True, help="reel identifier (folder name)")
    ap.add_argument("--frame", required=True, type=int, help="frame number, e.g. 23")
    ap.add_argument("--agent", default="claude", choices=sorted(REGISTRY),
                    help="which model backend to use (default: claude)")
    args = ap.parse_args()

    corpus = args.corpus.resolve()
    reel_dir = corpus / "data" / "reels" / args.reel
    pages_dir = corpus / "data" / "pages" / args.reel
    out_dir = corpus / "output" / "rows" / args.reel
    prompt = (corpus / "config" / "transcription_prompt.txt").read_text()
    schema = json.loads((corpus / "config" / "page_schema.json").read_text())

    jpg = convert_frame(reel_dir, pages_dir, args.frame)
    print(f"[2] converted -> {jpg}", file=sys.stderr)

    rows = REGISTRY[args.agent](jpg, prompt, schema)
    print(f"[3] {args.agent} transcribed {len(rows.get('rows', []))} rows", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.reel}_{args.frame:04d}.{args.agent}.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"[4] wrote -> {out_path}", file=sys.stderr)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
