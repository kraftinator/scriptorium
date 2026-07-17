#!/usr/bin/env python3
"""Per-page transcription pipeline (Scriptorium engine, steps 2-4).

One page image in -> structured JSON rows out. Model-agnostic and method-agnostic:

  Step 2  convert a JP2 source frame to a working image (cached under data/pages)
  Step 3  transcribe with the chosen backend (--agent), either:
            --mode whole  : one call on the full page
            --mode tiled  : slice into row-bands, transcribe each, stitch
                            (line numbers assigned by geometry — see tile.py)
  Step 4  save the returned JSON under output/rows, namespaced by agent

Model-specific code lives in backends.py; tiling lives in tile.py. This file
just orchestrates. Everything corpus-specific loads from the corpus config.

Usage:
    python src/transcribe.py --corpus corpora/us_census_1850 \
        --reel populationschedu0604unix --frame 23 --agent gemini --mode tiled
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import REGISTRY  # noqa: E402
from tile import transcribe_tiled  # noqa: E402


def convert_frame(reel_dir: Path, pages_dir: Path, frame: int, fmt: str) -> Path:
    """Step 2: JP2 -> JPG/PNG (macOS `sips`, Linux `convert`), cached. Returns the image path."""
    stem = f"{reel_dir.name}_{frame:04d}"
    jp2 = reel_dir / f"{stem}.jp2"
    if not jp2.exists():
        sys.exit(f"source frame not found: {jp2}")
    pages_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if fmt == "png" else "jpg"
    out = pages_dir / f"{stem}.{ext}"
    if not out.exists():
        if sys.platform == "darwin":
            cmd = ["sips", "-s", "format", fmt, str(jp2), "--out", str(out)]
        else:
            cmd = ["convert", str(jp2), str(out)]
        subprocess.run(
            cmd,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe one census page to JSON.")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--reel", required=True)
    ap.add_argument("--frame", required=True, type=int)
    ap.add_argument("--agent", default="claude", choices=sorted(REGISTRY))
    ap.add_argument("--mode", default="whole", choices=["whole", "tiled"],
                    help="whole = one full-page call; tiled = row-band tiling")
    args = ap.parse_args()

    corpus = args.corpus.resolve()
    reel_dir = corpus / "data" / "reels" / args.reel
    pages_dir = corpus / "data" / "pages" / args.reel
    out_dir = corpus / "output" / "rows" / args.reel

    # tiled crops from a lossless PNG; whole-page uses a JPG (smaller).
    fmt = "png" if args.mode == "tiled" else "jpeg"
    page_img = convert_frame(reel_dir, pages_dir, args.frame, fmt)
    print(f"[2] converted -> {page_img}", file=sys.stderr)

    if args.mode == "tiled":
        rows = transcribe_tiled(page_img, args.agent, corpus, pages_dir / "_tiles")
    else:
        prompt = (corpus / "config" / "transcription_prompt.txt").read_text()
        schema = json.loads((corpus / "config" / "page_schema.json").read_text())
        rows = REGISTRY[args.agent](page_img, prompt, schema)
    print(f"[3] {args.agent}/{args.mode} -> {len(rows.get('rows', []))} rows",
          file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.reel}_{args.frame:04d}.{args.agent}.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"[4] wrote -> {out_path}", file=sys.stderr)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
