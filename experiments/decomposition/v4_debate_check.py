#!/usr/bin/env python3
"""Feed a name disagreement into the REAL v4 debate and print the result
(scratch experiment, not wired into src/).

Companion to decomp_reread.py. Once the grapheme-decomposition re-read has
broken a false agreement into a real disagreement (pilot: Claude="Gibson",
Gemini="Ogden" on L33), this hands those two candidates to the UNCHANGED
adjudicate_name_v4 to confirm the debate lands on the correct reading.

Pilot result: Gibson vs Ogden -> "Gibson" [MEDIUM, tier1] (agrees at the nudge
step, i.e. once "Gibson" is on the ballot both models converge on it).

Run (defaults reproduce the L33 pilot case):
    .venv/bin/python experiments/decomposition/v4_debate_check.py
    .venv/bin/python experiments/decomposition/v4_debate_check.py \
        --line 33 --claude Gibson --gemini Ogden
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import adjudicate as adj  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run two candidates through the real v4 debate.")
    ap.add_argument("--corpus", type=Path, default=REPO / "corpora" / "us_census_1850")
    ap.add_argument("--reel", default="populationschedu0604unix")
    ap.add_argument("--frame", type=int, default=23)
    ap.add_argument("--line", type=int, default=33)
    ap.add_argument("--claude", default="Gibson", help="Claude's candidate (v4 Tier-3 default)")
    ap.add_argument("--gemini", default="Ogden", help="Gemini's candidate")
    ap.add_argument("--field", choices=["interpreted_last_name", "interpreted_first_name"],
                    default="interpreted_last_name")
    args = ap.parse_args()

    corpus = args.corpus.resolve()
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    reel_dir = corpus / "data" / "reels" / args.reel
    scratch = corpus / "output" / "rows" / args.reel / "_adj"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = adj.get_page_png(reel_dir, args.reel, args.frame, scratch)
    img = Image.open(page_png)
    stem = f"{args.reel}_{args.frame:04d}"

    # v4 uses the full-width row crop (crop_row), same as the real strategy.
    crop = adj.crop_row(img, layout, args.line, scratch, stem)
    print(f"crop: {crop}")
    print(f"candidates: claude={args.claude!r} gemini={args.gemini!r}\n")

    res = adj.adjudicate_name_v4(crop, adj.FIELD_LABEL[args.field], args.claude, args.gemini)
    print(json.dumps(res, indent=2))
    print(f"\n>>> v4 result: {res['value']!r}  [{res['confidence']}/{res['tier']}]")


if __name__ == "__main__":
    main()
