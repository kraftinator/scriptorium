#!/usr/bin/env python3
"""Grapheme-decomposition re-read (scratch experiment, not wired into src/).

Breaks the "false agreement" blind spot where BOTH models mis-read the same
hard glyph identically (pilot case: L33 surname, paper says "Gibson" but both
models read "Cissen"). Root cause: holistic word-recognition lets a top-down
language prior override the actual strokes (the "b" ascender gets read as an
archaic long-s -> phantom "ss"; capital "G" read as "C" then the rest reshaped
into a plausible surname).

Fix demonstrated here: crop JUST the name at high zoom and force the model to
describe each letter's PHYSICAL STRUCTURE before naming the word. That suppresses
the word-prior and makes the two models stop sharing the blind spot -> the false
agreement becomes a real disagreement the existing v4 debate can resolve.

Result on L33: Claude -> "Gibson" (correct), Gemini -> "Ogden" (wrong, but no
longer "Cissen"). Then v4_debate_check.py feeds Gibson vs Ogden into the real v4
debate, which resolves to "Gibson".

Run (defaults reproduce the L33 pilot case):
    .venv/bin/python experiments/decomposition/decomp_reread.py
    .venv/bin/python experiments/decomposition/decomp_reread.py --frame 23 --line 33
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import backends  # noqa: E402  (loads .env; gives model IDs + _strip_fence)

PROMPT = """You are a paleographer reading a SINGLE surname written in 1850 American Spencerian cursive.

Do NOT guess a plausible surname. Do NOT let what looks like a common name influence you. Read BOTTOM-UP, one letter at a time, by physical stroke shape only.

For EACH letter position, before you name the letter, describe its structure:
- Tall ascender loop? (b, d, f, h, k, l, and the archaic long-s all rise above x-height)
- Does that tall stroke CLOSE into a bowl on the right at the baseline (that means b or p), or does it stay a plain open loop (l, h, k, f, or an archaic long-s)?
- Descender loop below the baseline? (f, g, j, p, y)
- Or a short x-height letter? (a c e i m n o r s u v w)

CRITICAL DISCRIMINATOR: an archaic long-s (long-s) is a tall stroke that does NOT close into a bowl. A 'b' is a tall stroke that DOES close into a bowl. Do not confuse them. Also state explicitly, for every pair of ADJACENT letters, whether they are the SAME shape or DIFFERENT shapes.

Only AFTER describing every letter's structure, assemble the surname.

Return ONLY this JSON (no prose, no fence):
{"letters":[{"pos":1,"structure":"...","letter":"X"}],"same_shape_adjacent_pairs":"...","surname":"..."}"""


def crop_surname(corpus: Path, reel: str, frame: int, line: int, out: Path) -> Path:
    """Tight, high-zoom crop of the surname portion of one name cell.

    The name cell is `name_col_frac` of the page width; the surname is roughly
    its right ~55%. Cropping tight + upscaling is what gives the model the
    resolution to read stroke shapes (a full-row crop loses it)."""
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    page = corpus / "data" / "reels" / reel / f"{reel}_{frame:04d}.png"
    if not page.exists():  # fall back to the JP2 via sips
        jp2 = corpus / "data" / "reels" / reel / f"{reel}_{frame:04d}.jp2"
        subprocess.run(["sips", "-s", "format", "png", str(jp2), "--out", str(page)],
                       capture_output=True)
    img = Image.open(page)
    W, H = img.size
    top, pitch = layout["row1_top"], layout["row_pitch"]
    y0 = max(0, top + (line - 1) * pitch - 30)
    y1 = min(H, top + line * pitch + 20)
    f0, f1 = layout.get("name_col_frac", [0.15, 0.37])
    x0 = int((f0 + 0.45 * (f1 - f0)) * W)  # right ~55% of the name cell = surname
    x1 = int(f1 * W)
    crop = img.crop((x0, y0, x1, y1))
    crop = crop.resize((crop.width * 5, crop.height * 5), Image.LANCZOS)
    crop.save(out)
    return out


def run_gemini(crop: Path) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
    cfg = types.GenerateContentConfig(
        system_instruction=PROMPT, temperature=0.0,
        response_mime_type="application/json")
    contents = [
        types.Part.from_bytes(data=crop.read_bytes(), mime_type="image/png"),
        "Read the surname in this image following your system instructions.",
    ]
    r = client.models.generate_content(
        model=backends.GEMINI_MODEL, contents=contents, config=cfg)
    return backends._strip_fence((r.text or "").strip())


def run_claude(crop: Path) -> str:
    full = f"{PROMPT}\n\nRead the surname in the image at:\n{crop}"
    r = subprocess.run(
        ["claude", "-p", full, "--model", backends.CLAUDE_MODEL,
         "--allowedTools", "Read", "--strict-mcp-config"],
        capture_output=True, text=True, timeout=240)
    return backends._strip_fence(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Grapheme-decomposition re-read of one name cell.")
    ap.add_argument("--corpus", type=Path, default=REPO / "corpora" / "us_census_1850")
    ap.add_argument("--reel", default="populationschedu0604unix")
    ap.add_argument("--frame", type=int, default=23)
    ap.add_argument("--line", type=int, default=33)
    args = ap.parse_args()

    scratch = args.corpus / "output" / "rows" / args.reel / "_adj"
    scratch.mkdir(parents=True, exist_ok=True)
    crop = crop_surname(args.corpus.resolve(), args.reel, args.frame, args.line,
                        scratch / f"{args.reel}_{args.frame:04d}_decomp_L{args.line:02d}.png")
    print(f"crop: {crop}\n")

    for name, fn in (("GEMINI", run_gemini), ("CLAUDE", run_claude)):
        model = backends.GEMINI_MODEL if name == "GEMINI" else backends.CLAUDE_MODEL
        print(f"===== {name} ({model}) =====")
        try:
            out = fn(crop)
            try:
                print(json.dumps(json.loads(out), indent=2))
            except json.JSONDecodeError:
                print("(raw)", out)
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
