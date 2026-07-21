#!/usr/bin/env python3
"""s7 — split-name read with grapheme decomposition, BOTH forward + backward.

Same tight cell crop and split-first/last shape as s2. Two models (Claude +
Gemini Flash). For each name cell we run TWO reads per model:

  forward  — describe each letter's physical structure LEFT→RIGHT, then name
  backward — describe each letter's physical structure RIGHT→LEFT, then name

The forward form is the proven fix for the L33 shared-blind-spot case
(experiments/decomposition/): forcing bottom-up letter-shape description
suppresses the top-down word prior. The backward form tests whether starting
from the trailing letters (which are less predictive of a specific word)
disrupts the leading-letter anchoring even more.

No adjudication; observational. Save results.json for the viewer.

Run:
    .venv/bin/python src/strategies/s7.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from backends import claude_backend, gemini_backend  # noqa: E402
import backends  # noqa: E402

_STRUCTURE_GUIDE = (
    "For each position note:\n"
    "  • Is there a tall ascender (a stroke rising above x-height)?\n"
    "  • If yes, does that tall stroke CLOSE into a bowl at the baseline (→ 'b', 'p') "
    "or stay open (→ 'l', 'h', 'k', 'f', or the archaic long-s ſ)?\n"
    "  • Is there a descender (loop below the baseline: 'f', 'g', 'j', 'p', 'y')?\n"
    "  • Or a plain x-height letter (a, c, e, i, m, n, o, r, s, u, v, w)?\n"
    "  • Note dots above (i, j), capital forms, punctuation (periods, apostrophes), "
    "and whether two adjacent letters share the SAME shape or DIFFERENT shapes.\n"
    "  • CRITICAL: an archaic long-s (ſ) is a tall stroke that does NOT close "
    "into a bowl; a 'b' IS a tall stroke that DOES close into a baseline bowl. "
    "Do not confuse them."
)


def _first_prompt(direction: str) -> str:
    order, arrow = (("LEFT and moving RIGHT", "left→right")
                    if direction == "forward"
                    else ("RIGHT and moving LEFT", "right→left"))
    return (
        "This image is the NAME cell of one row on an 1850 U.S. Census page — "
        "a handwritten given name (possibly with a middle initial) followed by "
        "a surname. Read ONLY the given name portion.\n\n"
        "If you try to recognize the whole name at a glance you WILL make "
        "letter-shape errors and be pulled toward familiar names. Follow this "
        "procedure strictly:\n\n"
        f"STEP 1 — Describe each letter's PHYSICAL STROKE STRUCTURE starting "
        f"from the {order} ({arrow}). For each letter position, describe the "
        f"structure BEFORE naming the letter.\n{_STRUCTURE_GUIDE}\n\n"
        "STEP 2 — ONLY after describing every letter's structure, assemble "
        "the given name (in normal left-to-right reading order). Transcribe "
        "the EXACT letters written; do NOT expand abbreviations ('Chls' stays "
        "'Chls', not 'Charles' or 'Chas') and do NOT normalize to a more "
        "familiar spelling. Include any middle initial (e.g. 'Milo J.', not "
        "just 'Milo').\n\n"
        'Return ONLY this JSON, no other text: {"letters":[{"pos":1,'
        '"structure":"...","letter":"X"},...],"name":"<given name>",'
        '"confidence":"HIGH|MEDIUM|LOW"}'
    )


def _last_prompt(direction: str) -> str:
    order, arrow = (("LEFT and moving RIGHT", "left→right")
                    if direction == "forward"
                    else ("RIGHT and moving LEFT", "right→left"))
    return (
        "This image is the NAME cell of one row on an 1850 U.S. Census page — "
        "a handwritten given name followed by a surname. Read ONLY the SURNAME "
        "portion.\n\n"
        "If the surname is a ditto mark (indicating 'same as the row above'), "
        'return ONLY: {"letters":[],"name":"[DITTO]","confidence":"HIGH"} '
        "and skip the procedure below.\n\n"
        "Otherwise, follow this procedure strictly:\n\n"
        f"STEP 1 — Describe each letter's PHYSICAL STROKE STRUCTURE starting "
        f"from the {order} ({arrow}). For each letter position, describe the "
        f"structure BEFORE naming the letter.\n{_STRUCTURE_GUIDE}\n\n"
        "STEP 2 — ONLY after describing every letter's structure, assemble "
        "the surname (in normal left-to-right reading order). Transcribe the "
        "EXACT letters written; do NOT normalize to a more familiar spelling.\n\n"
        'Return ONLY this JSON, no other text: {"letters":[{"pos":1,'
        '"structure":"...","letter":"X"},...],"name":"<surname>",'
        '"confidence":"HIGH|MEDIUM|LOW"}'
    )


FIRST_FWD = _first_prompt("forward")
FIRST_BWD = _first_prompt("backward")
LAST_FWD  = _last_prompt("forward")
LAST_BWD  = _last_prompt("backward")

# The `letters` array is optional in the schema — we only score `name`.
SCHEMA = {"type": "object", "properties": {
    "letters": {"type": "array", "items": {"type": "object", "properties": {
        "pos": {"type": "integer"},
        "structure": {"type": "string"},
        "letter": {"type": "string"}}}},
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "confidence"]}

NAME_COL_FRAC = (0.15, 0.37)
Y_PAD = 40
UPSCALE = 2


def crop_name_cell(img: Image.Image, top: int, pitch: int, line: int, dst: Path) -> Path:
    W, H = img.size
    y0 = max(0, top + (line - 1) * pitch - Y_PAD)
    y1 = min(H, top + line * pitch + Y_PAD)
    x0, x1 = int(NAME_COL_FRAC[0] * W), int(NAME_COL_FRAC[1] * W)
    cell = img.crop((x0, y0, x1, y1))
    cell = cell.resize((cell.width * UPSCALE, cell.height * UPSCALE), Image.LANCZOS)
    cell.save(dst)
    return dst


def get_page_png(reel_dir: Path, reel: str, frame: int, scratch: Path) -> Path:
    png = reel_dir / f"{reel}_{frame:04d}.png"
    if png.exists():
        return png
    cached = REPO / "corpora" / "us_census_1850" / "data" / "pages" / reel / f"{reel}_{frame:04d}.png"
    if cached.exists():
        return cached
    jp2 = reel_dir / f"{reel}_{frame:04d}.jp2"
    out = scratch / f"{reel}_{frame:04d}.png"
    if not out.exists():
        cmd = (["sips", "-s", "format", "png", str(jp2), "--out", str(out)]
               if sys.platform == "darwin"
               else ["convert", str(jp2), str(out)])
        subprocess.run(cmd, capture_output=True, check=True)
    return out


def truth_for(cases, line, field):
    for c in cases:
        if c["line"] == line and c["field"] == field:
            return c["correct"]
    return None


def read_one(backend, crop: Path, prompt: str) -> dict:
    try:
        r = backend(crop, prompt, SCHEMA)
        return {"name": r.get("name", "(missing)"),
                "confidence": r.get("confidence", "?"),
                "letters": r.get("letters") or []}
    except Exception as e:
        return {"name": f"ERR: {type(e).__name__}: {str(e)[:60]}",
                "confidence": "?", "letters": []}


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s7"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s7: decomposition read (fwd + bwd) on {len(lines)} rows of "
          f"{reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | "
          f"{'C fwd':<12} {'C bwd':<12} | {'G fwd':<12} {'G bwd':<12} | truth")
    print("-" * 96)

    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        row = {"line": line, "crop": crop.name, "fields": {}}
        for field, fwd_prompt, bwd_prompt, field_key in (
            ("first", FIRST_FWD, FIRST_BWD, "interpreted_first_name"),
            ("last",  LAST_FWD,  LAST_BWD,  "interpreted_last_name"),
        ):
            cf = read_one(claude_backend, crop, fwd_prompt)
            cb = read_one(claude_backend, crop, bwd_prompt)
            gf = read_one(gemini_backend, crop, fwd_prompt)
            gb = read_one(gemini_backend, crop, bwd_prompt)
            truth = truth_for(gt["cases"], line, field_key) or ""
            print(f"L{line:02d} {field:<5} | "
                  f"{cf['name']!r:<12} {cb['name']!r:<12} | "
                  f"{gf['name']!r:<12} {gb['name']!r:<12} | {truth!r}",
                  flush=True)
            row["fields"][field] = {
                "truth": truth,
                "claude":       {"forward": cf, "backward": cb},
                "gemini_flash": {"forward": gf, "backward": gb},
            }
        results.append(row)

    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "models": {"claude": backends.CLAUDE_MODEL,
                   "gemini_flash": backends.GEMINI_MODEL},
        "first_forward_prompt":  FIRST_FWD,
        "first_backward_prompt": FIRST_BWD,
        "last_forward_prompt":   LAST_FWD,
        "last_backward_prompt":  LAST_BWD,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
