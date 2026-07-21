#!/usr/bin/env python3
"""s8 — each model does TWO reads per cell: original s2 prompt + forward
decomposition prompt (from s7). Claude + Gemini Flash, same crop as before.

Purpose: within-model comparison of plain vs decomposition on the same crop.
Where decomposition changes a model's answer, we learn something about which
cells decomposition helps on.

Standalone. No adjudication.

Run:
    .venv/bin/python src/strategies/s8.py
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

# --- ORIGINAL prompts (fresh copies of s2's) --------------------------------
ORIG_FIRST = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name (first name, possibly with a middle "
    "initial), then a surname. Read ONLY the given name portion (including "
    "any middle initial that is part of the given name, e.g. 'Milo J.', not "
    "just 'Milo'). Transcribe the EXACT letters as written — do NOT expand "
    "abbreviations ('Chls' stays 'Chls', not 'Charles' or 'Chas') and do NOT "
    "normalize to a more familiar spelling. Return JSON: name (the given name "
    "only), confidence (HIGH/MEDIUM/LOW). No prose outside the JSON."
)
ORIG_LAST = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name, then a surname. Read ONLY the SURNAME "
    "portion. Transcribe the EXACT letters as written — do NOT normalize to a "
    "more familiar spelling. If the surname is a ditto mark (indicating "
    "'same as the row above'), return the literal string [DITTO]. Return "
    "JSON: name (the surname only, or [DITTO]), confidence (HIGH/MEDIUM/LOW). "
    "No prose outside the JSON."
)

# --- DECOMPOSITION prompts (forward-only, fresh copies of s7's FWD) ---------
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
DECOMP_FIRST = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name (possibly with a middle initial) followed by "
    "a surname. Read ONLY the given name portion.\n\n"
    "If you try to recognize the whole name at a glance you WILL make "
    "letter-shape errors and be pulled toward familiar names. Follow this "
    "procedure strictly:\n\n"
    "STEP 1 — Describe each letter's PHYSICAL STROKE STRUCTURE starting from "
    "the LEFT and moving RIGHT. For each letter position, describe the "
    f"structure BEFORE naming the letter.\n{_STRUCTURE_GUIDE}\n\n"
    "STEP 2 — ONLY after describing every letter's structure, assemble the "
    "given name (in normal left-to-right reading order). Transcribe the "
    "EXACT letters written; do NOT expand abbreviations ('Chls' stays "
    "'Chls', not 'Charles' or 'Chas') and do NOT normalize to a more "
    "familiar spelling. Include any middle initial (e.g. 'Milo J.', not "
    "just 'Milo').\n\n"
    'Return ONLY this JSON, no other text: {"letters":[{"pos":1,'
    '"structure":"...","letter":"X"},...],"name":"<given name>",'
    '"confidence":"HIGH|MEDIUM|LOW"}'
)
DECOMP_LAST = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name followed by a surname. Read ONLY the SURNAME "
    "portion.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return ONLY: {"letters":[],"name":"[DITTO]","confidence":"HIGH"} '
    "and skip the procedure below.\n\n"
    "Otherwise, follow this procedure strictly:\n\n"
    "STEP 1 — Describe each letter's PHYSICAL STROKE STRUCTURE starting from "
    "the LEFT and moving RIGHT. For each letter position, describe the "
    f"structure BEFORE naming the letter.\n{_STRUCTURE_GUIDE}\n\n"
    "STEP 2 — ONLY after describing every letter's structure, assemble the "
    "surname (in normal left-to-right reading order). Transcribe the "
    "EXACT letters written; do NOT normalize to a more familiar spelling.\n\n"
    'Return ONLY this JSON, no other text: {"letters":[{"pos":1,'
    '"structure":"...","letter":"X"},...],"name":"<surname>",'
    '"confidence":"HIGH|MEDIUM|LOW"}'
)

# Schema tolerates the extra `letters` array (used by decomp only).
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
    scratch = Path(__file__).parent / "_s8"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s8: original + decomposition read on {len(lines)} rows of "
          f"{reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | "
          f"{'C orig':<12} {'C decomp':<12} | "
          f"{'G orig':<12} {'G decomp':<12} | truth")
    print("-" * 96)

    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        row = {"line": line, "crop": crop.name, "fields": {}}
        for field, orig_prompt, decomp_prompt, field_key in (
            ("first", ORIG_FIRST, DECOMP_FIRST, "interpreted_first_name"),
            ("last",  ORIG_LAST,  DECOMP_LAST,  "interpreted_last_name"),
        ):
            co = read_one(claude_backend, crop, orig_prompt)
            cd = read_one(claude_backend, crop, decomp_prompt)
            go = read_one(gemini_backend, crop, orig_prompt)
            gd = read_one(gemini_backend, crop, decomp_prompt)
            truth = truth_for(gt["cases"], line, field_key) or ""
            print(f"L{line:02d} {field:<5} | "
                  f"{co['name']!r:<12} {cd['name']!r:<12} | "
                  f"{go['name']!r:<12} {gd['name']!r:<12} | {truth!r}",
                  flush=True)
            row["fields"][field] = {
                "truth": truth,
                "claude":       {"original": co, "decomposition": cd},
                "gemini_flash": {"original": go, "decomposition": gd},
            }
        results.append(row)

    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "models": {"claude": backends.CLAUDE_MODEL,
                   "gemini_flash": backends.GEMINI_MODEL},
        "original_first_prompt":       ORIG_FIRST,
        "original_last_prompt":        ORIG_LAST,
        "decomposition_first_prompt":  DECOMP_FIRST,
        "decomposition_last_prompt":   DECOMP_LAST,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
