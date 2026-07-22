#!/usr/bin/env python3
"""s16 — s15 (full-row context) + s13 (abbreviation hint) combined. L35 only.

s15 alone got Claude from "Mary N." → "Marv. M." — the closest Claude has ever
come to reading "Wm. H." on L35. Hypothesis: adding the abbreviation-priming
from s13 on top of s15's demographic context is what pushes Claude over the
line to "Wm.".

Full-row crop (all 14 columns). Prompt combines the column guide, demographic
inference notes, and the explicit archaic-abbreviation table. Claude + Gemini
Flash. Scope: L35 only. 4 calls.

Run:
    .venv/bin/python src/strategies/s16.py
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

_COLUMN_GUIDE = (
    "This image is ONE FULL ROW from an 1850 U.S. Census population schedule "
    "(Schedule 1: Free Inhabitants). Reading left-to-right, the columns are:\n"
    "  1. Dwelling number\n"
    "  2. Family number\n"
    "  3. NAMES of every person\n"
    "  4. Age\n"
    "  5. Sex (M or F)\n"
    "  6. Color (W, B, M)\n"
    "  7. Occupation (of males over 15)\n"
    "  8. Value of real estate\n"
    "  9. Place of birth\n"
    " 10-14. Various flags (married-in-year, school, read/write, infirmities)\n"
)

_ABBREVIATIONS = (
    "1850-era given names FREQUENTLY use archaic abbreviations. Report the "
    "ABBREVIATION as written, NOT the expanded form:\n"
    "  Wm.   = William\n"
    "  Chas. or Chls. = Charles\n"
    "  Jno.  = John\n"
    "  Geo.  = George\n"
    "  Jas.  = James\n"
    "  Saml. = Samuel\n"
    "  Robt. = Robert\n"
    "  Danl. = Daniel\n"
    "  Thos. = Thomas\n"
    "  Jos.  = Joseph\n"
    "  Benj. = Benjamin\n"
    "  Nichs. = Nicholas\n"
    "  Alex. = Alexander\n"
    "  Fredk. = Frederick"
)

FIRST_PROMPT = (
    f"{_COLUMN_GUIDE}\n"
    "Your task: read the GIVEN NAME portion of the person named in column 3 "
    "(first name plus any middle initial).\n\n"
    "Use the other visible columns as DEMOGRAPHIC CONTEXT for plausibility. "
    "Examples:\n"
    "  - Sex M + Occupation like 'Carpenter' → male-associated names "
    "('Wm.', 'Chas.', 'John', 'Henry', etc.) are far more likely. "
    "'Mary' or 'Sarah' would be very implausible.\n"
    "  - Sex F + no occupation → female-associated names ('Mary', "
    "'Elizabeth', 'Sarah', 'Julia', 'Ann', etc.) are more likely.\n\n"
    f"{_ABBREVIATIONS}\n\n"
    "Read the strokes actually on the page (do not invent a name), but also "
    "check that your read is CONSISTENT with the demographic context and "
    "the abbreviation patterns above. If the demographic context says male "
    "and the strokes fit an abbreviation like 'Wm.' better than a female "
    "name, report the abbreviation.\n\n"
    "Include any middle initial (e.g. 'Wm. H.', 'Mary A.', 'Milo J.').\n\n"
    'Return ONLY this JSON: {"name":"<given name>","confidence":'
    '"HIGH|MEDIUM|LOW"}. No prose outside.'
)

LAST_PROMPT = (
    f"{_COLUMN_GUIDE}\n"
    "Your task: read the SURNAME portion of the person named in column 3.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    "return the literal string [DITTO].\n\n"
    "Otherwise transcribe the EXACT letters as written; do NOT normalize to "
    "a more familiar spelling.\n\n"
    'Return ONLY this JSON: {"name":"<surname or [DITTO]>","confidence":'
    '"HIGH|MEDIUM|LOW"}. No prose outside.'
)

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "confidence"]}

Y_PAD = 40
UPSCALE = 2
TARGET_LINES = [35]


def crop_full_row(img: Image.Image, top: int, pitch: int, line: int,
                  dst: Path) -> Path:
    W, H = img.size
    y0 = max(0, top + (line - 1) * pitch - Y_PAD)
    y1 = min(H, top + line * pitch + Y_PAD)
    row = img.crop((0, y0, W, y1))
    row = row.resize((row.width * UPSCALE, row.height * UPSCALE), Image.LANCZOS)
    row.save(dst)
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


def read_one(backend, crop: Path, prompt: str) -> tuple[str, str]:
    try:
        r = backend(crop, prompt, SCHEMA)
        return r.get("name", "(missing)"), r.get("confidence", "?")
    except Exception as e:
        return f"ERR: {type(e).__name__}: {str(e)[:60]}", "?"


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s16"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s16: s15 full-row context + s13 abbreviation hint on {TARGET_LINES} of "
          f"{reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | {'Claude':<15} | {'Gemini':<15} | truth")
    print("-" * 68)

    results = []
    for line in TARGET_LINES:
        crop = crop_full_row(img, top, pitch, line, scratch / f"L{line:02d}.png")
        row = {"line": line, "crop": crop.name, "fields": {}}
        for field, prompt, field_key in (
            ("first", FIRST_PROMPT, "interpreted_first_name"),
            ("last",  LAST_PROMPT,  "interpreted_last_name"),
        ):
            cn, cc = read_one(claude_backend, crop, prompt)
            gn, gc = read_one(gemini_backend, crop, prompt)
            truth = truth_for(gt["cases"], line, field_key) or ""
            print(f"L{line:02d} {field:<5} | {cn!r:<15} | {gn!r:<15} | {truth!r}",
                  flush=True)
            row["fields"][field] = {
                "truth": truth,
                "claude": {"name": cn, "confidence": cc},
                "gemini_flash": {"name": gn, "confidence": gc},
            }
        results.append(row)

    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "target_lines": TARGET_LINES,
        "models": {"claude": backends.CLAUDE_MODEL,
                   "gemini_flash": backends.GEMINI_MODEL},
        "first_prompt": FIRST_PROMPT, "last_prompt": LAST_PROMPT,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
