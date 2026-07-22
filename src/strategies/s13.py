#!/usr/bin/env python3
"""s13 — archaic-abbreviation hint added to the prompt, L35 only.

The stubborn L35 first name is "Wm. H." (William H.), consistently misread by
both models as "Mary [x]". Hypothesis: the models don't recognize "Wm." as a
period abbreviation and instead pattern-match to a familiar modern first name.

Fix: prime the prompt with the common 1850-era abbreviations (Wm., Chas., Jno.,
Geo., etc.) so those readings are candidates the model considers.

Same tight cell crop as s2 (no wider-crop change — this is a prompt-only
intervention). Claude + Gemini Flash. Scope: L35 only. 4 calls total.

Run:
    .venv/bin/python src/strategies/s13.py
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

_ARCHAIC_ABBREVIATIONS = (
    "Important context — 1850 U.S. Census records frequently write given "
    "names as archaic abbreviations, and you must transcribe them AS "
    "abbreviations (not the expanded form). Common ones include:\n"
    "  Wm.   = William\n"
    "  Chas. or Chls. = Charles\n"
    "  Jno.  = John\n"
    "  Geo.  = George\n"
    "  Jas.  = James\n"
    "  Saml. = Samuel\n"
    "  Robt. = Robert\n"
    "  Danl. = Daniel\n"
    "  Thos. = Thomas\n"
    "  Nichs. = Nicholas\n"
    "  Benj. = Benjamin\n"
    "  Alex. = Alexander\n"
    "  Jos.  = Joseph\n"
    "  Fredk. = Frederick\n"
    "If the strokes match one of these patterns, report the abbreviation "
    "(e.g. 'Wm.', not 'William')."
)

FIRST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "it contains a handwritten given name (possibly with a middle initial) "
    "followed by a surname. Read ONLY the given name portion (including any "
    "middle initial that is part of the given name, e.g. 'Milo J.').\n\n"
    f"{_ARCHAIC_ABBREVIATIONS}\n\n"
    "Transcribe the EXACT letters as written — do NOT expand abbreviations "
    "(so 'Chls' stays 'Chls'; 'Wm.' stays 'Wm.'; you should NEVER return "
    "the expanded form). Do NOT normalize to a more familiar modern name.\n\n"
    'Return ONLY this JSON: {"name":"<given name>","confidence":'
    '"HIGH|MEDIUM|LOW"}. No prose outside.'
)

LAST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name followed by a surname. Read ONLY the SURNAME "
    "portion.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return the literal string [DITTO]. Otherwise transcribe the EXACT '
    "letters as written; do NOT normalize to a more familiar spelling.\n\n"
    'Return ONLY this JSON: {"name":"<surname or [DITTO]>","confidence":'
    '"HIGH|MEDIUM|LOW"}. No prose outside.'
)

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "confidence"]}

NAME_COL_FRAC = (0.15, 0.37)
Y_PAD = 40
UPSCALE = 2
TARGET_LINES = [35]


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
    scratch = Path(__file__).parent / "_s13"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s13: abbreviation-hint prompt on {TARGET_LINES} of {reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | {'Claude':<15} | {'Gemini':<15} | truth")
    print("-" * 68)

    results = []
    for line in TARGET_LINES:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
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
