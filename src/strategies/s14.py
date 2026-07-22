#!/usr/bin/env python3
"""s14 — candidate-list prompt, L35 only. Option 5.

Give the model an explicit menu of common 1850-era first names (male
abbreviations, common male names spelled out, common female names) plus
'or something else', and tell it to pick the best match. Heavier-handed than
s13's mere hint — forces the model to actively consider abbreviated forms
against the shapes on the page.

Same tight cell crop as s2/s13 (prompt-only intervention). Claude + Gemini
Flash. Scope: L35 only. 4 calls total.

Run:
    .venv/bin/python src/strategies/s14.py
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

_CANDIDATE_LIST = (
    "The given name in 1850 U.S. Census records is USUALLY one of these "
    "common patterns:\n\n"
    "  Male abbreviations (return the ABBREVIATION as written, NOT the "
    "expanded form):\n"
    "    Wm. (William), Chas. or Chls. (Charles), Jno. (John), Geo. (George), "
    "Jas. (James), Saml. (Samuel), Robt. (Robert), Danl. (Daniel), Thos. "
    "(Thomas), Jos. (Joseph), Benj. (Benjamin), Nichs. (Nicholas), Alex. "
    "(Alexander), Fredk. (Frederick), Isaac, Ezra, Milo, Silas\n\n"
    "  Common male names written out:\n"
    "    John, William, James, George, Henry, Thomas, Charles, Samuel, David, "
    "Joseph, Andrew, Peter, Jacob, Daniel, Edward, Robert, Michael, Alexander\n\n"
    "  Common female names written out:\n"
    "    Mary, Elizabeth, Sarah, Jane, Nancy, Susan, Margaret, Catherine, "
    "Rebecca, Ann, Julia, Eliza, Lucy, Emma, Louisa, Hannah, Harriet, "
    "Clarinda, Julietta\n\n"
    "  Or a name not in this list."
)

FIRST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name (possibly with a middle initial) "
    "followed by a surname. Read ONLY the given name portion.\n\n"
    f"{_CANDIDATE_LIST}\n\n"
    "Study the strokes in the image and pick which candidate best MATCHES the "
    "letters actually written. If it matches a male abbreviation like 'Wm.' "
    "or 'Chas.', return the abbreviation. If it matches a common name, "
    "return that spelling. If it's a name not in the list, transcribe the "
    "EXACT letters written.\n\n"
    "Include any middle initial (e.g. 'Wm. H.', 'Mary A.', 'Milo J.').\n\n"
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
    scratch = Path(__file__).parent / "_s14"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s14: candidate-list prompt on {TARGET_LINES} of {reel} frame {frame:04d}")
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
