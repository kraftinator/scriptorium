#!/usr/bin/env python3
"""s12 — wider crop with adjacent rows for scribe-consistency context.

The name cell for a given row is cropped along with 2 rows above and 2 rows
below (name-column only, 5 rows total), and the target row is outlined in RED.
The prompt tells the model: read only the row inside the red box; use the
other rows as reference for how this same scribe writes capital letters,
so you can compare the marks in the target row against known letter shapes
elsewhere on the page.

Motivation: L35 first ("Wm. H.") is systematically misread as "Mary [x]" by
both models. Same-scribe context might help — if the models can see the
scribe's "W" and "m" elsewhere, they might recognize them in the target row.

Claude + Gemini Flash, split first/last. Standalone. 32 calls.

Run:
    .venv/bin/python src/strategies/s12.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from backends import claude_backend, gemini_backend  # noqa: E402
import backends  # noqa: E402

FIRST_PROMPT = (
    "This image shows a strip of a handwritten 1850 U.S. Census name column, "
    "containing FIVE consecutive rows. ONE of the rows is outlined in RED — "
    "that is your TARGET row. The other four rows are shown ONLY as "
    "reference for how this same scribe writes capital letters and other "
    "shapes — do NOT read them out.\n\n"
    "Look at the marks IN THE RED BOX only. Read the given name (first name "
    "plus any middle initial) as written. Compare individual letter shapes "
    "against how the same shapes appear in the surrounding rows if that "
    "helps disambiguate a stroke (same scribe, same style).\n\n"
    "Transcribe the EXACT letters as written — do NOT expand abbreviations "
    "('Chls' stays 'Chls'; 'Wm.' stays 'Wm.') and do NOT normalize to a "
    "modern spelling. Include any middle initial.\n\n"
    'Return ONLY this JSON: {"name":"<given name>","confidence":"HIGH|MEDIUM|LOW"}. '
    "No prose outside."
)

LAST_PROMPT = (
    "This image shows a strip of a handwritten 1850 U.S. Census name column, "
    "containing FIVE consecutive rows. ONE of the rows is outlined in RED — "
    "that is your TARGET row. The other four rows are shown ONLY as "
    "reference for how this same scribe writes capital letters and other "
    "shapes — do NOT read them out.\n\n"
    "Look at the marks IN THE RED BOX only. Read the SURNAME portion of the "
    "target row. If the surname is a ditto mark (indicating 'same as the "
    "row above'), return the literal string [DITTO].\n\n"
    "Compare individual letter shapes against how the same shapes appear in "
    "the surrounding rows if that helps disambiguate a stroke (same scribe, "
    "same style). Transcribe the EXACT letters written — do NOT normalize.\n\n"
    'Return ONLY this JSON: {"name":"<surname or [DITTO]>",'
    '"confidence":"HIGH|MEDIUM|LOW"}. No prose outside.'
)

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "confidence"]}

NAME_COL_FRAC = (0.15, 0.37)
CONTEXT_ROWS_EACH_SIDE = 2   # rows above and below target
Y_PAD = 20                    # px padding around the 5-row block
UPSCALE = 2
BOX_COLOR = (220, 30, 30)     # red outline
BOX_WIDTH = 4                 # px, pre-upscale


def crop_context_cell(img: Image.Image, top: int, pitch: int, line: int,
                      dst: Path) -> Path:
    """5-row crop (target ± 2) of the name column, with the target row boxed in red."""
    W, H = img.size
    r_start = max(1, line - CONTEXT_ROWS_EACH_SIDE)
    r_end = line + CONTEXT_ROWS_EACH_SIDE
    y0_strip = max(0, top + (r_start - 1) * pitch - Y_PAD)
    y1_strip = min(H, top + r_end * pitch + Y_PAD)
    x0, x1 = int(NAME_COL_FRAC[0] * W), int(NAME_COL_FRAC[1] * W)

    strip = img.crop((x0, y0_strip, x1, y1_strip)).convert("RGB").copy()

    # box the target row within the strip's coordinate space
    box_y0 = (top + (line - 1) * pitch) - y0_strip
    box_y1 = (top + line * pitch)       - y0_strip
    draw = ImageDraw.Draw(strip)
    # rectangle relative to strip origin, ±few px for visibility
    draw.rectangle(
        [(2, max(0, box_y0 - 2)),
         (strip.width - 3, min(strip.height - 1, box_y1 + 2))],
        outline=BOX_COLOR, width=BOX_WIDTH,
    )

    strip = strip.resize((strip.width * UPSCALE, strip.height * UPSCALE),
                         Image.LANCZOS)
    strip.save(dst)
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
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s12"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s12: context-crop (5 rows, target boxed) on {len(lines)} rows of "
          f"{reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | {'Claude':<18} | {'Gemini':<18} | truth")
    print("-" * 78)

    results = []
    for line in lines:
        crop = crop_context_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        row = {"line": line, "crop": crop.name, "fields": {}}
        for field, prompt, field_key in (
            ("first", FIRST_PROMPT, "interpreted_first_name"),
            ("last",  LAST_PROMPT,  "interpreted_last_name"),
        ):
            cn, cc = read_one(claude_backend, crop, prompt)
            gn, gc = read_one(gemini_backend, crop, prompt)
            truth = truth_for(gt["cases"], line, field_key) or ""
            print(f"L{line:02d} {field:<5} | "
                  f"{cn!r:<18} | {gn!r:<18} | {truth!r}", flush=True)
            row["fields"][field] = {
                "truth": truth,
                "claude": {"name": cn, "confidence": cc},
                "gemini_flash": {"name": gn, "confidence": gc},
            }
        results.append(row)

    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "models": {"claude": backends.CLAUDE_MODEL,
                   "gemini_flash": backends.GEMINI_MODEL},
        "first_prompt": FIRST_PROMPT, "last_prompt": LAST_PROMPT,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
