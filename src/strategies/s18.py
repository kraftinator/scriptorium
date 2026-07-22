#!/usr/bin/env python3
"""s18 — force each model to enumerate 3 candidates before picking one.

Hypothesis: Opus over-primes to familiar common names (Mary, John, William)
and locks in on its first impulse. If we force it to LIST 3 plausible readings
first, ranked by stroke-fit not by name familiarity, then pick from that list,
it may consider alternatives that its default reading suppresses.

Generic prompt (no cell-specific hints). Same tight name-cell crop as s2.
Claude + Gemini Flash. Runs on ALL 8 fixture rows so we can see if the
'enumerate then pick' structure helps hard cells without hurting easy ones.

32 calls, ~15-25 min.

Run:
    .venv/bin/python src/strategies/s18.py
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

FIRST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name (possibly with a middle initial) followed by "
    "a surname. Read ONLY the given name portion.\n\n"
    "IMPORTANT: models over-recognize familiar common names (Mary, John, "
    "William, Sarah, etc.) even when the strokes don't clearly match. Do "
    "NOT commit to your first impulse. Follow this procedure:\n\n"
    "STEP 1 — List UP TO 3 plausible readings of the given name, ranked by "
    "how well each matches the ACTUAL LETTER SHAPES on the page (not by how "
    "common the name is). For each candidate, briefly note the stroke evidence "
    "that supports it. Consider:\n"
    "  - Common 1850-era male abbreviations: Wm. (William), Chas./Chls. "
    "(Charles), Jno. (John), Geo. (George), Jas. (James), Saml. (Samuel), "
    "Robt. (Robert), Danl. (Daniel), Thos. (Thomas), Jos. (Joseph), Benj. "
    "(Benjamin)\n"
    "  - Common modern first names (male and female)\n"
    "  - Any other reading the strokes could support (uncommon names, "
    "different letter interpretations, etc.)\n\n"
    "STEP 2 — From your candidate list, pick the one that best matches the "
    "actual strokes. If two or more candidates are equally plausible, set "
    "confidence to LOW.\n\n"
    "Transcribe the EXACT letters as written — do NOT expand abbreviations "
    "('Chls' stays 'Chls', 'Wm.' stays 'Wm.'). Include any middle initial.\n\n"
    'Return ONLY this JSON:\n'
    '{"candidates":[{"reading":"<name>","why":"<brief stroke evidence>"},...],'
    '"final":"<your chosen reading>","confidence":"HIGH|MEDIUM|LOW"}\n'
    "No prose outside."
)

LAST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name followed by a surname. Read ONLY the SURNAME "
    "portion.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return: {"candidates":[{"reading":"[DITTO]","why":"ditto mark visible"}],'
    '"final":"[DITTO]","confidence":"HIGH"}\n\n'
    "Otherwise: models over-recognize familiar surnames even when the strokes "
    "don't clearly match. Do NOT commit to your first impulse.\n\n"
    "STEP 1 — List UP TO 3 plausible readings of the surname, ranked by how "
    "well each matches the ACTUAL LETTER SHAPES. Note stroke evidence for "
    "each. Consider common surnames but also uncommon options if the strokes "
    "support them.\n\n"
    "STEP 2 — From your list, pick the one that best matches the strokes. "
    "If two or more are equally plausible, set confidence to LOW.\n\n"
    "Transcribe the EXACT letters written — do NOT normalize to a more "
    "familiar spelling.\n\n"
    'Return ONLY this JSON:\n'
    '{"candidates":[{"reading":"<name>","why":"<brief stroke evidence>"},...],'
    '"final":"<your chosen reading>","confidence":"HIGH|MEDIUM|LOW"}\n'
    "No prose outside."
)

SCHEMA = {"type": "object", "properties": {
    "candidates": {"type": "array", "items": {"type": "object", "properties": {
        "reading": {"type": "string"}, "why": {"type": "string"}}}},
    "final": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["final", "confidence"]}

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


def read_one(backend, crop: Path, prompt: str) -> dict:
    try:
        r = backend(crop, prompt, SCHEMA)
        return {"final": r.get("final", "(missing)"),
                "candidates": r.get("candidates") or [],
                "confidence": r.get("confidence", "?")}
    except Exception as e:
        return {"final": f"ERR: {type(e).__name__}: {str(e)[:60]}",
                "candidates": [], "confidence": "?"}


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = TARGET_LINES

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s18"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s18: enumerate-3-candidates prompt on {len(lines)} rows of "
          f"{reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | {'Claude final':<15} | {'Gemini final':<15} | truth")
    print("-" * 68)

    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        row = {"line": line, "crop": crop.name, "fields": {}}
        for field, prompt, field_key in (
            ("first", FIRST_PROMPT, "interpreted_first_name"),
            ("last",  LAST_PROMPT,  "interpreted_last_name"),
        ):
            c = read_one(claude_backend, crop, prompt)
            g = read_one(gemini_backend, crop, prompt)
            truth = truth_for(gt["cases"], line, field_key) or ""
            print(f"L{line:02d} {field:<5} | {c['final']!r:<15} | "
                  f"{g['final']!r:<15} | {truth!r}", flush=True)
            row["fields"][field] = {
                "truth": truth,
                "claude": c,
                "gemini_flash": g,
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
