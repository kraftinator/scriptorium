#!/usr/bin/env python3
"""s20 — same as s2 but with a calibration rubric + needs_review escape hatch.

User's insight: Opus reports MEDIUM on both L33 Gibsen (clearly written) and
L35 Mary N. (guessed on unreadable strokes). The pipeline needs uncertainty
INFORMATION to escalate hard cells to a second reader — a well-calibrated 10/12
is more useful than an over-confident 11/12.

Two calibration levers in this prompt:
  1. Explicit rubric telling the model when to use LOW/MEDIUM/HIGH.
  2. A binary `needs_review` flag: return true whenever the model is guessing,
     the strokes are ambiguous, or a human transcriber might disagree — even
     if confidence is MEDIUM or HIGH. Independent signal.

Same tight cell crop as s2. Claude Opus + Gemini Flash. All 8 rows.
32 calls, ~15-25 min.

Run:
    .venv/bin/python src/strategies/s20.py
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

_CALIBRATION_RUBRIC = (
    "CALIBRATION IS CRITICAL. Use the confidence levels as follows:\n"
    "  HIGH   — every letter shape is clearly identifiable; a second human "
    "transcriber would almost certainly agree with your reading.\n"
    "  MEDIUM — 1-2 letters are somewhat ambiguous but you can commit to a "
    "reading (say, 70-90% sure).\n"
    "  LOW    — multiple letters are hard to identify OR the reading is "
    "essentially a guess based on your best interpretation. If you cannot "
    "clearly see each letter, you MUST report LOW.\n\n"
    "In addition, return a boolean `needs_review`:\n"
    "  true  — the cell is genuinely ambiguous, you had to guess, the strokes "
    "don't clearly match any obvious reading, or a human transcriber might "
    "reasonably disagree with you. Return true EVEN if confidence is MEDIUM "
    "or HIGH — this is an independent flag for cells worth a second look.\n"
    "  false — the strokes are clear enough that the reading is unambiguous.\n\n"
    "Do NOT default to MEDIUM to avoid commitment. When strokes are hard to "
    "make out, LOW + needs_review=true is the RIGHT answer, not MEDIUM."
)

FIRST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "it contains a handwritten given name (possibly with a middle initial) "
    "followed by a surname. Read ONLY the given name portion (including any "
    "middle initial that is part of the given name, e.g. 'Milo J.').\n\n"
    "Transcribe the EXACT letters as written — do NOT expand abbreviations "
    "('Chls' stays 'Chls', 'Wm.' stays 'Wm.') and do NOT normalize to a more "
    "familiar spelling.\n\n"
    f"{_CALIBRATION_RUBRIC}\n\n"
    'Return ONLY this JSON: {"name":"<given name>","confidence":'
    '"HIGH|MEDIUM|LOW","needs_review":<bool>}. No prose outside.'
)

LAST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name followed by a surname. Read ONLY the SURNAME "
    "portion.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return: {"name":"[DITTO]","confidence":"HIGH","needs_review":false}\n\n'
    "Otherwise, transcribe the EXACT letters as written — do NOT normalize.\n\n"
    f"{_CALIBRATION_RUBRIC}\n\n"
    'Return ONLY this JSON: {"name":"<surname or [DITTO]>","confidence":'
    '"HIGH|MEDIUM|LOW","needs_review":<bool>}. No prose outside.'
)

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "needs_review": {"type": "boolean"}},
    "required": ["name", "confidence", "needs_review"]}

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
                "needs_review": bool(r.get("needs_review", False))}
    except Exception as e:
        return {"name": f"ERR: {type(e).__name__}: {str(e)[:60]}",
                "confidence": "?", "needs_review": True}


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s20"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s20: calibrated read + needs_review flag on {len(lines)} rows of "
          f"{reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | "
          f"{'Claude':<15} {'C-conf':<7} {'C-rev':<6} | "
          f"{'Gemini':<15} {'G-conf':<7} {'G-rev':<6} | truth")
    print("-" * 108)

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
            print(f"L{line:02d} {field:<5} | "
                  f"{c['name']!r:<15} {c['confidence']:<7} "
                  f"{str(c['needs_review']):<6} | "
                  f"{g['name']!r:<15} {g['confidence']:<7} "
                  f"{str(g['needs_review']):<6} | {truth!r}",
                  flush=True)
            row["fields"][field] = {
                "truth": truth,
                "claude": c,
                "gemini_flash": g,
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
