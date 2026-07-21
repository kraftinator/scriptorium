#!/usr/bin/env python3
"""s9 — each model counts letters and unique letters (no name read).

Purpose: measure whether the two models AGREE at the shape-count level.
If they see the same number of letters and the same number of unique letters,
that's a shape-agreement signal independent of word recognition. Disagreement
in counts tells us the crop is genuinely ambiguous at the segmentation level.

Claude + Gemini Flash, same tight cell crop as s2. Split first/last.
Standalone. No adjudication.

Run:
    .venv/bin/python src/strategies/s9.py
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
    "a surname. Look ONLY at the given name portion (first name plus any "
    "middle initial). Do the following two counts, based purely on what letter "
    "SHAPES you can see:\n\n"
    "1. total_letters: how many individual letters are in the given name. "
    "Do NOT count periods, apostrophes, or spaces. Middle initials count as "
    "letters (e.g. 'Milo J.' has 5 letters).\n"
    "2. unique_letters: how many DISTINCT letter shapes are used. Treat "
    "uppercase and lowercase forms of the same letter as one shape "
    "(e.g. 'Anna' has 4 total letters but 2 unique shapes: A/a and n).\n\n"
    'Return ONLY this JSON: {"total_letters": <int>, "unique_letters": '
    '<int>, "confidence": "HIGH|MEDIUM|LOW"}. No prose outside the JSON.'
)

LAST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name followed by a surname. Look ONLY at the SURNAME "
    "portion.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return ONLY: {"total_letters": 0, "unique_letters": 0, '
    '"confidence": "HIGH", "ditto": true}.\n\n'
    "Otherwise, do the following two counts, based purely on what letter "
    "SHAPES you can see:\n\n"
    "1. total_letters: how many individual letters are in the surname. "
    "Do NOT count periods, apostrophes, or spaces.\n"
    "2. unique_letters: how many DISTINCT letter shapes are used. Treat "
    "uppercase and lowercase forms of the same letter as one shape.\n\n"
    'Return ONLY this JSON: {"total_letters": <int>, "unique_letters": '
    '<int>, "confidence": "HIGH|MEDIUM|LOW"}. No prose outside the JSON.'
)

SCHEMA = {"type": "object", "properties": {
    "total_letters": {"type": "integer"},
    "unique_letters": {"type": "integer"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "ditto": {"type": "boolean"}},
    "required": ["total_letters", "unique_letters", "confidence"]}

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


def truth_counts(name: str) -> tuple[int, int]:
    """(total_letters, unique_letters) from a ground-truth string.

    Letters only (no periods, apostrophes, spaces). Unique is case-insensitive.
    Returns (0, 0) if the string is empty or [DITTO].
    """
    if not name or name.strip().upper() == "[DITTO]":
        return (0, 0)
    letters = [c for c in name if c.isalpha()]
    return (len(letters), len({c.lower() for c in letters}))


def truth_for(cases, line, field):
    for c in cases:
        if c["line"] == line and c["field"] == field:
            return c["correct"]
    return None


def read_one(backend, crop: Path, prompt: str) -> dict:
    try:
        r = backend(crop, prompt, SCHEMA)
        return {"total_letters": r.get("total_letters"),
                "unique_letters": r.get("unique_letters"),
                "confidence": r.get("confidence", "?"),
                "ditto": bool(r.get("ditto", False))}
    except Exception as e:
        return {"total_letters": None, "unique_letters": None,
                "confidence": "?", "ditto": False,
                "error": f"{type(e).__name__}: {str(e)[:60]}"}


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s9"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s9: letter-count read on {len(lines)} rows of {reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | {'C tot':>5} {'C uniq':>6} | "
          f"{'G tot':>5} {'G uniq':>6} | {'tot':>3} {'uniq':>4}  truth")
    print("-" * 84)

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
            t_total, t_unique = truth_counts(truth)
            print(f"L{line:02d} {field:<5} | "
                  f"{str(c['total_letters']):>5} {str(c['unique_letters']):>6} | "
                  f"{str(g['total_letters']):>5} {str(g['unique_letters']):>6} | "
                  f"{t_total:>3} {t_unique:>4}  {truth!r}",
                  flush=True)
            row["fields"][field] = {
                "truth": truth,
                "truth_total_letters": t_total,
                "truth_unique_letters": t_unique,
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
