#!/usr/bin/env python3
"""s10 — combined name + letter counts in a single call per model.

Merges s2's name read with s9's count read: each call returns name + total_letters
+ unique_letters + confidence in one JSON. Gives us per-cell signals we couldn't
get before:

  • self-consistency: does the model's returned name have the same letter count
    as its own total_letters? (catches internal contradictions)
  • cross-model count agreement: an independent shape-level confidence signal
    on top of name-agreement

Claude + Gemini Flash, split first/last, tight name-cell crop. Standalone.
Same call count as s2 (32).

Run:
    .venv/bin/python src/strategies/s10.py
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
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name (first name, possibly with a middle "
    "initial), then a surname. Look ONLY at the given name portion (first "
    "name plus any middle initial).\n\n"
    "Return three things in one JSON object:\n"
    "  1. name: the given name transcribed EXACTLY as written — do NOT expand "
    "abbreviations ('Chls' stays 'Chls', not 'Charles' or 'Chas') and do NOT "
    "normalize to a more familiar spelling. Include the middle initial "
    "(e.g. 'Milo J.', not just 'Milo').\n"
    "  2. total_letters: how many individual letters are in the given name. "
    "Do NOT count periods, apostrophes, or spaces. Middle initials count as "
    "letters (so 'Milo J.' has 5 letters).\n"
    "  3. unique_letters: how many DISTINCT letter shapes are used. Treat "
    "uppercase and lowercase forms of the same letter as one shape "
    "('Anna' has 4 total letters but 2 unique shapes).\n\n"
    'Return ONLY this JSON: {"name":"<given name>","total_letters":<int>,'
    '"unique_letters":<int>,"confidence":"HIGH|MEDIUM|LOW"}. No prose outside.'
)

LAST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — "
    "a handwritten given name followed by a surname. Look ONLY at the SURNAME "
    "portion.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return ONLY: {"name":"[DITTO]","total_letters":0,"unique_letters":0,'
    '"confidence":"HIGH"} and skip the rest.\n\n'
    "Otherwise, return four things in one JSON object:\n"
    "  1. name: the surname transcribed EXACTLY as written — do NOT normalize "
    "to a more familiar spelling.\n"
    "  2. total_letters: how many individual letters are in the surname. "
    "Do NOT count periods, apostrophes, or spaces.\n"
    "  3. unique_letters: how many DISTINCT letter shapes are used. Treat "
    "uppercase and lowercase forms of the same letter as one shape.\n"
    "  4. confidence: HIGH, MEDIUM, or LOW.\n\n"
    'Return ONLY this JSON: {"name":"<surname or [DITTO]>","total_letters":'
    '<int>,"unique_letters":<int>,"confidence":"HIGH|MEDIUM|LOW"}. No prose '
    "outside."
)

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "total_letters": {"type": "integer"},
    "unique_letters": {"type": "integer"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "total_letters", "unique_letters", "confidence"]}

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
    if not name or name.strip().upper() == "[DITTO]":
        return (0, 0)
    letters = [c for c in name if c.isalpha()]
    return (len(letters), len({c.lower() for c in letters}))


def truth_for(cases, line, field):
    for c in cases:
        if c["line"] == line and c["field"] == field:
            return c["correct"]
    return None


def name_letter_count(name: str) -> int:
    """Number of alpha characters in a name (matches how truth_counts computes total)."""
    if not name or name.strip().upper() == "[DITTO]":
        return 0
    return sum(1 for c in name if c.isalpha())


def read_one(backend, crop: Path, prompt: str) -> dict:
    try:
        r = backend(crop, prompt, SCHEMA)
        return {"name": r.get("name", "(missing)"),
                "total_letters": r.get("total_letters"),
                "unique_letters": r.get("unique_letters"),
                "confidence": r.get("confidence", "?")}
    except Exception as e:
        return {"name": f"ERR: {type(e).__name__}: {str(e)[:60]}",
                "total_letters": None, "unique_letters": None,
                "confidence": "?"}


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s10"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s10: name + count read on {len(lines)} rows of {reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | "
          f"{'C name':<15} {'C t/u':>7} {'C-selfok':>9} | "
          f"{'G name':<15} {'G t/u':>7} {'G-selfok':>9} | "
          f"{'truth name':<15} {'truth t/u':>10}")
    print("-" * 120)

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
            t_tot, t_uniq = truth_counts(truth)

            # self-consistency: model's own name-length vs its own total_letters
            c_selfok = (c["total_letters"] is not None
                        and name_letter_count(c["name"]) == c["total_letters"])
            g_selfok = (g["total_letters"] is not None
                        and name_letter_count(g["name"]) == g["total_letters"])

            print(f"L{line:02d} {field:<5} | "
                  f"{c['name']!r:<15} {str(c['total_letters']) + '/' + str(c['unique_letters']):>7} "
                  f"{'YES' if c_selfok else ' no':>9} | "
                  f"{g['name']!r:<15} {str(g['total_letters']) + '/' + str(g['unique_letters']):>7} "
                  f"{'YES' if g_selfok else ' no':>9} | "
                  f"{truth!r:<15} {f'{t_tot}/{t_uniq}':>10}",
                  flush=True)

            row["fields"][field] = {
                "truth": truth,
                "truth_total_letters": t_tot,
                "truth_unique_letters": t_uniq,
                "claude": {**c, "self_consistent": c_selfok},
                "gemini_flash": {**g, "self_consistent": g_selfok},
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
