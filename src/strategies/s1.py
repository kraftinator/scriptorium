#!/usr/bin/env python3
"""s1 — combined-name read on the designated rows, both models, NO adjudication.

Reads the whole name cell (first + last together, as the scribe wrote it) as
ONE string from each model independently, then prints the two readings side by
side with the ground truth. No debate, no reconciliation — just the raw data.

Purpose: test whether reading the name as a single string (letting the models
handle first/middle/last boundaries themselves) beats the current per-field
split baseline on the same fixture cells.

Standalone by design: this script owns its prompt, its crop shape, and its
output format. It imports only the raw model I/O from backends.py.

Run:
    .venv/bin/python src/strategies/s1.py
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

PROMPT = (
    "Read the ENTIRE handwritten name written on this single row of an 1850 "
    "U.S. Census page — first name, any middle initial, and last name — as one "
    "string exactly as the scribe wrote it. Transcribe the EXACT letters "
    "written; do NOT expand abbreviations ('Chls' stays 'Chls', not 'Charles' "
    "or 'Chas') and do NOT normalize to a more familiar spelling. Include any "
    "middle initial (e.g. 'Milo J.', not just 'Milo'). If the surname is a "
    "ditto mark, write [DITTO]. Return JSON: name (the full name as one "
    "string), confidence (HIGH/MEDIUM/LOW). No prose outside the JSON."
)
SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "confidence"]}

# name column occupies roughly x in [0.15, 0.37] of the page width; the rest of
# the layout (row1_top, row_pitch) comes from the corpus layout.json.
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
    """Return a PNG path for the page, converting from JP2 if needed."""
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


def ground_truth_full_name(cases: list[dict], line: int) -> str:
    """Assemble expected 'first last' from the fixture cases for one line."""
    first = last = None
    for c in cases:
        if c["line"] != line:
            continue
        if c["field"] == "interpreted_first_name":
            first = c["correct"]
        elif c["field"] == "interpreted_last_name":
            last = c["correct"]
    parts = [p for p in (first, last) if p]
    return " ".join(parts) if parts else "(no fixture)"


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s1"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s1: combined-name read on {len(lines)} rows of {reel} frame {frame:04d}")
    print(f"{'line':>4}  {'claude':<25}  {'gemini':<25}  {'ground truth':<25}")
    print("-" * 90)
    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        try:
            c = claude_backend(crop, PROMPT, SCHEMA)
            c_name = c.get("name", "(missing)")
            c_conf = c.get("confidence", "?")
        except Exception as e:
            c_name, c_conf = f"ERR: {type(e).__name__}", "?"
        try:
            g = gemini_backend(crop, PROMPT, SCHEMA)
            g_name = g.get("name", "(missing)")
            g_conf = g.get("confidence", "?")
        except Exception as e:
            g_name, g_conf = f"ERR: {type(e).__name__}", "?"
        truth = ground_truth_full_name(gt["cases"], line)
        print(f"L{line:02d}   {c_name!r:<25}  {g_name!r:<25}  {truth!r:<25}  "
              f"[c={c_conf} g={g_conf}]", flush=True)
        results.append({
            "line": line, "crop": crop.name, "truth": truth,
            "claude": {"name": c_name, "confidence": c_conf},
            "gemini": {"name": g_name, "confidence": g_conf},
        })
    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "prompt": PROMPT, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
