#!/usr/bin/env python3
"""s2 — split-name read on the designated rows, both models, NO adjudication.

Reads the FIRST name and the LAST name as SEPARATE calls from each model.
Same crop (whole name cell) both times; two different focused prompts. Prints
first/last per model alongside the ground truth. No debate, no reconciliation.

Purpose: the counterpart to s1 (combined-name read). Together they let us test
whether split-vs-combined framing changes model accuracy on the same crop.

Standalone: owns its prompts, crop shape, output shape. Imports only raw model
I/O from backends.py.

Run:
    .venv/bin/python src/strategies/s2.py
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

FIRST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name (first name, possibly with a middle "
    "initial), then a surname. Read ONLY the given name portion (including "
    "any middle initial that is part of the given name, e.g. 'Milo J.', not "
    "just 'Milo'). Transcribe the EXACT letters as written — do NOT expand "
    "abbreviations ('Chls' stays 'Chls', not 'Charles' or 'Chas') and do NOT "
    "normalize to a more familiar spelling. Return JSON: name (the given name "
    "only), confidence (HIGH/MEDIUM/LOW). No prose outside the JSON."
)
LAST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name, then a surname. Read ONLY the SURNAME "
    "portion. Transcribe the EXACT letters as written — do NOT normalize to a "
    "more familiar spelling. If the surname is a ditto mark (indicating "
    "'same as the row above'), return the literal string [DITTO]. Return "
    "JSON: name (the surname only, or [DITTO]), confidence (HIGH/MEDIUM/LOW). "
    "No prose outside the JSON."
)
SCHEMA = {"type": "object", "properties": {
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


def truth_for(cases: list[dict], line: int, field: str):
    for c in cases:
        if c["line"] == line and c["field"] == field:
            return c["correct"]
    return None


def read_one(backend, crop: Path, prompt: str) -> tuple[str, str]:
    try:
        r = backend(crop, prompt, SCHEMA)
        return r.get("name", "(missing)"), r.get("confidence", "?")
    except Exception as e:
        return f"ERR: {type(e).__name__}", "?"


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s2"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s2: split-name read on {len(lines)} rows of {reel} frame {frame:04d}")
    print(f"{'line':>4}  "
          f"{'claude first':<15} {'claude last':<15}  "
          f"{'gemini first':<15} {'gemini last':<15}  "
          f"{'truth first':<12} {'truth last':<12}")
    print("-" * 120)
    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        cf_name, cf_conf = read_one(claude_backend, crop, FIRST_PROMPT)
        cl_name, cl_conf = read_one(claude_backend, crop, LAST_PROMPT)
        gf_name, gf_conf = read_one(gemini_backend, crop, FIRST_PROMPT)
        gl_name, gl_conf = read_one(gemini_backend, crop, LAST_PROMPT)
        tf = truth_for(gt["cases"], line, "interpreted_first_name") or ""
        tl = truth_for(gt["cases"], line, "interpreted_last_name") or ""
        print(f"L{line:02d}   "
              f"{cf_name!r:<15} {cl_name!r:<15}  "
              f"{gf_name!r:<15} {gl_name!r:<15}  "
              f"{tf!r:<12} {tl!r:<12}", flush=True)
        results.append({
            "line": line, "crop": crop.name,
            "truth": {"first": tf, "last": tl},
            "claude": {"first": {"name": cf_name, "confidence": cf_conf},
                       "last":  {"name": cl_name, "confidence": cl_conf}},
            "gemini": {"first": {"name": gf_name, "confidence": gf_conf},
                       "last":  {"name": gl_name, "confidence": gl_conf}},
        })
    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "first_prompt": FIRST_PROMPT, "last_prompt": LAST_PROMPT,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
