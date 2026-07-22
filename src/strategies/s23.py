#!/usr/bin/env python3
"""s23 — s1's shape (combined-name read, tight cell crop) with claude-fable-5 only.

Half the calls of s2/s21/s22 (8 vs 16-32) because a single call returns the
WHOLE name (first + last) instead of two calls (first, last). Test whether
Fable-5 handles the combined framing well.

Same prompt shape as s1 (no calibration rubric — plain exact-transcription).
8 rows × 1 model × 1 call = 8 calls.

Run:
    .venv/bin/python src/strategies/s23.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import backends  # noqa: E402

FABLE_MODEL = "claude-fable-5"

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


def fable_backend(jpg: Path, prompt: str, schema: dict) -> dict:
    full = (f"{prompt}\n\nRead the census page image at:\n{jpg}\n\n"
            f"Conform the output exactly to this JSON schema:\n{json.dumps(schema, separators=(',', ':'))}")
    last = ""
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["claude", "-p", full, "--model", FABLE_MODEL,
                 "--allowedTools", "Read", "--strict-mcp-config"],
                capture_output=True, text=True, timeout=backends.CLAUDE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            last = f"CLI hung > {backends.CLAUDE_TIMEOUT_S}s"
            time.sleep(2 ** attempt); continue
        text = backends._strip_fence(r.stdout.strip())
        if r.returncode == 0 and text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                last = "invalid JSON from CLI"
        else:
            last = f"exit {r.returncode}, empty={not text}"
        time.sleep(2 ** attempt)
    raise RuntimeError(f"claude-fable-5 failed after 3 attempts: {last}")


def ground_truth_full_name(cases, line):
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


def read_one(crop: Path, prompt: str) -> tuple[str, str]:
    try:
        r = fable_backend(crop, prompt, SCHEMA)
        return r.get("name", "(missing)"), r.get("confidence", "?")
    except Exception as e:
        return f"ERR: {type(e).__name__}: {str(e)[:60]}", "?"


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s23"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s23: combined-name read on {FABLE_MODEL} only, {len(lines)} rows of "
          f"{reel} frame {frame:04d}")
    print(f"{'line':>4}  {'fable-5':<25}  {'confidence':<8}  ground truth")
    print("-" * 78)
    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        name, conf = read_one(crop, PROMPT)
        truth = ground_truth_full_name(gt["cases"], line)
        print(f"L{line:02d}   {name!r:<25}  {conf:<8}  {truth!r}", flush=True)
        results.append({
            "line": line, "crop": crop.name, "truth": truth,
            "fable_5": {"name": name, "confidence": conf},
        })
    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "models": {"fable_5": FABLE_MODEL},
        "prompt": PROMPT, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
