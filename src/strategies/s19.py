#!/usr/bin/env python3
"""s19 — s2's shape but three Claude models (Opus, Sonnet, Haiku), no Gemini.

Same tight name-cell crop + simple exact-transcription prompts as s2. Full 8
fixture rows. Purpose: extend s17's L35-only Claude-variant comparison to the
whole marquee set. Where does each Claude variant win / lose across the full
range of cells?

3 models × 2 fields × 8 rows = 48 calls, ~30-45 min.

Run:
    .venv/bin/python src/strategies/s19.py
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

CLAUDE_MODELS = {
    "opus": backends.CLAUDE_MODEL,   # claude-opus-4-8 (baseline)
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

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


def claude_with_model(jpg: Path, prompt: str, schema: dict, model: str) -> dict:
    """Local variant of backends.claude_backend that takes a model parameter."""
    full = (f"{prompt}\n\nRead the census page image at:\n{jpg}\n\n"
            f"Conform the output exactly to this JSON schema:\n{json.dumps(schema, separators=(',', ':'))}")
    last = ""
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["claude", "-p", full, "--model", model,
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
    raise RuntimeError(f"claude {model} failed after 3 attempts: {last}")


def truth_for(cases, line, field):
    for c in cases:
        if c["line"] == line and c["field"] == field:
            return c["correct"]
    return None


def read_one(crop: Path, prompt: str, model: str) -> tuple[str, str]:
    try:
        r = claude_with_model(crop, prompt, SCHEMA, model)
        return r.get("name", "(missing)"), r.get("confidence", "?")
    except Exception as e:
        return f"ERR: {type(e).__name__}: {str(e)[:60]}", "?"


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s19"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s19: three Claude variants (Opus + Sonnet + Haiku) on {len(lines)} rows "
          f"of {reel} frame {frame:04d}")
    print(f"{'ln':>3} {'fld':<5} | {'Opus':<15} | {'Sonnet':<15} | {'Haiku':<15} | truth")
    print("-" * 88)

    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        row = {"line": line, "crop": crop.name, "fields": {}}
        for field, prompt, field_key in (
            ("first", FIRST_PROMPT, "interpreted_first_name"),
            ("last",  LAST_PROMPT,  "interpreted_last_name"),
        ):
            reads = {}
            for label, model in CLAUDE_MODELS.items():
                name, conf = read_one(crop, prompt, model)
                reads[label] = {"model": model, "name": name, "confidence": conf}
            truth = truth_for(gt["cases"], line, field_key) or ""
            print(f"L{line:02d} {field:<5} | "
                  f"{reads['opus']['name']!r:<15} | "
                  f"{reads['sonnet']['name']!r:<15} | "
                  f"{reads['haiku']['name']!r:<15} | {truth!r}",
                  flush=True)
            row["fields"][field] = {"truth": truth, **reads}
        results.append(row)

    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "models": {k: v for k, v in CLAUDE_MODELS.items()},
        "first_prompt": FIRST_PROMPT, "last_prompt": LAST_PROMPT,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
