#!/usr/bin/env python3
"""s5 — split-name read with THREE models: Claude + Gemini Flash + Grok.

Same crop + prompts as s2 (tight name-cell crop, exact-letter first/last
prompts), but adds Grok (xAI) as a genuinely different-vendor third reader.
No adjudication; observational.

Grok call uses xAI's OpenAI-compatible endpoint via Python's built-in
urllib — no new pip dependencies. Requires XAI_API_KEY in .env.

Run:
    .venv/bin/python src/strategies/s5.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from backends import claude_backend, gemini_backend  # noqa: E402
import backends  # noqa: E402  (loads .env at import → XAI_API_KEY available)

GROK_MODEL = "grok-4.5"
GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"
GROK_TIMEOUT_S = 120

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


def grok_backend(crop: Path, prompt: str, schema: dict) -> dict:
    """Local Grok caller — xAI's OpenAI-compatible chat/completions endpoint,
    hit via urllib so we don't need the openai package.

    Requires XAI_API_KEY in os.environ (loaded from .env by backends._load_dotenv
    at import). The schema is described in the prompt (Grok honors
    response_format=json_object but not a raw JSON schema).
    """
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY not in environment")
    b64 = base64.b64encode(crop.read_bytes()).decode("ascii")
    body = {
        "model": GROK_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": "Transcribe this image according to your system instructions. "
                         "Reply with a single JSON object with fields: name (string), "
                         "confidence (HIGH/MEDIUM/LOW)."},
            ]},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        GROK_ENDPOINT, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    last = ""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=GROK_TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            return json.loads(backends._strip_fence(text))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {e.reason}"
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            last = f"{last} {body_text}"
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt); continue
            raise RuntimeError(f"grok call failed: {last}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < 3:
                time.sleep(2 ** attempt); continue
            raise RuntimeError(f"grok call failed after retries: {last}") from e
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"grok returned unexpected shape: {e}") from e


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
    scratch = Path(__file__).parent / "_s5"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s5: split-name read, THREE models (claude + gemini flash + grok) on "
          f"{len(lines)} rows of {reel} frame {frame:04d}")
    print(f"{'ln':>3} | {'C first':<11} {'C last':<11} | "
          f"{'F first':<11} {'F last':<11} | {'K first':<11} {'K last':<11} | "
          f"{'truth f':<10} {'truth l':<10}")
    print("-" * 132)

    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        cf, cf_c = read_one(claude_backend, crop, FIRST_PROMPT)
        cl, cl_c = read_one(claude_backend, crop, LAST_PROMPT)
        ff, ff_c = read_one(gemini_backend, crop, FIRST_PROMPT)
        fl, fl_c = read_one(gemini_backend, crop, LAST_PROMPT)
        kf, kf_c = read_one(grok_backend,   crop, FIRST_PROMPT)
        kl, kl_c = read_one(grok_backend,   crop, LAST_PROMPT)
        tf = truth_for(gt["cases"], line, "interpreted_first_name") or ""
        tl = truth_for(gt["cases"], line, "interpreted_last_name") or ""
        print(f"L{line:02d} | "
              f"{cf!r:<11} {cl!r:<11} | "
              f"{ff!r:<11} {fl!r:<11} | "
              f"{kf!r:<11} {kl!r:<11} | "
              f"{tf!r:<10} {tl!r:<10}", flush=True)
        results.append({
            "line": line, "crop": crop.name,
            "truth": {"first": tf, "last": tl},
            "claude":       {"first": {"name": cf, "confidence": cf_c},
                             "last":  {"name": cl, "confidence": cl_c}},
            "gemini_flash": {"first": {"name": ff, "confidence": ff_c},
                             "last":  {"name": fl, "confidence": fl_c}},
            "grok":         {"first": {"name": kf, "confidence": kf_c},
                             "last":  {"name": kl, "confidence": kl_c}},
        })
    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "models": {"claude": backends.CLAUDE_MODEL,
                   "gemini_flash": backends.GEMINI_MODEL,
                   "grok": GROK_MODEL},
        "first_prompt": FIRST_PROMPT, "last_prompt": LAST_PROMPT,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
