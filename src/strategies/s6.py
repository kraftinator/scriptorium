#!/usr/bin/env python3
"""s6 — s5's shape with OpenAI (via codex CLI) replacing Grok.

Three models on the split-name read: Claude + Gemini Flash + OpenAI. No
adjudication; observational.

OpenAI is called through the `codex` CLI so we use the user's ChatGPT
subscription (no OpenAI API key needed). Shells out like our claude_backend.

Run:
    .venv/bin/python src/strategies/s6.py
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

CODEX_TIMEOUT_S = 180

FIRST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name (first name, possibly with a middle "
    "initial), then a surname. Read ONLY the given name portion (including "
    "any middle initial that is part of the given name, e.g. 'Milo J.', not "
    "just 'Milo'). Transcribe the EXACT letters as written — do NOT expand "
    "abbreviations ('Chls' stays 'Chls', not 'Charles' or 'Chas') and do NOT "
    "normalize to a more familiar spelling. Return ONLY a JSON object: "
    '{"name":"<given name only>","confidence":"HIGH|MEDIUM|LOW"} with no '
    "other text."
)
LAST_PROMPT = (
    "This image is the NAME cell of one row on an 1850 U.S. Census page — it "
    "contains a handwritten given name, then a surname. Read ONLY the SURNAME "
    "portion. Transcribe the EXACT letters as written — do NOT normalize to a "
    "more familiar spelling. If the surname is a ditto mark (indicating "
    "'same as the row above'), return the literal string [DITTO]. Return "
    'ONLY a JSON object: {"name":"<surname or [DITTO]>","confidence":"HIGH|'
    'MEDIUM|LOW"} with no other text.'
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


def codex_backend(crop: Path, prompt: str, schema: dict) -> dict:
    """OpenAI via the `codex` CLI — uses the user's ChatGPT subscription, no key.

    codex output is verbose (headers, echoed prompt, response, token usage).
    We rely on backends._strip_fence to extract the JSON object (first `{` to
    last `}`), which tolerates the wrapping.
    """
    result = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
         "-s", "read-only", "-i", str(crop)],
        input=prompt, capture_output=True, text=True, timeout=CODEX_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"codex exit {result.returncode}: {result.stderr[:200]}")
    text = backends._strip_fence(result.stdout.strip())
    if not text:
        raise RuntimeError("codex returned no JSON")
    return json.loads(text)


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
    scratch = Path(__file__).parent / "_s6"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s6: split-name read, THREE models (claude + gemini flash + openai/codex) "
          f"on {len(lines)} rows of {reel} frame {frame:04d}")
    print(f"{'ln':>3} | {'C first':<11} {'C last':<11} | "
          f"{'F first':<11} {'F last':<11} | {'O first':<11} {'O last':<11} | "
          f"{'truth f':<10} {'truth l':<10}")
    print("-" * 132)

    results = []
    for line in lines:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        cf, cf_c = read_one(claude_backend, crop, FIRST_PROMPT)
        cl, cl_c = read_one(claude_backend, crop, LAST_PROMPT)
        ff, ff_c = read_one(gemini_backend, crop, FIRST_PROMPT)
        fl, fl_c = read_one(gemini_backend, crop, LAST_PROMPT)
        of, of_c = read_one(codex_backend,  crop, FIRST_PROMPT)
        ol, ol_c = read_one(codex_backend,  crop, LAST_PROMPT)
        tf = truth_for(gt["cases"], line, "interpreted_first_name") or ""
        tl = truth_for(gt["cases"], line, "interpreted_last_name") or ""
        print(f"L{line:02d} | "
              f"{cf!r:<11} {cl!r:<11} | "
              f"{ff!r:<11} {fl!r:<11} | "
              f"{of!r:<11} {ol!r:<11} | "
              f"{tf!r:<10} {tl!r:<10}", flush=True)
        results.append({
            "line": line, "crop": crop.name,
            "truth": {"first": tf, "last": tl},
            "claude":       {"first": {"name": cf, "confidence": cf_c},
                             "last":  {"name": cl, "confidence": cl_c}},
            "gemini_flash": {"first": {"name": ff, "confidence": ff_c},
                             "last":  {"name": fl, "confidence": fl_c}},
            "openai":       {"first": {"name": of, "confidence": of_c},
                             "last":  {"name": ol, "confidence": ol_c}},
        })
    (scratch / "results.json").write_text(json.dumps({
        "corpus": gt["corpus"], "reel": reel, "frame": frame,
        "models": {"claude": backends.CLAUDE_MODEL,
                   "gemini_flash": backends.GEMINI_MODEL,
                   "openai": "codex/subscription (default model)"},
        "first_prompt": FIRST_PROMPT, "last_prompt": LAST_PROMPT,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
