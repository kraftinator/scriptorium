#!/usr/bin/env python3
"""s11 — shape-only visual-feature probes, bundled in one call per model.

Ask each model to count VISUAL FEATURES it can see (tall strokes, descenders,
dots, closed bowls at the baseline) — without identifying the name. Purpose:
give us discriminators on the hard cells where name-agreement was ambiguous.

E.g. L33 Gibson has 1 tall stroke + 1 closed bowl (the 'b'); Cisen has 0 of
either. So even if both models produce plausible 6-letter surnames, the
shape-feature counts should be different — and one will match the truth.

Claude + Gemini Flash, split first/last, tight name-cell crop. Bundled: all
four counts returned in one JSON per call. Same call count as s2 (32).

Run:
    .venv/bin/python src/strategies/s11.py
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

_FEATURE_GUIDE = (
    "Count each of these VISUAL FEATURES you can see. Do NOT try to identify "
    "the name — just count strokes and shapes.\n\n"
    "1. tall_strokes: number of letters whose stroke rises ABOVE the x-height "
    "of surrounding letters. Lowercase ascenders (b, d, f, h, k, l, t, or the "
    "archaic long-s ſ) AND any capital letter (A–Z all rise above x-height) "
    "each count.\n"
    "2. descenders: number of letters whose stroke goes BELOW the baseline "
    "(g, j, p, q, y — some styles of f also).\n"
    "3. dots: number of dots visible ABOVE letters (from i or j — or their "
    "cursive equivalents).\n"
    "4. closed_bowls_at_baseline: number of TALL strokes that CLOSE into a "
    "bowl AT THE BASELINE. This distinguishes 'b' (tall stroke that closes "
    "into a baseline bowl) from an archaic long-s ſ (tall stroke that stays "
    "open) and from 'l', 'h', 'k' (tall strokes with no baseline bowl). "
    "Only count bowls attached to tall strokes and closing at the baseline."
)

FIRST_PROMPT = (
    "Look ONLY at the given name portion (first name plus any middle initial) "
    "in this handwritten 1850 U.S. Census name cell.\n\n"
    f"{_FEATURE_GUIDE}\n\n"
    'Return ONLY this JSON: {"tall_strokes": <int>, "descenders": <int>, '
    '"dots": <int>, "closed_bowls_at_baseline": <int>, '
    '"confidence": "HIGH|MEDIUM|LOW"}. No prose outside.'
)

LAST_PROMPT = (
    "Look ONLY at the SURNAME portion of this handwritten 1850 U.S. Census "
    "name cell.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return ONLY: {"tall_strokes": 0, "descenders": 0, "dots": 0, '
    '"closed_bowls_at_baseline": 0, "confidence": "HIGH", "ditto": true} '
    "and skip the rest.\n\n"
    "Otherwise, count these visual features (do NOT try to identify the name):\n\n"
    f"{_FEATURE_GUIDE}\n\n"
    'Return ONLY this JSON: {"tall_strokes": <int>, "descenders": <int>, '
    '"dots": <int>, "closed_bowls_at_baseline": <int>, '
    '"confidence": "HIGH|MEDIUM|LOW"}. No prose outside.'
)

SCHEMA = {"type": "object", "properties": {
    "tall_strokes": {"type": "integer"},
    "descenders": {"type": "integer"},
    "dots": {"type": "integer"},
    "closed_bowls_at_baseline": {"type": "integer"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "ditto": {"type": "boolean"}},
    "required": ["tall_strokes", "descenders", "dots",
                 "closed_bowls_at_baseline", "confidence"]}

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


# Category membership rules used to compute expected feature counts from a
# ground-truth name. Case-folded to lowercase; capital letters ALSO count as
# tall_strokes (they rise above x-height). Approximate — cursive can vary.
_ASCENDER_LOWERS = set("bdfhklt")
_DESCENDER_LOWERS = set("gjpqy")
_DOTTED_LOWERS = set("ij")
_BOWL_LOWERS = set("bdpq")


def expected_features(name: str) -> dict[str, int] | None:
    """Best-effort feature counts from a ground-truth name string.
    Returns None for empty/ditto (no name = no expected counts).
    """
    if not name or name.strip().upper() == "[DITTO]":
        return {"tall_strokes": 0, "descenders": 0,
                "dots": 0, "closed_bowls_at_baseline": 0}
    tall = desc = dots = bowls = 0
    for ch in name:
        if not ch.isalpha():
            continue
        low = ch.lower()
        if ch.isupper() or low in _ASCENDER_LOWERS:
            tall += 1
        if low in _DESCENDER_LOWERS:
            desc += 1
        if low in _DOTTED_LOWERS:
            dots += 1
        if low in _BOWL_LOWERS:
            bowls += 1
    return {"tall_strokes": tall, "descenders": desc,
            "dots": dots, "closed_bowls_at_baseline": bowls}


def truth_for(cases, line, field):
    for c in cases:
        if c["line"] == line and c["field"] == field:
            return c["correct"]
    return None


def read_one(backend, crop: Path, prompt: str) -> dict:
    try:
        r = backend(crop, prompt, SCHEMA)
        return {"tall_strokes": r.get("tall_strokes"),
                "descenders": r.get("descenders"),
                "dots": r.get("dots"),
                "closed_bowls_at_baseline": r.get("closed_bowls_at_baseline"),
                "confidence": r.get("confidence", "?")}
    except Exception as e:
        return {"tall_strokes": None, "descenders": None,
                "dots": None, "closed_bowls_at_baseline": None,
                "confidence": "?",
                "error": f"{type(e).__name__}: {str(e)[:60]}"}


def _fmt(rec: dict) -> str:
    def v(k): return "-" if rec.get(k) is None else str(rec[k])
    return f"T{v('tall_strokes')} D{v('descenders')} d{v('dots')} B{v('closed_bowls_at_baseline')}"


def main() -> None:
    gt = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
    corpus = (REPO / gt["corpus"]).resolve()
    reel, frame = gt["reel"], gt["frame"]
    lines = sorted({c["line"] for c in gt["cases"]})

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s11"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    print(f"s11: shape-feature probes on {len(lines)} rows of {reel} frame {frame:04d}")
    print(f"     T=tall_strokes  D=descenders  d=dots  B=closed_bowls_at_baseline")
    print(f"{'ln':>3} {'fld':<5} | {'Claude':<15} | {'Gemini':<15} | {'truth expected':<20} name")
    print("-" * 92)

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
            expected = expected_features(truth)
            expected_str = _fmt(expected) if expected else "-"
            print(f"L{line:02d} {field:<5} | {_fmt(c):<15} | {_fmt(g):<15} | "
                  f"{expected_str:<20} {truth!r}", flush=True)
            row["fields"][field] = {
                "truth": truth,
                "expected": expected,
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
