#!/usr/bin/env python3
"""s3 — neutral cross-model nudge on the s2 disagreements. Observational, no winner.

For every cell where s2 produced a disagreement between Claude and Gemini,
re-ask each model — telling them ONLY that another transcriber read it a
certain way (the *other* model's s2 answer) — and record their re-read. The
framing is deliberately neutral: "another transcriber read this as X, look
again" — not "X is correct" — to avoid priming a confident correct read to
flip. No adjudication, no winner selection; s3 just SHOWS how each model's
answer moves when it sees the other's.

Standalone: owns its prompts, crop shape, output shape. Imports only raw
model I/O from backends.py.

Run:
    .venv/bin/python src/strategies/s3.py
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


def first_nudge(other_read: str) -> str:
    return (
        "This image is the NAME cell of one row on an 1850 U.S. Census page — "
        "a handwritten given name (first name, possibly with a middle initial) "
        f"followed by a surname. Another transcriber read the given name as "
        f"'{other_read}'. Look again at the image and report YOUR reading of "
        "the given name (including any middle initial that is part of it). "
        "You are free to agree with the other transcriber, keep a different "
        "reading, or read it as something else entirely — judge by the exact "
        "letters written, not by what makes a familiar name. Do NOT expand "
        "abbreviations ('Chls' stays 'Chls'). Return JSON: name (the given "
        "name only), confidence (HIGH/MEDIUM/LOW). No prose outside the JSON."
    )


def last_nudge(other_read: str) -> str:
    return (
        "This image is the NAME cell of one row on an 1850 U.S. Census page — "
        "a handwritten given name followed by a surname. Another transcriber "
        f"read the SURNAME as '{other_read}'. Look again at the image and "
        "report YOUR reading of the surname. You are free to agree with the "
        "other transcriber, keep a different reading, or read it as something "
        "else entirely — judge by the exact letters written, not by what "
        "makes a familiar name. If the surname is a ditto mark, return the "
        "literal string [DITTO]. Return JSON: name, confidence "
        "(HIGH/MEDIUM/LOW). No prose outside the JSON."
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


def _norm(v: str) -> str:
    """Whitespace-insensitive, casefold, drop [DITTO] — matches viewer's _match."""
    s = (v or "").casefold().replace("[ditto]", "")
    return "".join(s.split())


def read_one(backend, crop: Path, prompt: str) -> tuple[str, str]:
    try:
        r = backend(crop, prompt, SCHEMA)
        return r.get("name", "(missing)"), r.get("confidence", "?")
    except Exception as e:
        return f"ERR: {type(e).__name__}", "?"


def main() -> None:
    s2_path = Path(__file__).parent / "_s2" / "results.json"
    if not s2_path.exists():
        sys.exit(f"missing {s2_path} — run src/strategies/s2.py first")
    s2 = json.loads(s2_path.read_text())
    corpus = (REPO / s2["corpus"]).resolve()
    reel, frame = s2["reel"], s2["frame"]

    reel_dir = corpus / "data" / "reels" / reel
    scratch = Path(__file__).parent / "_s3"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]

    disagreements = []
    for r in s2["results"]:
        cf, cl = r["claude"]["first"]["name"], r["claude"]["last"]["name"]
        gf, gl = r["gemini"]["first"]["name"], r["gemini"]["last"]["name"]
        if _norm(cf) != _norm(gf):
            disagreements.append((r["line"], "first", cf, gf))
        if _norm(cl) != _norm(gl):
            disagreements.append((r["line"], "last", cl, gl))

    print(f"s3: nudge each model with the other's answer — {len(disagreements)} "
          f"disagreements from s2 on {reel} frame {frame:04d}")
    print(f"{'line':>4} {'fld':<5} {'claude→(saw)':<24} {'→ claude re-read':<24} "
          f"{'gemini→(saw)':<24} {'→ gemini re-read':<24}")
    print("-" * 132)

    results = []
    for line, field, cand_c, cand_g in disagreements:
        crop = crop_name_cell(img, top, pitch, line, scratch / f"L{line:02d}.png")
        nudge = first_nudge if field == "first" else last_nudge
        # Claude sees Gemini's answer:
        c_new, c_conf = read_one(claude_backend, crop, nudge(cand_g))
        # Gemini sees Claude's answer:
        g_new, g_conf = read_one(gemini_backend, crop, nudge(cand_c))
        # A tiny observation: did each side FLIP toward the other, KEEP, or DRIFT to something new?
        def move(orig, new, saw):
            n_o, n_n, n_s = _norm(orig), _norm(new), _norm(saw)
            if n_n == n_o: return "kept"
            if n_n == n_s: return "flipped"
            return "drifted"
        c_move = move(cand_c, c_new, cand_g)
        g_move = move(cand_g, g_new, cand_c)
        print(f"L{line:02d}  {field:<5} "
              f"{cand_c!r:<12}→({cand_g!r:<8}) {(c_new + ' [' + c_move + ']')!r:<24} "
              f"{cand_g!r:<12}→({cand_c!r:<8}) {(g_new + ' [' + g_move + ']')!r:<24}",
              flush=True)
        results.append({
            "line": line, "field": field, "crop": crop.name,
            "claude": {"orig": cand_c, "saw": cand_g,
                       "new": {"name": c_new, "confidence": c_conf}, "move": c_move},
            "gemini": {"orig": cand_g, "saw": cand_c,
                       "new": {"name": g_new, "confidence": g_conf}, "move": g_move},
        })

    (scratch / "results.json").write_text(json.dumps({
        "corpus": s2["corpus"], "reel": reel, "frame": frame,
        "first_nudge_template": first_nudge("<other-read>"),
        "last_nudge_template": last_nudge("<other-read>"),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
