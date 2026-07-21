"""Per-page orchestrator: reads the two per-agent JSON files + page image, and
for each row/field dispatches to the selected strategy on name disagreements.

Kept strategy-agnostic: the strategy is passed in as (crop_fn, adjudicate_fn),
so any file in strategies/ that exports the contract just works. Output shape
mirrors src/adjudicate.py but writes to .adjudicated2.<strategy>.json so it
never collides with the v1–v4 outputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from adjudicate2.common import ROW_FIELDS, norm
from adjudicate2.prompts import FIELD_LABEL, NAME_FIELDS


def _page_png(reel_dir: Path, reel: str, frame: int, scratch: Path) -> Path:
    """Return a PNG path for the page, converting from JP2 on demand."""
    png = reel_dir / f"{reel}_{frame:04d}.png"
    if png.exists():
        return png
    jp2 = reel_dir / f"{reel}_{frame:04d}.jp2"
    out = scratch / f"{reel}_{frame:04d}.png"
    if not out.exists():
        if sys.platform == "darwin":
            cmd = ["sips", "-s", "format", "png", str(jp2), "--out", str(out)]
        else:
            cmd = ["convert", str(jp2), str(out)]
        subprocess.run(cmd, capture_output=True)
    return out


def run_page(corpus: Path, reel: str, frame: int, strategy_name: str,
             crop_fn, adjudicate_fn, lines: list[int] | None = None) -> Path:
    """Adjudicate one page with the given strategy. Returns the output path.

    Reads the two per-agent files at output/rows/<reel>/<stem>.{claude,gemini}.json
    and writes <stem>.adjudicated2.<strategy>.json (or .partial.json when
    lines is a subset).
    """
    out_dir = corpus / "output" / "rows" / reel
    per = {}
    for agent in ("claude", "gemini"):
        p = out_dir / f"{reel}_{frame:04d}.{agent}.json"
        if not p.exists():
            sys.exit(f"missing per-agent file: {p}")
        doc = json.loads(p.read_text())
        per[agent] = {str(r["line_number"]): r for r in doc.get("rows", [])}
    meta = json.loads(
        (out_dir / f"{reel}_{frame:04d}.claude.json").read_text()
    ).get("metadata", {})

    layout = json.loads((corpus / "config" / "layout.json").read_text())
    reel_dir = corpus / "data" / "reels" / reel
    scratch = out_dir / "_adj2"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = _page_png(reel_dir, reel, frame, scratch)
    img = Image.open(page_png)
    stem = f"{reel}_{frame:04d}"

    all_lines = sorted({ln for a in per for ln in per[a]},
                       key=lambda x: int(x) if str(x).isdigit() else 999)
    if lines:
        want = {str(x) for x in lines}
        all_lines = [ln for ln in all_lines if ln in want]

    n_names = sum(
        1 for ln in all_lines if per["claude"].get(ln) and per["gemini"].get(ln)
        for f in NAME_FIELDS
        if norm(per["claude"][ln].get(f)) != norm(per["gemini"][ln].get(f)))
    print(f"[adj2] strategy={strategy_name}: {n_names} name disagreements",
          file=sys.stderr, flush=True)

    done_names = 0
    out_rows, resolutions = [], []
    tally = {"agree": 0, "tier1": 0, "tier2": 0, "tier3": 0,
             "claude-default": 0, "error": 0}
    for ln in all_lines:
        rc, rg = per["claude"].get(ln), per["gemini"].get(ln)
        if rc is None or rg is None:
            present = rc or rg
            row = dict(present); row["confidence"] = "LOW"
            row["_note"] = "row present in only one agent; kept as-is (LOW)"
            out_rows.append(row); tally["claude-default"] += 1
            continue
        row = {"line_number": ln}
        row_conf = "HIGH"
        for field in ROW_FIELDS:
            c0, g0 = rc.get(field), rg.get(field)
            if norm(c0) == norm(g0):
                row[field] = c0
                tally["agree"] += 1
                continue
            if field in NAME_FIELDS:
                crop = crop_fn(img, layout, int(ln), scratch, stem)
                res = adjudicate_fn(crop, FIELD_LABEL[field], c0, g0)
                done_names += 1
                print(f"[adj2] ({done_names}/{n_names}) L{ln} {field[12:]}: "
                      f"{c0!r} vs {g0!r} -> {res['value']!r} "
                      f"[{res['confidence']}/{res['tier']}]",
                      file=sys.stderr, flush=True)
            else:
                res = {"claude": c0, "gemini": g0, "value": c0,
                       "confidence": "LOW", "tier": "claude-default", "rationale": None}
            row[field] = res["value"]
            tally[res["tier"]] = tally.get(res["tier"], 0) + 1
            resolutions.append({"line_number": ln, "field": field, **res})
            if res["confidence"] == "LOW":
                row_conf = "LOW"
            elif res["confidence"] == "MEDIUM" and row_conf != "LOW":
                row_conf = "MEDIUM"
        row["confidence"] = row_conf
        out_rows.append(row)

    result = {"agents": ["claude", "gemini"], "strategy": strategy_name,
              "metadata": meta, "rows": out_rows, "resolutions": resolutions,
              "tally": tally}
    part = ".partial" if lines else ""
    out_path = out_dir / f"{reel}_{frame:04d}.adjudicated2.{strategy_name}{part}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[adj2] wrote {out_path}", file=sys.stderr)
    return out_path
