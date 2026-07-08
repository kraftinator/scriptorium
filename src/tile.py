"""Tiled transcription: slice a page into row-bands, transcribe each, and stitch.

Why: the vision APIs downsample a full ~6600x8400 page so far that fine strokes
blur and the 42 tightly-spaced rows can't be resolved (proven on the Barton
page — full page reads "Snickaback", a row-band reads "Knickerbocker"). Slicing
into bands restores per-row resolution.

Line numbers are assigned by GEOMETRY: we know each band's pixel range and thus
which physical rows it covers, so the model never has to read or count line
numbers (which it does unreliably). The model just reads names top-to-bottom;
we label the rows by position.

Backend-agnostic: uses the same REGISTRY adapters as the whole-page pipeline, so
tiling works with any model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import REGISTRY  # noqa: E402

META_PROMPT = (
    "The attached image is the printed HEADER of an 1850 U.S. Census population "
    "schedule (the top banner, above the data rows). Read only the header and "
    "return the metadata fields: page_number, location_county, location_town, "
    "enumeration_date, assistant_marshal. Return an empty rows array. Output "
    "strictly valid JSON matching the schema."
)


def transcribe_tiled(page_img: Path, agent: str, corpus: Path, scratch: Path) -> dict:
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    schema = json.loads((corpus / "config" / "page_schema.json").read_text())
    band_prompt = (corpus / "config" / "band_prompt.txt").read_text()
    backend = REGISTRY[agent]

    top = layout["row1_top"]
    pitch = layout["row_pitch"]
    n_rows = layout["n_rows"]
    band_rows = layout["band_rows"]
    margin = layout.get("crop_margin", 0)

    img = Image.open(page_img)
    W, H = img.size
    stem = page_img.stem
    scratch.mkdir(parents=True, exist_ok=True)

    # --- metadata from the header strip (higher res than the full page) ---
    try:
        hp = scratch / f"{stem}_header.png"
        img.crop((0, 0, W, top)).save(hp)
        metadata = backend(hp, META_PROMPT, schema).get("metadata", {})
    except Exception as e:  # metadata is non-fatal; rows are the point
        print(f"  [tile] header/metadata failed: {e}", file=sys.stderr)
        metadata = {}

    # --- row bands; line numbers anchored to the printed margin numbers,
    # validated against each band's expected range so a stray sliver row (whose
    # number falls outside the band) is filtered instead of shifting everything.
    merged: dict[str, dict] = {}
    for r0 in range(1, n_rows + 1, band_rows):
        r1 = min(r0 + band_rows - 1, n_rows)
        y0 = max(0, top + (r0 - 1) * pitch - margin)
        y1 = min(H, top + r1 * pitch + margin)
        bp = scratch / f"{stem}_band_{r0:02d}_{r1:02d}.png"
        img.crop((0, y0, W, y1)).save(bp)
        prompt = (f"{band_prompt}\n\nThis strip contains the rows numbered {r0} "
                  f"to {r1} in the far-left margin (a thin sliver of the row "
                  f"just above or below may peek in at the edges — ignore it).")
        try:
            got = backend(bp, prompt, schema).get("rows", [])
        except Exception as e:
            print(f"  [tile] band {r0}-{r1} failed: {e}", file=sys.stderr)
            got = []
        for row in got:
            try:
                n = int(str(row.get("line_number")).strip())
            except (TypeError, ValueError):
                continue
            if r0 <= n <= r1 and str(n) not in merged:  # in-band core; first wins
                row["line_number"] = str(n)
                merged[str(n)] = row

    missing = [str(i) for i in range(1, n_rows + 1) if str(i) not in merged]
    if missing:
        print(f"  [tile] missing lines: {', '.join(missing)}", file=sys.stderr)
    rows = [merged[str(i)] for i in range(1, n_rows + 1) if str(i) in merged]
    return {"metadata": metadata, "rows": rows}
