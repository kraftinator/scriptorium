#!/usr/bin/env python3
"""Prototype readability-detector for name cells — pure pixel analysis, NO model calls.

Goal: distinguish cells where Fable's LOW-confidence read is likely a
hallucination (crop is too damaged / bleed-heavy for anyone to read reliably)
from cells where Fable's LOW is a real but hard read.

Motivation: on frame 0023 pipeline_v2 output, both L40 and L42 had
  Opus first_name = [ILLEGIBLE]
  Fable first_name = <plausible name> LOW+review
User verified L40's Fable read ("George") is wrong; L42's ("Hannah") is
right (per genealogy). Fable-confidence alone can't distinguish these; maybe
the underlying crop's visual quality can.

For each of the 42 rows, crops the FIRST-NAME half of the name cell (left
~55% of the name column), computes several readability metrics, and prints
them alongside the pipeline_v2 output so you can eyeball where a threshold
should live.

Run:
    .venv/bin/python experiments/readability_detector/probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "corpora/us_census_1850/data/pages/populationschedu0604unix/populationschedu0604unix_0023.png"
PIPELINE_OUT = REPO / "corpora/us_census_1850/output/rows/populationschedu0604unix/populationschedu0604unix_0023.pipeline_v2.json"
LAYOUT = REPO / "corpora/us_census_1850/config/layout.json"
SCRATCH = Path(__file__).parent / "_out"

# Same name-column geometry as pipeline_v2 / s21
NAME_COL_FRAC = (0.15, 0.37)
# Given-name portion sits in the left ~55% of the name column
FIRST_NAME_XFRAC = (0.0, 0.55)
Y_PAD = 10   # tight vertical crop to avoid adjacent-row bleed
DARK_THR = 200


def first_name_crop(img: Image.Image, top: int, pitch: int, line: int) -> Image.Image:
    """Crop the FIRST-NAME half of the name cell for one row."""
    W, H = img.size
    y0 = max(0, top + (line - 1) * pitch + Y_PAD)
    y1 = min(H, top + line * pitch - Y_PAD)
    x0, x1 = int(NAME_COL_FRAC[0] * W), int(NAME_COL_FRAC[1] * W)
    xa, xb = FIRST_NAME_XFRAC
    fx0 = x0 + int(xa * (x1 - x0))
    fx1 = x0 + int(xb * (x1 - x0))
    return img.crop((fx0, y0, fx1, y1))


def readability_features(cell: Image.Image, thr: int = DARK_THR) -> dict:
    """Compute readability metrics for a cell.

    - dark_ratio: fraction of pixels darker than thr (raw ink density)
    - written_cols_frac: fraction of columns with ≥10% vertical dark pixels
                        (filters out printed ruled lines; requires real writing)
    - ink_h_stddev_frac: std deviation of horizontal ink density across cell width,
                       divided by the mean. High = spread-out writing (real name);
                       low = uniform ink (bleed, smudge, background noise) or
                       concentrated ink (small isolated mark).
    - top_col_ink_frac: mean of the top-25% densest columns / cell height. High =
                       there are columns with dense strokes (a real letter). Low
                       = no columns stand out; likely no writing.
    """
    g = ImageOps.grayscale(cell)
    W, H = g.size
    if W == 0 or H == 0:
        return {"dark_ratio": 0.0, "written_cols_frac": 0.0,
                "ink_h_stddev_frac": 0.0, "top_col_ink_frac": 0.0}
    pixels = g.load()
    profile = [sum(1 for y in range(H) if pixels[x, y] < thr) for x in range(W)]
    total_dark = sum(profile)
    total_px = W * H
    dark_ratio = total_dark / total_px

    min_col_ink = max(3, int(0.10 * H))
    written_cols = [i for i, c in enumerate(profile) if c >= min_col_ink]
    written_cols_frac = len(written_cols) / W

    mean = total_dark / W
    variance = sum((c - mean) ** 2 for c in profile) / W
    stddev = variance ** 0.5
    ink_h_stddev_frac = (stddev / mean) if mean > 0 else 0.0

    top_n = max(1, W // 4)
    top_cols = sorted(profile, reverse=True)[:top_n]
    top_col_ink_frac = (sum(top_cols) / top_n) / H

    return {
        "dark_ratio": dark_ratio,
        "written_cols_frac": written_cols_frac,
        "ink_h_stddev_frac": ink_h_stddev_frac,
        "top_col_ink_frac": top_col_ink_frac,
    }


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    layout = json.loads(LAYOUT.read_text())
    top, pitch, n_rows = layout["row1_top"], layout["row_pitch"], layout["n_rows"]
    img = Image.open(PAGE)

    # Pipeline output for context
    pipe = json.loads(PIPELINE_OUT.read_text())
    rows_by_ln = {r["line_number"]: r for r in pipe["rows"]}

    print("probe: readability detector on first-name half of every name cell")
    print(f"{'ln':>3}  {'dark':>6}  {'wcols':>6}  {'stddev':>7}  {'topcol':>7}  "
          f"{'pipeline_first':<18}  {'source':<10}  {'opus_first':<18}")
    print("-" * 110)
    rows_data: list[dict] = []
    for line in range(1, n_rows + 1):
        cell = first_name_crop(img, top, pitch, line)
        feats = readability_features(cell)
        dst = SCRATCH / f"L{line:02d}_firstname.png"
        cell.save(dst)
        r = rows_by_ln.get(str(line), {})
        fn = r.get("interpreted_first_name")
        fm = r.get("_first_name_meta") or {}
        opus_first = fm.get("opus_read", {}).get("name") if fm.get("escalated") else None
        source = (fm.get("model") or "").replace("claude-", "").replace("opus-4-8", "opus")
        if fm.get("escalated"):
            source = "FABLE-esc"
        print(f"L{line:02d}  {feats['dark_ratio']:6.3f}  "
              f"{feats['written_cols_frac']:6.3f}  "
              f"{feats['ink_h_stddev_frac']:7.3f}  "
              f"{feats['top_col_ink_frac']:7.3f}  "
              f"{str(fn or ''):<18}  {source:<10}  {str(opus_first or ''):<18}",
              flush=True)
        rows_data.append({"line": line, **feats,
                          "pipeline_first_name": fn,
                          "pipeline_first_source": source,
                          "opus_first_name": opus_first})

    # Highlight the illegible-escalated cells (Opus said [ILLEGIBLE], Fable filled in)
    esc_illegible = [r for r in rows_data
                     if r["pipeline_first_source"] == "FABLE-esc"
                     and (r["opus_first_name"] or "").strip() == "[ILLEGIBLE]"]
    print()
    print(f"cells where Opus said [ILLEGIBLE] AND Fable escalation filled in "
          f"({len(esc_illegible)} of {len(rows_data)}):")
    for r in esc_illegible:
        marker = "  ← user says WRONG" if r["line"] == 40 else (
            "  ← user says RIGHT (per genealogy)" if r["line"] == 42 else "")
        print(f"  L{r['line']:02d}  dark={r['dark_ratio']:.3f}  "
              f"wcols={r['written_cols_frac']:.3f}  "
              f"stddev={r['ink_h_stddev_frac']:.3f}  "
              f"top={r['top_col_ink_frac']:.3f}  → "
              f"pipeline={r['pipeline_first_name']!r}{marker}")

    (SCRATCH / "results.json").write_text(json.dumps({
        "rows": rows_data,
    }, indent=2))
    print(f"\nsaved crops + results.json → {SCRATCH}")


if __name__ == "__main__":
    main()
