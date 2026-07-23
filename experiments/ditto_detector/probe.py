#!/usr/bin/env python3
"""Prototype ditto-detector — pixel analysis, NO model calls.

Hypothesis: a ditto surname is a small mark (or nothing) in the right portion
of the name cell, while a real surname is a full word. Dark-pixel density in
the right half of the crop should separate them cleanly.

For each of the 42 rows on frame 0023:
  1. Crop the name cell (same geometry as pipeline_v2 / s21).
  2. Take the RIGHT HALF of that crop (where the surname sits).
  3. Convert to grayscale, count dark pixels (below a threshold).
  4. Predict ditto if dark-pixel ratio < some threshold.

Compare predictions to pipeline_v2's model-derived ditto labels for the same
page. Report:
  - dark-pixel ratio per row
  - pipeline_v2 label (ditto or real)
  - predicted label at several thresholds
  - accuracy at each threshold

Also saves the right-half crops so you can eyeball edge cases.

Run:
    .venv/bin/python experiments/ditto_detector/probe.py
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

# Match pipeline_v2's name-cell crop
NAME_COL_FRAC = (0.15, 0.37)
Y_PAD = 40
# Dark-pixel threshold (0-255). Below this = "dark" (ink). Adjust if scan
# brightness differs by reel.
DARK_THRESHOLD = 200


def surname_region(img: Image.Image, top: int, pitch: int, line: int) -> Image.Image:
    """Crop the surname region of the name cell — right 60% of the column,
    with tight vertical bounds (small Y padding to avoid adjacent-row bleed)."""
    W, H = img.size
    # Tighter vertical crop to reduce bleed from neighboring rows
    ypad = 10
    y0 = max(0, top + (line - 1) * pitch + ypad)
    y1 = min(H, top + line * pitch - ypad)
    x0, x1 = int(NAME_COL_FRAC[0] * W), int(NAME_COL_FRAC[1] * W)
    # Right 60% — skip the first ~40% where the given name lives
    surname_start = x0 + int(0.4 * (x1 - x0))
    return img.crop((surname_start, y0, x1, y1))


def column_ink_profile(cell: Image.Image, thr: int = DARK_THRESHOLD) -> list[int]:
    """Return per-column dark-pixel counts (length = image width)."""
    g = ImageOps.grayscale(cell)
    W, H = g.size
    pixels = g.load()
    return [sum(1 for y in range(H) if pixels[x, y] < thr) for x in range(W)]


def extract_features(cell: Image.Image, thr: int = DARK_THRESHOLD) -> dict:
    """Extract writing-extent features from the surname region.

    Key insight: the census form has printed vertical ruled lines that show
    up as thin columns of dark pixels — every column technically has "some"
    ink because of them. So we require a MEANINGFUL number of dark pixels
    per column (min_col_ink) to count as a "written" column. Real handwriting
    covers many pixels vertically in a column; printed ruled lines cover only
    a few.
    """
    profile = column_ink_profile(cell, thr)
    W = len(profile)
    H = cell.size[1]
    total_dark = sum(profile)
    total_px = W * H
    dark_ratio = total_dark / max(1, total_px)

    # A "written" column has at least this many dark pixels. Ruled lines are
    # typically 1-3 px thick; real writing spans much more vertical space per
    # column. 10% of cell height is a reasonable threshold for a scribe stroke.
    min_col_ink = max(3, int(0.1 * H))
    written_cols = [i for i, c in enumerate(profile) if c >= min_col_ink]
    written_cols_frac = len(written_cols) / max(1, W)
    if written_cols:
        rightmost_ink_frac = written_cols[-1] / max(1, W - 1)
        leftmost_ink_frac = written_cols[0] / max(1, W - 1)
        written_cols_span = (written_cols[-1] - written_cols[0] + 1) / max(1, W)
    else:
        rightmost_ink_frac = 0.0
        leftmost_ink_frac = 1.0
        written_cols_span = 0.0
    return {
        "dark_ratio": dark_ratio,
        "min_col_ink_thr": min_col_ink,
        "rightmost_ink_frac": rightmost_ink_frac,
        "leftmost_ink_frac": leftmost_ink_frac,
        "written_cols_frac": written_cols_frac,
        "written_cols_span": written_cols_span,
    }


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    layout = json.loads(LAYOUT.read_text())
    top, pitch, n_rows = layout["row1_top"], layout["row_pitch"], layout["n_rows"]

    img = Image.open(PAGE)

    # Ground-truth labels from pipeline_v2 output
    pipe = json.loads(PIPELINE_OUT.read_text())
    is_ditto_by_line: dict[int, bool | None] = {}
    for r in pipe["rows"]:
        ln = int(r["line_number"])
        v = r.get("interpreted_last_name")
        if v is None:
            is_ditto_by_line[ln] = None
        else:
            is_ditto_by_line[ln] = (str(v).strip().upper() == "[DITTO]")

    print(f"probe: ditto detector on 42 rows of frame 0023 "
          f"(dark_thr={DARK_THRESHOLD}, surname_region = right 60% of name column, "
          f"±10px vertical pad)")
    print(f"{'ln':>3}  {'ratio':>7}  {'rmost':>6}  {'cols':>6}  {'span':>6}  "
          f"{'pipe':<6}")
    print("-" * 60)
    rows_data: list[dict] = []
    for line in range(1, n_rows + 1):
        cell = surname_region(img, top, pitch, line)
        feats = extract_features(cell)
        dst = SCRATCH / f"L{line:02d}_surname.png"
        cell.save(dst)
        label = is_ditto_by_line.get(line)
        label_str = ("ditto" if label else "real") if label is not None else "?"
        print(f"L{line:02d}  {feats['dark_ratio']:7.4f}  "
              f"{feats['rightmost_ink_frac']:6.3f}  "
              f"{feats['written_cols_frac']:6.3f}  "
              f"{feats['written_cols_span']:6.3f}  {label_str:<6}",
              flush=True)
        rows_data.append({"line": line, **feats,
                          "pipe_ditto": label, "crop": dst.name})

    scoreable = [r for r in rows_data if r["pipe_ditto"] is not None]
    n_ditto = sum(1 for r in scoreable if r["pipe_ditto"])
    n_real  = sum(1 for r in scoreable if not r["pipe_ditto"])
    print(f"\n(scoreable rows: {len(scoreable)} — ditto={n_ditto} real={n_real})")

    def sweep(feature_key: str, thresholds, direction: str = "less"):
        """Sweep thresholds for one feature. direction='less' means ditto if
        feature < threshold; direction='more' means ditto if feature > threshold.
        Print acc/prec/rec at each threshold."""
        print(f"\n=== feature: {feature_key} (ditto if {direction} than thr) ===")
        print(f"{'thr':>7}  {'TP':>3}  {'FP':>3}  {'FN':>3}  {'TN':>3}  "
              f"{'acc':>6}  {'prec':>6}  {'rec':>6}")
        for thr in thresholds:
            tp = fp = fn = tn = 0
            for r in scoreable:
                v = r[feature_key]
                pred = (v < thr) if direction == "less" else (v > thr)
                if pred and r["pipe_ditto"]: tp += 1
                elif pred and not r["pipe_ditto"]: fp += 1
                elif not pred and r["pipe_ditto"]: fn += 1
                else: tn += 1
            acc = (tp + tn) / max(1, len(scoreable))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            print(f"{thr:7.3f}  {tp:>3}  {fp:>3}  {fn:>3}  {tn:>3}  "
                  f"{acc:6.1%}  {prec:6.1%}  {rec:6.1%}")

    sweep("rightmost_ink_frac",
          (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), direction="less")
    sweep("written_cols_frac",
          (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9), direction="less")
    sweep("written_cols_span",
          (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), direction="less")
    sweep("dark_ratio",
          (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15), direction="less")

    (SCRATCH / "results.json").write_text(json.dumps({
        "page": str(PAGE), "threshold_used_for_dark": DARK_THRESHOLD,
        "rows": rows_data,
    }, indent=2))
    print(f"\nsaved crops + results.json → {SCRATCH}")


if __name__ == "__main__":
    main()
