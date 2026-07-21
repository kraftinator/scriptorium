"""Row / name-cell crop helpers (fresh copy from src/adjudicate.py).

Duplicated so this tree can evolve its own crop shapes (tighter windows, new
zoom levels, cell-level crops for grapheme decomposition, etc.) without
touching v1–v4. Add new crop functions here; strategies reference them by
importing from this module.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def crop_row(img: Image.Image, layout: dict, L: int, scratch: Path, stem: str) -> Path:
    """Full-width row crop (v1/v2/v4 shape)."""
    top, pitch = layout["row1_top"], layout["row_pitch"]
    W, H = img.size
    y0 = max(0, top + (L - 1) * pitch - 40)
    y1 = min(H, top + L * pitch + 40)
    p = scratch / f"{stem}_adj_row{L:02d}.png"
    img.crop((0, y0, W, y1)).save(p)
    return p


def crop_name_cell(img: Image.Image, layout: dict, L: int, scratch: Path, stem: str) -> Path:
    """Tight crop of just the name cell for row L, upscaled 2x.

    The name is a narrow slice of a wide row; a full-row crop loses stroke
    detail after the vision API's downsample. Cropping to the cell and
    upscaling gives the model the resolution to read letter shapes.
    """
    top, pitch = layout["row1_top"], layout["row_pitch"]
    W, H = img.size
    y0 = max(0, top + (L - 1) * pitch - 40)
    y1 = min(H, top + L * pitch + 40)
    f0, f1 = layout.get("name_col_frac", [0.0, 1.0])
    x0, x1 = int(f0 * W), int(f1 * W)
    cell = img.crop((x0, y0, x1, y1))
    cell = cell.resize((cell.width * 2, cell.height * 2), Image.LANCZOS)
    p = scratch / f"{stem}_adj_row{L:02d}.png"
    cell.save(p)
    return p
