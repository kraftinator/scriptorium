#!/usr/bin/env python3
"""pipeline_v2 — production pipeline per the "final system design" agreed in
strategies s1-s23.

Design (locked):
  1. NON-NAME pass — Opus reads row-bands (6 rows / call, tile.py-style,
     full-width crop). Uses the corpus page schema with the name fields
     REMOVED so the model doesn't waste effort on names in this pass.
     Whatever Opus says stands (single model, no cross-check).

  2. NAME pass — Opus with the s21 calibration rubric (LOW/MEDIUM/HIGH
     definitions + familiar-name red flag + stroke verification + a
     needs_review boolean), split first/last, tight name-cell crop
     (2× upscale). Two calls per row per name field.

  3. ESCALATION — for each name field:
       - if name == "[DITTO]": keep as-is, do NOT escalate.
       - elif confidence == LOW AND needs_review == True: re-read with
         `claude-fable-5` (same crop, same s21 prompt) and take Fable's
         read as final.
       - else: keep Opus's read.

  4. METADATA pass — read the header title-band for town/county/date
     (unchanged from tile.py's META_PROMPT).

Output: <output>/rows/<reel>/<reel>_<frame>.pipeline_v2.json — merged
per-row record ready for downstream ditto resolution + DuckDB ingestion.

No human intervention. No Gemini. Ditto resolution is a separate downstream
step (not in this file).

Usage:
    .venv/bin/python src/pipeline_v2.py --corpus corpora/us_census_1850 \\
        --reel populationschedu0604unix --frame 23
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import claude_backend  # noqa: E402
import backends  # noqa: E402

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
OPUS_MODEL = backends.CLAUDE_MODEL          # "claude-opus-4-8"
FABLE_MODEL = "claude-fable-5"

# ---------------------------------------------------------------------------
# Name-pass prompts — the s21 rubric verbatim
#
# PROMPT-CACHING STRATEGY: the calibration rubric + read-a-name instructions
# are IDENTICAL across all name-pass calls on a page (84 × per-page). Sending
# them as the `--system-prompt` on the `claude` CLI (with
# --exclude-dynamic-system-prompt-sections) makes them stable across
# subprocess invocations, which enables the Anthropic server-side prompt cache
# (5-minute TTL) to hit. The per-call user prompt then only has to reference
# the specific image file — small and unique per call, unaffected by caching.
# ---------------------------------------------------------------------------
_NAME_CALIBRATION_RUBRIC = (
    "CALIBRATION IS CRITICAL. Use the confidence levels as follows:\n"
    "  HIGH   — every letter shape is clearly identifiable; a second human "
    "transcriber would almost certainly agree with your reading.\n"
    "  MEDIUM — 1-2 letters are somewhat ambiguous but you can commit to a "
    "reading (say, 70-90% sure).\n"
    "  LOW    — multiple letters are hard to identify OR the reading is "
    "essentially a guess based on your best interpretation. If you cannot "
    "clearly see each letter, you MUST report LOW.\n\n"
    "TWO ADDITIONAL RULES THAT ALWAYS PUSH TOWARD LOW:\n\n"
    "RULE A — FAMILIAR-NAME RED FLAG: if your reading matches a very common "
    "familiar name (examples: Mary, John, William, Sarah, Elizabeth, James, "
    "Anna, Thomas, Charles, Henry), that is a WARNING SIGN that you may have "
    "jumped to a familiar pattern instead of reading the actual strokes. "
    "When your reading is a very common name, apply EXTRA scrutiny AND "
    "downgrade confidence by at least one level (HIGH→MEDIUM, MEDIUM→LOW).\n\n"
    "RULE B — STROKE VERIFICATION: after choosing your reading, look at the "
    "image AGAIN. For EACH letter in your chosen reading, ask: can I point "
    "to a specific stroke or set of strokes on the page that unambiguously "
    "supports that letter shape? If you CANNOT clearly see the strokes "
    "supporting any letter in your reading, use LOW.\n\n"
    "In addition, return a boolean `needs_review`:\n"
    "  true  — the cell is genuinely ambiguous, you had to guess, or a human "
    "transcriber might reasonably disagree with you. Return true EVEN if "
    "confidence is MEDIUM or HIGH — this is an independent flag for cells "
    "worth a second look.\n"
    "  false — the strokes are clear enough that the reading is unambiguous.\n\n"
    "Do NOT default to MEDIUM to avoid commitment. When strokes are hard to "
    "make out, LOW + needs_review=true is the RIGHT answer, not MEDIUM."
)

# System prompts (stable — cached across calls)
NAME_FIRST_SYSTEM = (
    "You are transcribing handwritten names from 1850 U.S. Census pages. "
    "The image shown to you is the NAME cell of one row — a handwritten given "
    "name (possibly with a middle initial) followed by a surname. On each "
    "call your task is to read ONLY the given name portion (including any "
    "middle initial that is part of the given name, e.g. 'Milo J.').\n\n"
    "Transcribe the EXACT letters as written — do NOT expand abbreviations "
    "('Chls' stays 'Chls', 'Wm.' stays 'Wm.') and do NOT normalize to a more "
    "familiar spelling.\n\n"
    f"{_NAME_CALIBRATION_RUBRIC}\n\n"
    'Return ONLY this JSON: {"name":"<given name>","confidence":'
    '"HIGH|MEDIUM|LOW","needs_review":<bool>}. No prose outside.'
)

NAME_LAST_SYSTEM = (
    "You are transcribing handwritten names from 1850 U.S. Census pages. "
    "The image shown to you is the NAME cell of one row — a handwritten given "
    "name followed by a surname. On each call your task is to read ONLY the "
    "SURNAME portion.\n\n"
    "If the surname is a ditto mark (indicating 'same as the row above'), "
    'return: {"name":"[DITTO]","confidence":"HIGH","needs_review":false}\n\n'
    "Otherwise, transcribe the EXACT letters as written — do NOT normalize.\n\n"
    f"{_NAME_CALIBRATION_RUBRIC}\n\n"
    'Return ONLY this JSON: {"name":"<surname or [DITTO]>","confidence":'
    '"HIGH|MEDIUM|LOW","needs_review":<bool>}. No prose outside.'
)

# Per-call user prompt (tiny — cache miss is cheap here)
NAME_USER_PROMPT = (
    "Read the census name cell image at:\n{image}\n\n"
    'Conform strictly to the JSON schema in your system prompt.'
)

NAME_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "needs_review": {"type": "boolean"}},
    "required": ["name", "confidence", "needs_review"]}

# ---------------------------------------------------------------------------
# Metadata pass — title-band crop for town/county/date (unchanged from tile.py)
# ---------------------------------------------------------------------------
META_PROMPT = (
    "The attached image is the printed TITLE BANNER at the very top of an "
    "1850 U.S. Census population schedule. It reads: 'SCHEDULE 1.—Free "
    "Inhabitants in [town], in the County of [county], State of [state], "
    "enumerated by me on the [day] day of [month], 1850, ... Ass't Marshal.', "
    "with the bracketed parts handwritten. Read those handwritten fills and "
    "return: location_town (just the place name, e.g. 'Barton'), "
    "location_county (e.g. 'Tioga'), and enumeration_date. Set page_number "
    "and assistant_marshal to null — they are not in this crop. Return an "
    "empty rows array. Output strictly valid JSON matching the schema."
)

# ---------------------------------------------------------------------------
# Geometry — cell crop for name pass (matches s21/s22)
# ---------------------------------------------------------------------------
NAME_COL_FRAC = (0.15, 0.37)
Y_PAD_NAME = 40
UPSCALE_NAME = 2

# Fields removed from the row schema for the non-name pass (Opus doesn't read
# these here — the name pass handles them).
NAME_FIELDS = {"raw_name", "interpreted_first_name", "interpreted_last_name"}

# --- Ditto pre-detector (pixel analysis, NO model call) ---
# Validated in experiments/ditto_detector/probe.py on frame 0023 — threshold
# 0.65 gave 25/29 dittos detected with 0 false positives (100% precision).
# The 4 missed dittos fall through to the normal Opus call — no harm done.
DITTO_DARK_THRESHOLD = 200          # per-pixel: <thr counts as dark ink
DITTO_MIN_COL_INK_FRAC = 0.10       # a column is "written" only if ≥10% of its
                                    # vertical extent is dark (filters printed
                                    # ruled lines out)
DITTO_WRITTEN_COL_THRESHOLD = 0.65  # if fraction of "written" columns in the
                                    # surname region is BELOW this → ditto
DITTO_SURNAME_XFRAC = (0.4, 1.0)    # surname sits in right 60% of the name col
DITTO_YPAD = 10                     # tight vertical crop to avoid neighbor
                                    # rows bleeding in


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def convert_frame(reel_dir: Path, pages_dir: Path, reel: str, frame: int) -> Path:
    """JP2 → PNG (cached). Returns the PNG path."""
    stem = f"{reel}_{frame:04d}"
    png = pages_dir / f"{stem}.png"
    if png.exists():
        return png
    jp2 = reel_dir / f"{stem}.jp2"
    if not jp2.exists():
        sys.exit(f"source frame not found: {jp2}")
    pages_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        cmd = ["sips", "-s", "format", "png", str(jp2), "--out", str(png)]
    else:
        cmd = ["convert", str(jp2), str(png)]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return png


def _meta_field_schema(value_type: str | list[str] = "string") -> dict:
    """Generic per-field meta shape: value + confidence + needs_review.
    Used for fields where we want the model to self-flag suspicious reads
    (place_of_birth, age, ...)."""
    vt = value_type if isinstance(value_type, list) else [value_type, "null"]
    return {"type": "object",
            "properties": {
                "value": {"type": vt},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "needs_review": {"type": "boolean"},
            },
            "required": ["value", "confidence", "needs_review"]}


# Backwards-compat alias — some downstream code may reference this name
def _place_of_birth_meta_schema() -> dict:
    return _meta_field_schema("string")


# Non-name fields that use the meta shape (value + confidence + needs_review)
# and get post-pass Fable-5 escalation on LOW + needs_review.
META_FIELDS = ("age", "sex", "place_of_birth")


def transform_schema_for_non_name(page_schema: dict) -> dict:
    """Return a deep copy of page_schema with (a) name fields removed from
    the row schema so the non-name pass doesn't try to read them, and
    (b) META_FIELDS upgraded to the meta shape (value + confidence +
    needs_review) so the model can self-flag suspicious reads."""
    s = copy.deepcopy(page_schema)
    rows_items = s.get("properties", {}).get("rows", {}).get("items", {})
    props = rows_items.get("properties", {})
    for f in NAME_FIELDS:
        props.pop(f, None)
    for f in META_FIELDS:
        if f in props:
            props[f] = _meta_field_schema("string")
    if "required" in rows_items:
        rows_items["required"] = [r for r in rows_items["required"]
                                  if r not in NAME_FIELDS]
    return s


# Kept as a thin alias for backwards compat with anyone importing this fn
def strip_name_fields(page_schema: dict) -> dict:
    return transform_schema_for_non_name(page_schema)


def is_likely_ditto_surname(img: Image.Image, top: int, pitch: int,
                            line: int) -> bool:
    """Pixel-analysis pre-check for the surname region — no model call.

    Returns True if the surname region looks like a ditto mark (or blank);
    False if it looks like handwritten letters. Validated on frame 0023:
    100% precision (never marks a real surname as ditto), 86% recall (catches
    25/29 dittos; missed ones are heavy `"`/`do.` styles that get the normal
    model call anyway).

    See experiments/ditto_detector/probe.py for the tuning.
    """
    W, H = img.size
    y0 = max(0, top + (line - 1) * pitch + DITTO_YPAD)
    y1 = min(H, top + line * pitch - DITTO_YPAD)
    x0, x1 = int(NAME_COL_FRAC[0] * W), int(NAME_COL_FRAC[1] * W)
    xa, xb = DITTO_SURNAME_XFRAC
    sx0 = x0 + int(xa * (x1 - x0))
    sx1 = x0 + int(xb * (x1 - x0))
    cell = img.crop((sx0, y0, sx1, y1))
    g = ImageOps.grayscale(cell)
    cw, ch = g.size
    if cw == 0 or ch == 0:
        return False
    pixels = g.load()
    min_col_ink = max(3, int(DITTO_MIN_COL_INK_FRAC * ch))
    written_cols = sum(
        1 for x in range(cw)
        if sum(1 for y in range(ch) if pixels[x, y] < DITTO_DARK_THRESHOLD) >= min_col_ink
    )
    written_cols_frac = written_cols / cw
    return written_cols_frac < DITTO_WRITTEN_COL_THRESHOLD


def crop_name_cell(img: Image.Image, top: int, pitch: int, line: int,
                   dst: Path) -> Path:
    W, H = img.size
    y0 = max(0, top + (line - 1) * pitch - Y_PAD_NAME)
    y1 = min(H, top + line * pitch + Y_PAD_NAME)
    x0, x1 = int(NAME_COL_FRAC[0] * W), int(NAME_COL_FRAC[1] * W)
    cell = img.crop((x0, y0, x1, y1))
    cell = cell.resize((cell.width * UPSCALE_NAME, cell.height * UPSCALE_NAME),
                       Image.LANCZOS)
    cell.save(dst)
    return dst


def _cached_claude_call(model: str, system_prompt: str, user_prompt: str,
                        schema: dict) -> dict:
    """Call `claude` CLI with --system-prompt + --exclude-dynamic-system-prompt-sections
    so the (long, stable) system prompt is eligible for Anthropic's server-side
    prompt cache (5-min TTL). Same subprocess/retry/timeout shape as
    backends.claude_backend, but structured for repeat-call caching.
    """
    schema_str = json.dumps(schema, separators=(",", ":"))
    system_full = (f"{system_prompt}\n\nSchema (must be followed exactly):\n{schema_str}")
    last = ""
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["claude", "-p", user_prompt,
                 "--model", model,
                 "--system-prompt", system_full,
                 "--exclude-dynamic-system-prompt-sections",
                 "--allowedTools", "Read", "--strict-mcp-config"],
                capture_output=True, text=True,
                timeout=backends.CLAUDE_TIMEOUT_S,
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


def opus_name_read(crop: Path, system_prompt: str, schema: dict) -> dict:
    """Read a name cell with Opus using a cache-friendly system prompt."""
    return _cached_claude_call(
        OPUS_MODEL, system_prompt,
        NAME_USER_PROMPT.format(image=str(crop)), schema)


def fable_name_read(crop: Path, system_prompt: str, schema: dict) -> dict:
    """Escalation read with Fable-5, same cache-friendly shape."""
    return _cached_claude_call(
        FABLE_MODEL, system_prompt,
        NAME_USER_PROMPT.format(image=str(crop)), schema)


# ---------------------------------------------------------------------------
# Non-name pass — tile-style row-band reads with the name fields stripped
# ---------------------------------------------------------------------------
# Addendum injected into the non-name band prompt for self-flagging
# suspicious AGE reads. If the age looks like it could be off (e.g. single
# digit adjacent to what might be a leading digit that's hard to make out —
# real bug: an elderly person's "61" misread as "8"), the model should rate
# confidence LOW and set needs_review=true.
_AGE_RULES = (
    "\n\nSPECIAL FIELD — age: the value should be a whole number 0-110, "
    "or a fraction like '3/12' for infants under 1 year. Values must be "
    "returned as a STRING (e.g. '61', '8', '3/12').\n"
    "Common failure mode: dropping a leading digit — e.g. reading '8' when "
    "the actual number is '18' or '68' or '61'. If a leading digit is at "
    "all ambiguous, or if the strokes could plausibly be a 2-digit number "
    "instead of a 1-digit number, rate confidence LOW and set "
    "needs_review=true.\n"
    "Every row's age must be returned as "
    '{"value": <str|null>, "confidence": "HIGH|MEDIUM|LOW", '
    '"needs_review": <bool>}.'
)

_SEX_RULES = (
    "\n\nSPECIAL FIELD — sex: the value should be 'M' (male) or 'F' (female). "
    "The scribe's M/F glyphs can look similar (both are short capital strokes). "
    "If the stroke is at all ambiguous between M and F, rate confidence LOW "
    "and set needs_review=true. Return null only if the cell is genuinely "
    "empty.\n"
    "Every row's sex must be returned as "
    '{"value": "M"|"F"|null, "confidence": "HIGH|MEDIUM|LOW", '
    '"needs_review": <bool>}.'
)

# Addendum injected into the non-name band prompt: makes the model self-check
# place_of_birth values and rate confidence LOW + needs_review on non-standard
# reads. Downstream escalation re-reads such cells with Fable-5.
_PLACE_OF_BIRTH_RULES = (
    "\n\nSPECIAL FIELD — place_of_birth: the value SHOULD be one of:\n"
    "  - A US state (e.g. 'New York', 'Pennsylvania', 'Vermont'), OR\n"
    "  - A foreign country (e.g. 'Ireland', 'England', 'Germany', 'Scotland', "
    "'Wales', 'France', 'Prussia', 'Canada'), OR\n"
    '  - The literal string "[DITTO]" if the cell is a ditto mark (meaning '
    '"same as the row above" — this is CORRECT and should be rated HIGH '
    "confidence, needs_review=false).\n\n"
    "If your read is NOT one of the above (e.g. a city name like 'Kingston' "
    "or an unclear string), rate confidence LOW and set needs_review=true. "
    "In that case still return your best reading in `value`. For a clean "
    "state/country reading, use HIGH or MEDIUM as appropriate.\n\n"
    "Every row's place_of_birth must be returned as "
    '{"value": <str|null>, "confidence": "HIGH|MEDIUM|LOW", '
    '"needs_review": <bool>}.'
)


def non_name_pass(page_img: Path, corpus: Path, scratch: Path,
                  only_lines: set[int] | None = None) -> dict[str, dict]:
    """Read every row's non-name fields via tiled row-bands with Opus.

    Returns {line_number: row_dict_without_name_fields}. Line numbers are
    anchored to the printed margin numbers (same rule as tile.py).

    place_of_birth is returned as a meta-object {value, confidence,
    needs_review} so downstream can escalate suspicious birthplaces.
    """
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    page_schema = json.loads((corpus / "config" / "page_schema.json").read_text())
    non_name_schema = transform_schema_for_non_name(page_schema)
    band_prompt = (corpus / "config" / "band_prompt.txt").read_text()

    # Same geometry as tile.py
    top = layout["row1_top"]
    pitch = layout["row_pitch"]
    n_rows = layout["n_rows"]
    band_rows = layout["band_rows"]
    margin = layout.get("crop_margin", 0)

    img = Image.open(page_img)
    W, H = img.size
    stem = page_img.stem
    band_dir = scratch / "_bands"
    band_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    for r0 in range(1, n_rows + 1, band_rows):
        r1 = min(r0 + band_rows - 1, n_rows)
        # If only_lines is set, skip bands that don't overlap the target set.
        if only_lines is not None and not any(r0 <= L <= r1 for L in only_lines):
            continue
        y0 = max(0, top + (r0 - 1) * pitch - margin)
        y1 = min(H, top + r1 * pitch + margin)
        bp = band_dir / f"{stem}_band_{r0:02d}_{r1:02d}.png"
        img.crop((0, y0, W, y1)).save(bp)
        prompt = (f"{band_prompt}\n\nThis strip contains the rows numbered "
                  f"{r0} to {r1} in the far-left margin (a thin sliver of the "
                  f"row just above or below may peek in at the edges — "
                  f"ignore it). Do NOT attempt to read the name/surname column "
                  f"(fields raw_name, interpreted_first_name, "
                  f"interpreted_last_name are omitted from the schema)."
                  f"{_AGE_RULES}"
                  f"{_SEX_RULES}"
                  f"{_PLACE_OF_BIRTH_RULES}")
        try:
            got = claude_backend(bp, prompt, non_name_schema).get("rows", [])
        except Exception as e:
            print(f"  [non-name] band {r0}-{r1} failed: {e}", file=sys.stderr)
            got = []
        for row in got:
            try:
                n = int(str(row.get("line_number")).strip())
            except (TypeError, ValueError):
                continue
            if r0 <= n <= r1 and str(n) not in merged:
                row["line_number"] = str(n)
                merged[str(n)] = row
    return merged


# ---------------------------------------------------------------------------
# Name pass — tight cell crop, split first/last, Opus with s21 rubric,
# Fable-5 escalation on (LOW && needs_review && not [DITTO])
# ---------------------------------------------------------------------------
def _read_name_field(crop: Path, system_prompt: str) -> dict:
    """Call Opus with the s21 rubric via cache-friendly system prompt.
    Returns {name, confidence, needs_review, model, escalated}."""
    try:
        r = opus_name_read(crop, system_prompt, NAME_SCHEMA)
        return {"name": r.get("name", "(missing)"),
                "confidence": r.get("confidence", "?"),
                "needs_review": bool(r.get("needs_review", False)),
                "model": OPUS_MODEL, "escalated": False}
    except Exception as e:
        return {"name": f"ERR: {type(e).__name__}: {str(e)[:60]}",
                "confidence": "?", "needs_review": True,
                "model": OPUS_MODEL, "escalated": False}


def _escalate_to_fable(crop: Path, system_prompt: str, opus_read: dict) -> dict:
    """Re-read the cell with Fable-5. Same cache-friendly shape."""
    try:
        r = fable_name_read(crop, system_prompt, NAME_SCHEMA)
        return {"name": r.get("name", "(missing)"),
                "confidence": r.get("confidence", "?"),
                "needs_review": bool(r.get("needs_review", False)),
                "model": FABLE_MODEL, "escalated": True,
                "opus_read": opus_read}
    except Exception as e:
        return {**opus_read, "escalated": True,
                "opus_read": opus_read,
                "escalation_error": f"{type(e).__name__}: {str(e)[:60]}"}


def _should_escalate(read: dict) -> bool:
    """Escalate if: name is NOT [DITTO] AND confidence == LOW AND needs_review == True."""
    if str(read.get("name", "")).strip().upper() == "[DITTO]":
        return False
    return (read.get("confidence") == "LOW"
            and read.get("needs_review") is True)


def name_pass(page_img: Path, corpus: Path, scratch: Path,
              only_lines: set[int] | None = None) -> dict[str, dict]:
    """Read every row's first_name and last_name via tight cell crops.
    Escalates to Fable-5 per the rule. Returns {line_number: {
        first_name: <read>, last_name: <read>}} where <read> is
        {name, confidence, needs_review, model, escalated, [opus_read]}.
    """
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top = layout["row1_top"]
    pitch = layout["row_pitch"]
    n_rows = layout["n_rows"]

    img = Image.open(page_img)
    stem = page_img.stem
    cell_dir = scratch / "_names"
    cell_dir.mkdir(parents=True, exist_ok=True)

    per_row: dict[str, dict] = {}
    ditto_skipped = 0
    for line in range(1, n_rows + 1):
        if only_lines is not None and line not in only_lines:
            continue
        crop = crop_name_cell(img, top, pitch, line, cell_dir / f"{stem}_L{line:02d}.png")
        row_reads = {}

        # --- first_name — always call the model (ditto detector is surname-only) ---
        opus_first = _read_name_field(crop, NAME_FIRST_SYSTEM)
        if _should_escalate(opus_first):
            print(f"  [name] L{line:02d} first_name: Opus "
                  f"{opus_first['name']!r} LOW+review → escalating to Fable-5",
                  file=sys.stderr, flush=True)
            first_final = _escalate_to_fable(crop, NAME_FIRST_SYSTEM, opus_first)
            print(f"  [name] L{line:02d} first_name: Fable → "
                  f"{first_final['name']!r} [{first_final['confidence']}]",
                  file=sys.stderr, flush=True)
        else:
            first_final = opus_first
        row_reads["first_name"] = first_final

        # --- last_name — cheap pixel pre-check for ditto marks ---
        if is_likely_ditto_surname(img, top, pitch, line):
            row_reads["last_name"] = {
                "name": "[DITTO]", "confidence": "HIGH", "needs_review": False,
                "model": "pixel_detector", "escalated": False,
            }
            ditto_skipped += 1
            print(f"  [name] L{line:02d} last_name: pixel-detected [DITTO] "
                  f"(no model call)", file=sys.stderr, flush=True)
        else:
            opus_last = _read_name_field(crop, NAME_LAST_SYSTEM)
            if _should_escalate(opus_last):
                print(f"  [name] L{line:02d} last_name: Opus "
                      f"{opus_last['name']!r} LOW+review → escalating to Fable-5",
                      file=sys.stderr, flush=True)
                last_final = _escalate_to_fable(crop, NAME_LAST_SYSTEM, opus_last)
                print(f"  [name] L{line:02d} last_name: Fable → "
                      f"{last_final['name']!r} [{last_final['confidence']}]",
                      file=sys.stderr, flush=True)
            else:
                last_final = opus_last
            row_reads["last_name"] = last_final

        per_row[str(line)] = row_reads
        print(f"  [name] L{line:02d}: first={row_reads['first_name']['name']!r} "
              f"last={row_reads['last_name']['name']!r}",
              file=sys.stderr, flush=True)
    print(f"  [name] pixel-detector skipped {ditto_skipped} surname model calls",
          file=sys.stderr, flush=True)
    return per_row


# ---------------------------------------------------------------------------
# age escalation — same LOW+needs_review rule as pob. Full-row crop, targeted
# prompt. Skip escalation if Opus's value is null (nothing to escalate).
# ---------------------------------------------------------------------------
_AGE_ESCALATION_SYSTEM = (
    "You are re-reading ONE field on an 1850 U.S. Census page row: age "
    "(column 4). Look at the full-row image and read ONLY the age value. "
    "The value is typically a whole number 0-110 (returned as a string), "
    "or a fraction like '3/12' for infants under 1 year. Be especially "
    "careful about LEADING DIGITS — a value written as '61' can easily be "
    "misread as '8' if the leading '6' is faint or blends with adjacent ink. "
    "If the strokes are ambiguous, use LOW + needs_review=true.\n"
    'Return ONLY this JSON: {"value":"<age>","confidence":'
    '"HIGH|MEDIUM|LOW","needs_review":<bool>}.'
)
_AGE_ESCALATION_USER = (
    "Read the age (column 4) from the census row image at:\n{image}"
)

AGE_ESCALATION_SCHEMA = _meta_field_schema("string")


def escalate_age(page_img: Path, corpus: Path, scratch: Path,
                 non_names: dict[str, dict],
                 only_lines: set[int] | None = None) -> tuple[dict[str, dict], int]:
    """Same shape as escalate_place_of_birth but targeting the age field."""
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]
    img = Image.open(page_img)
    row_dir = scratch / "_age_rows"
    row_dir.mkdir(parents=True, exist_ok=True)

    escalations = 0
    for ln, row in non_names.items():
        if only_lines is not None and int(ln) not in only_lines:
            continue
        age = row.get("age")
        if not isinstance(age, dict):
            continue
        # Escalate whenever Opus flagged LOW+review, including null-age cases.
        # (Unlike pob, a null age is essentially always wrong — every person
        # has an age. Fable might succeed where Opus punted.)
        if not (age.get("confidence") == "LOW"
                and age.get("needs_review") is True):
            continue
        v = age.get("value")
        crop = _crop_full_row(img, top, pitch, int(ln),
                              row_dir / f"L{int(ln):02d}_fullrow.png")
        print(f"  [age] L{int(ln):02d}: Opus {v!r} LOW+review → "
              f"escalating to Fable-5", file=sys.stderr, flush=True)
        try:
            fable_read = _cached_claude_call(
                FABLE_MODEL, _AGE_ESCALATION_SYSTEM,
                _AGE_ESCALATION_USER.format(image=str(crop)),
                AGE_ESCALATION_SCHEMA,
            )
            new_age = {
                "value": fable_read.get("value"),
                "confidence": fable_read.get("confidence", "?"),
                "needs_review": bool(fable_read.get("needs_review", False)),
                "model": FABLE_MODEL, "escalated": True,
                "opus_read": age,
            }
            print(f"  [age] L{int(ln):02d}: Fable → {new_age['value']!r} "
                  f"[{new_age['confidence']}]", file=sys.stderr, flush=True)
        except Exception as e:
            new_age = {**age, "model": OPUS_MODEL, "escalated": True,
                       "escalation_error": f"{type(e).__name__}: {str(e)[:60]}"}
        row["age"] = new_age
        escalations += 1
    return non_names, escalations


# ---------------------------------------------------------------------------
# sex escalation — same LOW+needs_review rule. Value must be 'M' or 'F' (or
# null if genuinely blank). Escalate even on null since sex is nearly always
# recorded.
# ---------------------------------------------------------------------------
_SEX_ESCALATION_SYSTEM = (
    "You are re-reading ONE field on an 1850 U.S. Census page row: sex "
    "(column 5). Look at the full-row image and return 'M' (male), 'F' "
    "(female), or null if the cell is genuinely empty. The scribe's M/F "
    "glyphs can look similar — if the stroke is at all ambiguous, use LOW "
    "+ needs_review=true.\n"
    'Return ONLY this JSON: {"value":"M"|"F"|null,"confidence":'
    '"HIGH|MEDIUM|LOW","needs_review":<bool>}.'
)
_SEX_ESCALATION_USER = (
    "Read the sex (column 5) from the census row image at:\n{image}"
)

SEX_ESCALATION_SCHEMA = _meta_field_schema("string")


def escalate_sex(page_img: Path, corpus: Path, scratch: Path,
                 non_names: dict[str, dict],
                 only_lines: set[int] | None = None) -> tuple[dict[str, dict], int]:
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]
    img = Image.open(page_img)
    row_dir = scratch / "_sex_rows"
    row_dir.mkdir(parents=True, exist_ok=True)

    escalations = 0
    for ln, row in non_names.items():
        if only_lines is not None and int(ln) not in only_lines:
            continue
        sx = row.get("sex")
        if not isinstance(sx, dict):
            continue
        if not (sx.get("confidence") == "LOW"
                and sx.get("needs_review") is True):
            continue
        v = sx.get("value")
        crop = _crop_full_row(img, top, pitch, int(ln),
                              row_dir / f"L{int(ln):02d}_fullrow.png")
        print(f"  [sex] L{int(ln):02d}: Opus {v!r} LOW+review → "
              f"escalating to Fable-5", file=sys.stderr, flush=True)
        try:
            fable_read = _cached_claude_call(
                FABLE_MODEL, _SEX_ESCALATION_SYSTEM,
                _SEX_ESCALATION_USER.format(image=str(crop)),
                SEX_ESCALATION_SCHEMA,
            )
            new_sx = {
                "value": fable_read.get("value"),
                "confidence": fable_read.get("confidence", "?"),
                "needs_review": bool(fable_read.get("needs_review", False)),
                "model": FABLE_MODEL, "escalated": True,
                "opus_read": sx,
            }
            print(f"  [sex] L{int(ln):02d}: Fable → {new_sx['value']!r} "
                  f"[{new_sx['confidence']}]", file=sys.stderr, flush=True)
        except Exception as e:
            new_sx = {**sx, "model": OPUS_MODEL, "escalated": True,
                      "escalation_error": f"{type(e).__name__}: {str(e)[:60]}"}
        row["sex"] = new_sx
        escalations += 1
    return non_names, escalations


# ---------------------------------------------------------------------------
# place_of_birth escalation — same LOW+needs_review rule as names, re-read
# with Fable-5 on a full-row crop. The place_of_birth column doesn't have its
# own tight geometry defined yet, so we hand Fable the whole row and ask for
# just that field.
# ---------------------------------------------------------------------------
_PLACE_OF_BIRTH_ESCALATION_SYSTEM = (
    "You are re-reading ONE field on an 1850 U.S. Census page row: "
    "place_of_birth (column 9). Look at the full-row image and read ONLY "
    "the birthplace value. The value must be a US state (e.g. 'New York', "
    "'Pennsylvania') or a foreign country (e.g. 'Ireland', 'England', "
    "'Germany', 'Scotland', 'Prussia'). If the strokes clearly show a valid "
    "state/country, return it as the value with HIGH or MEDIUM confidence. "
    "If the strokes are ambiguous or the reading isn't clearly a state or "
    "country, transcribe your best reading and rate LOW + needs_review=true.\n"
    'Return ONLY this JSON: {"value":"<state/country>","confidence":'
    '"HIGH|MEDIUM|LOW","needs_review":<bool>}.'
)
_PLACE_OF_BIRTH_ESCALATION_USER = (
    "Read the place_of_birth (column 9) from the census row image at:\n{image}"
)

PLACE_OF_BIRTH_SCHEMA = _place_of_birth_meta_schema()


def _crop_full_row(img: Image.Image, top: int, pitch: int, line: int,
                   dst: Path) -> Path:
    W, H = img.size
    y0 = max(0, top + (line - 1) * pitch - 40)
    y1 = min(H, top + line * pitch + 40)
    row = img.crop((0, y0, W, y1))
    row.save(dst)
    return dst


def escalate_place_of_birth(page_img: Path, corpus: Path, scratch: Path,
                            non_names: dict[str, dict],
                            only_lines: set[int] | None = None) -> tuple[dict[str, dict], int]:
    """For every row whose place_of_birth came back LOW + needs_review, re-read
    with Fable-5 on a full-row crop. Mutates non_names in place; returns it
    and the number of escalations performed."""
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    top, pitch = layout["row1_top"], layout["row_pitch"]
    img = Image.open(page_img)
    row_dir = scratch / "_pob_rows"
    row_dir.mkdir(parents=True, exist_ok=True)

    escalations = 0
    for ln, row in non_names.items():
        if only_lines is not None and int(ln) not in only_lines:
            continue
        pob = row.get("place_of_birth")
        if not isinstance(pob, dict):
            continue  # model didn't honor the meta schema — skip
        if not (pob.get("confidence") == "LOW"
                and pob.get("needs_review") is True):
            continue
        # [DITTO] is a valid pob value (means "same as row above"). The
        # non-name prompt's "should be a state or country" rule flags it as
        # non-standard even though it's legitimate — don't waste a Fable call.
        if str(pob.get("value") or "").strip().upper() == "[DITTO]":
            continue
        # If Opus itself returned null (or empty string), trust it — there's
        # nothing to read. Escalating to Fable on nothing produces
        # confabulation (L40 "Connecticut" invented from a blank cell).
        # The null flows through as-is (already LOW+review flagged).
        v = pob.get("value")
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        crop = _crop_full_row(img, top, pitch, int(ln),
                              row_dir / f"L{int(ln):02d}_fullrow.png")
        print(f"  [pob] L{int(ln):02d}: Opus {pob.get('value')!r} LOW+review → "
              f"escalating to Fable-5", file=sys.stderr, flush=True)
        try:
            fable_read = _cached_claude_call(
                FABLE_MODEL, _PLACE_OF_BIRTH_ESCALATION_SYSTEM,
                _PLACE_OF_BIRTH_ESCALATION_USER.format(image=str(crop)),
                PLACE_OF_BIRTH_SCHEMA,
            )
            new_pob = {
                "value": fable_read.get("value"),
                "confidence": fable_read.get("confidence", "?"),
                "needs_review": bool(fable_read.get("needs_review", False)),
                "model": FABLE_MODEL, "escalated": True,
                "opus_read": pob,
            }
            print(f"  [pob] L{int(ln):02d}: Fable → {new_pob['value']!r} "
                  f"[{new_pob['confidence']}]", file=sys.stderr, flush=True)
        except Exception as e:
            new_pob = {**pob, "model": OPUS_MODEL, "escalated": True,
                       "escalation_error": f"{type(e).__name__}: {str(e)[:60]}"}
        row["place_of_birth"] = new_pob
        escalations += 1
    return non_names, escalations


# ---------------------------------------------------------------------------
# Metadata pass — title band crop
# ---------------------------------------------------------------------------
def metadata_pass(page_img: Path, corpus: Path, scratch: Path) -> dict:
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    page_schema = json.loads((corpus / "config" / "page_schema.json").read_text())
    header_top = layout.get("header_top", 0)
    header_bottom = layout.get("header_bottom", layout["row1_top"])
    img = Image.open(page_img)
    W, _ = img.size
    hp = scratch / f"{page_img.stem}_header.png"
    img.crop((0, header_top, W, header_bottom)).save(hp)
    try:
        return claude_backend(hp, META_PROMPT, page_schema).get("metadata", {})
    except Exception as e:
        print(f"  [meta] header failed: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Merge + write output
# ---------------------------------------------------------------------------
def merge(non_names: dict[str, dict], names: dict[str, dict],
          layout_n_rows: int) -> list[dict]:
    """Combine per-row records. Rows without a non-name read still get their
    name-pass values (and vice-versa). Line-number ordered."""
    lines = sorted({*non_names, *names}, key=lambda x: int(x))
    out = []
    for ln in lines:
        nn = non_names.get(ln, {"line_number": ln})
        nm = names.get(ln, {})
        row = dict(nn)
        row["line_number"] = ln
        # Flatten meta-field shapes into value + _meta (parallel to how
        # name fields work below)
        for meta_field in META_FIELDS:
            v = row.get(meta_field)
            if isinstance(v, dict) and "value" in v:
                row[meta_field] = v["value"]
                row[f"_{meta_field}_meta"] = {
                    k: val for k, val in v.items() if k != "value"}
        if "first_name" in nm:
            row["interpreted_first_name"] = nm["first_name"]["name"]
            row["_first_name_meta"] = {
                k: v for k, v in nm["first_name"].items() if k != "name"}
        if "last_name" in nm:
            row["interpreted_last_name"] = nm["last_name"]["name"]
            row["_last_name_meta"] = {
                k: v for k, v in nm["last_name"].items() if k != "name"}
        out.append(row)
    # ensure every printed row exists in the output (missing bands = empty row)
    have = {r["line_number"] for r in out}
    for i in range(1, layout_n_rows + 1):
        if str(i) not in have:
            out.append({"line_number": str(i), "_missing": True})
    out.sort(key=lambda r: int(r["line_number"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pipeline_v2: Opus non-name bands + Opus split-name cells "
                    "(s21 rubric) + Fable-5 escalation on LOW+review "
                    "(non-DITTO only). No Gemini, no human review.")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--reel", required=True)
    ap.add_argument("--frame", required=True, type=int)
    ap.add_argument("--lines", nargs="*", type=int, default=None,
                    help="only process these line numbers (for cheap testing)")
    args = ap.parse_args()
    only_lines = set(args.lines) if args.lines else None

    corpus = args.corpus.resolve()
    reel_dir = corpus / "data" / "reels" / args.reel
    pages_dir = corpus / "data" / "pages" / args.reel
    out_dir = corpus / "output" / "rows" / args.reel
    scratch = out_dir / "_pipeline_v2"
    scratch.mkdir(parents=True, exist_ok=True)

    page_img = convert_frame(reel_dir, pages_dir, args.reel, args.frame)
    print(f"[pipeline_v2] page: {page_img}", file=sys.stderr, flush=True)

    if only_lines:
        print(f"[pipeline_v2] === metadata pass SKIPPED (--lines mode) ===",
              file=sys.stderr, flush=True)
        metadata = {}
    else:
        print(f"[pipeline_v2] === metadata pass ===", file=sys.stderr, flush=True)
        metadata = metadata_pass(page_img, corpus, scratch)
        print(f"[pipeline_v2] metadata: {metadata}", file=sys.stderr, flush=True)

    print(f"[pipeline_v2] === non-name pass (Opus, row-bands) ==="
          + (f" [lines={sorted(only_lines)}]" if only_lines else ""),
          file=sys.stderr, flush=True)
    non_names = non_name_pass(page_img, corpus, scratch, only_lines)
    print(f"[pipeline_v2] non-name pass: read {len(non_names)} rows",
          file=sys.stderr, flush=True)

    print(f"[pipeline_v2] === age escalation (LOW+review → Fable-5) ===",
          file=sys.stderr, flush=True)
    non_names, age_esc = escalate_age(page_img, corpus, scratch, non_names, only_lines)
    print(f"[pipeline_v2] age: {age_esc} escalations to Fable-5",
          file=sys.stderr, flush=True)

    print(f"[pipeline_v2] === sex escalation (LOW+review → Fable-5) ===",
          file=sys.stderr, flush=True)
    non_names, sex_esc = escalate_sex(page_img, corpus, scratch, non_names, only_lines)
    print(f"[pipeline_v2] sex: {sex_esc} escalations to Fable-5",
          file=sys.stderr, flush=True)

    print(f"[pipeline_v2] === place_of_birth escalation (LOW+review → Fable-5) ===",
          file=sys.stderr, flush=True)
    non_names, pob_esc = escalate_place_of_birth(page_img, corpus, scratch, non_names, only_lines)
    print(f"[pipeline_v2] place_of_birth: {pob_esc} escalations to Fable-5",
          file=sys.stderr, flush=True)

    print(f"[pipeline_v2] === name pass (Opus + Fable-5 escalation) ==="
          + (f" [lines={sorted(only_lines)}]" if only_lines else ""),
          file=sys.stderr, flush=True)
    names = name_pass(page_img, corpus, scratch, only_lines)
    escalated = sum(1 for r in names.values()
                    for f in r.values() if f.get("escalated"))
    print(f"[pipeline_v2] name pass: read {len(names)} rows, "
          f"{escalated} name cells escalated to Fable-5",
          file=sys.stderr, flush=True)

    layout = json.loads((corpus / "config" / "layout.json").read_text())
    rows = merge(non_names, names, layout["n_rows"])
    result = {
        "pipeline": "v2",
        "models": {"primary": OPUS_MODEL, "name_escalation": FABLE_MODEL},
        "reel": args.reel, "frame": args.frame,
        "metadata": metadata, "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".partial.L{'-'.join(f'{L:02d}' for L in sorted(only_lines))}" if only_lines else ""
    out_path = out_dir / f"{args.reel}_{args.frame:04d}.pipeline_v2{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[pipeline_v2] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
