"""Shared helpers for the adjudicate2 tree (fresh copy — no imports from src/).

Duplicates the small utilities the strategies + orchestrator need:
  ROW_FIELDS   — fields compared across agents (mirrors src/reconcile.py)
  norm(v)      — normalize a value for agreement comparison (blanks + dittos collapse)
  _conf(vals)  — derived confidence from a set of voter confidences

Kept minimal on purpose; strategies pick these up by importing from this module.
"""
from __future__ import annotations

ROW_FIELDS = [
    "dwelling_number", "family_number",
    "interpreted_first_name", "interpreted_last_name",
    "age", "sex", "color", "occupation", "real_estate_value", "place_of_birth",
    "married_within_year", "attended_school", "cannot_read_write", "infirmities",
]

_DITTO = {"[ditto]", '"', "'", ",", ",,", "//", "”", "“", "″"}


def norm(v) -> str:
    """Normalize for agreement checks: strip/casefold, collapse ditto+blank forms."""
    if v is None:
        return ""
    s = str(v).strip().casefold()
    return "" if s in _DITTO else s


def _conf(vals) -> str:
    """Derived confidence: LOW if any voter is unsure, HIGH only if all sure."""
    vals = [str(v).upper() for v in vals if v]
    if "LOW" in vals:
        return "LOW"
    if vals and all(v == "HIGH" for v in vals):
        return "HIGH"
    return "MEDIUM"
