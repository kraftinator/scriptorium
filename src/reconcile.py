#!/usr/bin/env python3
"""Reconcile two per-agent transcriptions into one consensus page.

A separate, optional downstream step. It does NOT call any model — it reads the
per-agent JSON files the pipeline already produced and merges them:

  - a field where both agents agree  -> that value, kept in place
  - a field where they disagree      -> value set null, both readings recorded
                                        under "conflicts", row flagged REVIEW

Agreement is the confidence signal: agreements are trustworthy (two independent
readers landed on the same thing); disagreements are the human-review pile.

Usage:
    python src/reconcile.py --corpus corpora/us_census_1850 \
        --reel populationschedu0604unix --frame 23 --agents claude gemini
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Data fields compared across agents (per-model meta like confidence_score /
# transcription_notes is not reconciled — it's each model's own commentary).
# raw_name is intentionally excluded: it's "first + last" concatenated, so it
# double-counts any name disagreement already captured by the interpreted fields.
ROW_FIELDS = [
    "dwelling_number", "family_number",
    "interpreted_first_name", "interpreted_last_name",
    "age", "sex", "color", "occupation", "real_estate_value", "place_of_birth",
    "married_within_year", "attended_school", "cannot_read_write", "infirmities",
]
META_FIELDS = [
    "page_number", "location_county", "location_town",
    "enumeration_date", "assistant_marshal",
]


# Ditto marks and blanks all mean "same as the row above" — collapse them to one
# equivalence class so a cell one model wrote "[DITTO]" and the other left blank
# (or wrote a raw " mark) counts as agreement, not a conflict.
_DITTO = {"[ditto]", '"', "'", ",", ",,", "//", "”", "“", "″"}


def norm(v):
    """Normalize a value for agreement comparison (empty forms + ditto marks match)."""
    if v is None:
        return ""
    s = str(v).strip().casefold()
    return "" if s in _DITTO else s


def reconcile_field(values: dict[str, object]):
    """values = {agent: value}. Returns (agreed_value_or_None, conflict_or_None)."""
    norms = {a: norm(v) for a, v in values.items()}
    if len(set(norms.values())) == 1:
        # all agree — return the first agent's original (non-normalized) value
        return next(iter(values.values())), None
    return None, {a: values[a] for a in values}


def reconcile_rows(per_agent: dict[str, dict]):
    """per_agent = {agent: {line_number: row}}. Returns (consensus_rows, stats)."""
    agents = list(per_agent)
    all_lines = sorted(
        {ln for rows in per_agent.values() for ln in rows},
        key=lambda x: int(x) if str(x).isdigit() else 999,
    )
    from collections import Counter
    out, review = [], 0
    field_conf: Counter = Counter()
    agreed_cells = total_cells = 0
    for ln in all_lines:
        rows = {a: per_agent[a].get(ln) for a in agents}
        crow = {"line_number": ln}
        conflicts = {}
        total_cells += len(ROW_FIELDS)
        if any(r is None for r in rows.values()):
            # present in some agents but not all -> every cell is a conflict
            crow["confidence"] = "REVIEW"
            crow["conflicts"] = {"_row": {a: ("present" if rows[a] else "missing") for a in agents}}
            for fld in ROW_FIELDS:
                crow[fld] = None
                field_conf[fld] += 1
            out.append(crow); review += 1
            continue
        for fld in ROW_FIELDS:
            agreed, conflict = reconcile_field({a: rows[a].get(fld) for a in agents})
            crow[fld] = agreed
            if conflict is not None:
                conflicts[fld] = conflict
                field_conf[fld] += 1
            else:
                agreed_cells += 1
        crow["confidence"] = "REVIEW" if conflicts else "HIGH"
        crow["conflicts"] = conflicts
        if conflicts:
            review += 1
        out.append(crow)
    stats = {
        "rows": len(out),
        "rows_full_agree": len(out) - review,
        "rows_with_conflict": review,
        "cells_total": total_cells,
        "cells_agree": agreed_cells,
        "cell_agreement_pct": round(100 * agreed_cells / total_cells, 1) if total_cells else 0,
        "field_conflicts": dict(field_conf.most_common()),
    }
    return out, stats


def reconcile_meta(per_agent_meta: dict[str, dict]):
    agents = list(per_agent_meta)
    meta, conflicts = {}, {}
    for fld in META_FIELDS:
        agreed, conflict = reconcile_field({a: per_agent_meta[a].get(fld) for a in agents})
        meta[fld] = agreed
        if conflict is not None:
            conflicts[fld] = conflict
    meta["conflicts"] = conflicts
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile per-agent pages into a consensus.")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--reel", required=True)
    ap.add_argument("--frame", required=True, type=int)
    ap.add_argument("--agents", nargs="+", default=["claude", "gemini"])
    args = ap.parse_args()

    out_dir = args.corpus.resolve() / "output" / "rows" / args.reel
    per_agent, per_meta = {}, {}
    for agent in args.agents:
        p = out_dir / f"{args.reel}_{args.frame:04d}.{agent}.json"
        if not p.exists():
            sys.exit(f"missing per-agent file: {p}\n(run: transcribe.py ... --agent {agent})")
        doc = json.loads(p.read_text())
        per_agent[agent] = {str(r["line_number"]): r for r in doc.get("rows", [])}
        per_meta[agent] = doc.get("metadata", {}) or {}

    rows, stats = reconcile_rows(per_agent)
    consensus = {
        "agents": args.agents,
        "summary": stats,
        "metadata": reconcile_meta(per_meta),
        "rows": rows,
    }
    out_path = out_dir / f"{args.reel}_{args.frame:04d}.consensus.json"
    out_path.write_text(json.dumps(consensus, indent=2))

    print(f"reconciled {args.agents} -> {out_path}", file=sys.stderr)
    print(f"  cell agreement: {stats['cells_agree']}/{stats['cells_total']} "
          f"({stats['cell_agreement_pct']}%)  <- the meaningful number", file=sys.stderr)
    print(f"  rows fully clean: {stats['rows_full_agree']}/{stats['rows']}  |  "
          f"rows with >=1 flagged cell: {stats['rows_with_conflict']}", file=sys.stderr)
    top = list(stats["field_conflicts"].items())[:6]
    print(f"  most-conflicted fields: {', '.join(f'{k}={v}' for k, v in top)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
