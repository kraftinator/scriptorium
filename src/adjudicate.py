#!/usr/bin/env python3
"""Adjudicate cross-model disagreements in a transcribed page.

Pipeline position: transcribe (per agent) -> [reconcile: pure merge] ->
adjudicate. This step RESOLVES disagreements (it calls models), where reconcile
only flags them. It reads the two per-agent JSON files + the page image, and for
each cell:

  - agree  -> accept, HIGH confidence.
  - disagree on a NAME (first/last) -> run the tier ladder:
      Tier 1  bare-answer nudge : show each model both candidate readings, re-decide.
      Tier 2  argument debate   : each argues a specific visual case; each then
                                  re-examines the other's argument.
      Tier 3  Claude default    : genuine deadlock -> take Claude, flag LOW.
  - disagree on any other field -> default to Claude, flag LOW (no debate).

Every resolution records the tier, the rationale (for debated names), and BOTH
original readings, so nothing is lost and the result is auditable.

Confidence is derived from WHAT HAPPENED, not the models' self-report:
agreed / tier-1-both-confident = HIGH; conceded or debate-resolved = MEDIUM;
defaulted (Tier 3 or non-name) = LOW.

Usage:
    python src/adjudicate.py --corpus corpora/us_census_1850 \
        --reel populationschedu0604unix --frame 23 --agents claude gemini
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import claude_backend, gemini_backend  # noqa: E402
from reconcile import ROW_FIELDS, norm  # noqa: E402

BACKENDS = {"claude": claude_backend, "gemini": gemini_backend}
NAME_FIELDS = {"interpreted_first_name", "interpreted_last_name"}
FIELD_LABEL = {
    "interpreted_first_name": "given name (first name)",
    "interpreted_last_name": "surname",
}

# --- judge/debate output shapes (small, self-contained) ---
NUDGE_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "confidence"]}
DEFEND_SCHEMA = {"type": "object", "properties": {
    "argument": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["argument", "confidence"]}
DECIDE_SCHEMA = {"type": "object", "properties": {
    "final_name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "reasoning": {"type": "string"}},
    "required": ["final_name", "confidence"]}
OPENARG_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "argument": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "argument", "confidence"]}
OPENREBUT_SCHEMA = {"type": "object", "properties": {
    "final_name": {"type": "string"},
    "persuaded": {"type": "boolean"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "reasoning": {"type": "string"}},
    "required": ["final_name", "persuaded", "confidence"]}


def nudge_prompt(label, a, b):
    return (f"You are re-examining ONE handwritten {label} on an 1850 U.S. Census "
            "page (a single ruled row). Two transcribers read this "
            f"{label} as '{a}' and '{b}'. Study the handwriting letter by letter and "
            "report the correct reading. Transcribe the EXACT letters written — do "
            "NOT prefer the more common or modern form of a name over what is "
            "actually on the page (e.g. if it is written 'Chls', report 'Chls', not "
            "the more familiar 'Chas'). Return JSON: name (your reading of the "
            f"{label}), confidence (HIGH/MEDIUM/LOW). No prose outside the JSON.")


def defend_prompt(label, own, other):
    return (f"Examine ONE handwritten {label} on this single-row 1850 U.S. Census "
            f"image. You previously read it as '{own}'; another transcriber read "
            f"'{other}'. Make the strongest LETTER-BY-LETTER case for why '{own}' is "
            "the correct reading of what is ACTUALLY written (cite specific letter "
            "shapes, especially the first capital). Judge by the exact letters on "
            "the page, not by which is a more familiar name. Return JSON: argument "
            f"(1-3 sentences), confidence (HIGH/MEDIUM/LOW that '{own}' is correct). "
            "No prose outside the JSON.")


def decide_prompt(label, a, arg_a, b, arg_b):
    return (f"Decide the correct reading of ONE handwritten {label} on this 1850 "
            "U.S. Census row. EXACTLY TWO readings are proposed, each argued:\n"
            f"  A) '{a}' — {arg_a}\n"
            f"  B) '{b}' — {arg_b}\n"
            "Look at the image and choose which of these two matches the exact "
            f"letters written. Your answer MUST be either '{a}' or '{b}', verbatim — "
            "do NOT invent a third reading. Return JSON: final_name (either "
            f"'{a}' or '{b}'), confidence (HIGH/MEDIUM/LOW), reasoning (1 sentence). "
            "No prose outside the JSON.")


def openarg_prompt(label, a, b):
    return (f"Two transcribers proposed '{a}' and '{b}' for this handwritten {label} "
            "on an 1850 U.S. Census row, but they could NOT agree — so neither may "
            f"be correct. Read the {label} fresh from the image, letter by letter, "
            "and give your best reading (it may be one of those two OR a different "
            "name entirely), plus a 1-2 sentence argument citing SPECIFIC letter "
            "shapes. Transcribe the EXACT letters written, not the most familiar "
            "name. Return JSON: name, argument, confidence (HIGH/MEDIUM/LOW). No "
            "prose outside the JSON.")


def openrebut_prompt(label, other_name, other_arg):
    return (f"Re-examine this handwritten {label} on an 1850 U.S. Census row. "
            f"Another expert transcriber reads it as '{other_name}' and argues: "
            f"\"{other_arg}\" Look again at the image, focusing on the features they "
            "cite. Give your final reading — agree with them, keep your own, or a "
            "different reading if the letters show something else. Return JSON: "
            "final_name, persuaded (true if you adopted their reading), confidence "
            "(HIGH/MEDIUM/LOW), reasoning (1 sentence). No prose outside the JSON.")


def openread_prompt(label):
    return (f"Read the {label} written in this image — a single handwritten name "
            "from an 1850 U.S. Census. What does it say? Transcribe the EXACT "
            "letters as written (do not prefer a more familiar or common name), "
            "and INCLUDE any middle initial written as part of a given name "
            "(e.g. 'Milo J.', not just 'Milo'). Return JSON: name, confidence "
            "(HIGH/MEDIUM/LOW). No prose outside the JSON.")


def _conf(vals):
    """Derived confidence: LOW if any voter is unsure, HIGH only if all sure."""
    vals = [str(v).upper() for v in vals if v]
    if "LOW" in vals:
        return "LOW"
    if vals and all(v == "HIGH" for v in vals):
        return "HIGH"
    return "MEDIUM"


def adjudicate_name_v1(crop: Path, label: str, cand_c, cand_g) -> dict:
    """V1 (original): show both candidates, then a FREE re-read debate — each
    proposes any reading with an argument and can be persuaded (may drift to a
    third reading). Full-row crop. Got Mary A. right by drift; missed Chls/Hulett.
    """
    base = {"claude": cand_c, "gemini": cand_g}
    try:
        c1 = claude_backend(crop, nudge_prompt(label, cand_c, cand_g), NUDGE_SCHEMA)
        g1 = gemini_backend(crop, nudge_prompt(label, cand_c, cand_g), NUDGE_SCHEMA)
    except Exception as e:
        return {**base, "value": cand_c, "confidence": "LOW", "tier": "error",
                "rationale": f"adjudication call failed: {e}"}
    if norm(c1.get("name")) == norm(g1.get("name")):
        return {**base, "value": c1.get("name"), "tier": "tier1", "rationale": None,
                "confidence": _conf([c1.get("confidence"), g1.get("confidence")])}
    try:
        oa = claude_backend(crop, openarg_prompt(label, cand_c, cand_g), OPENARG_SCHEMA)
        ob = gemini_backend(crop, openarg_prompt(label, cand_c, cand_g), OPENARG_SCHEMA)
        na, nb = oa.get("name"), ob.get("name")
        if norm(na) == norm(nb):
            return {**base, "value": na, "tier": "debate", "rationale": oa.get("argument"),
                    "confidence": _conf([oa.get("confidence"), ob.get("confidence")])}
        rg = gemini_backend(crop, openrebut_prompt(label, na, oa.get("argument")), OPENREBUT_SCHEMA)
        if norm(rg.get("final_name")) == norm(na):
            return {**base, "value": na, "tier": "debate", "rationale": oa.get("argument"),
                    "confidence": _conf([oa.get("confidence"), rg.get("confidence")])}
        rc = claude_backend(crop, openrebut_prompt(label, nb, ob.get("argument")), OPENREBUT_SCHEMA)
        if norm(rc.get("final_name")) == norm(nb):
            return {**base, "value": nb, "tier": "debate", "rationale": ob.get("argument"),
                    "confidence": _conf([ob.get("confidence"), rc.get("confidence")])}
    except Exception:
        pass
    return {**base, "value": cand_c, "confidence": "LOW", "tier": "tier3",
            "rationale": "unresolved after debate; defaulted to Claude"}


def adjudicate_name_v2(crop: Path, label: str, cand_c, cand_g) -> dict:
    """V2 (committed): show both candidates; ANCHORED debate — each defends its
    candidate, then both pick one of the two (no invented third reading); on
    deadlock an open persuasion round (LOW); else Claude+LOW. Full-row crop.
    """
    base = {"claude": cand_c, "gemini": cand_g}
    try:
        c1 = claude_backend(crop, nudge_prompt(label, cand_c, cand_g), NUDGE_SCHEMA)
        g1 = gemini_backend(crop, nudge_prompt(label, cand_c, cand_g), NUDGE_SCHEMA)
    except Exception as e:
        return {**base, "value": cand_c, "confidence": "LOW", "tier": "error",
                "rationale": f"adjudication call failed: {e}"}
    cn, gn = c1.get("name"), g1.get("name")
    if norm(cn) == norm(gn):
        return {**base, "value": cn, "tier": "tier1", "rationale": None,
                "confidence": _conf([c1.get("confidence"), g1.get("confidence")])}
    try:
        da = claude_backend(crop, defend_prompt(label, cand_c, cand_g), DEFEND_SCHEMA)
        db = gemini_backend(crop, defend_prompt(label, cand_g, cand_c), DEFEND_SCHEMA)
        dp = decide_prompt(label, cand_c, da["argument"], cand_g, db["argument"])
        dc = claude_backend(crop, dp, DECIDE_SCHEMA)
        dg = gemini_backend(crop, dp, DECIDE_SCHEMA)

        def pick(v):
            if norm(v) == norm(cand_c):
                return cand_c
            if norm(v) == norm(cand_g):
                return cand_g
            return None
        pc, pg = pick(dc.get("final_name")), pick(dg.get("final_name"))
        if pc is not None and pc == pg:
            rationale = da["argument"] if pc == cand_c else db["argument"]
            return {**base, "value": pc, "tier": "tier2", "rationale": rationale,
                    "confidence": _conf([dc.get("confidence"), dg.get("confidence")])}

        oa = claude_backend(crop, openarg_prompt(label, cand_c, cand_g), OPENARG_SCHEMA)
        ob = gemini_backend(crop, openarg_prompt(label, cand_c, cand_g), OPENARG_SCHEMA)
        na, nb = oa.get("name"), ob.get("name")
        if norm(na) == norm(nb):
            return {**base, "value": na, "tier": "tier2-open", "confidence": "LOW",
                    "rationale": oa.get("argument")}
        rg = gemini_backend(crop, openrebut_prompt(label, na, oa.get("argument")), OPENREBUT_SCHEMA)
        if norm(rg.get("final_name")) == norm(na):
            return {**base, "value": na, "tier": "tier2-open", "confidence": "LOW",
                    "rationale": oa.get("argument")}
        rc = claude_backend(crop, openrebut_prompt(label, nb, ob.get("argument")), OPENREBUT_SCHEMA)
        if norm(rc.get("final_name")) == norm(nb):
            return {**base, "value": nb, "tier": "tier2-open", "confidence": "LOW",
                    "rationale": ob.get("argument")}
    except Exception:
        pass
    return {**base, "value": cand_c, "confidence": "LOW", "tier": "tier3",
            "rationale": "unresolved after debate; defaulted to Claude"}


def adjudicate_name_v3(crop: Path, label: str, cand_c, cand_g) -> dict:
    """V3 (experimental): LEAD with an UNBIASED open read (cell crop, NO candidates
    shown, so the read isn't anchored to the guesses), then reconcile the clean
    reads: agree -> accept; disagree -> persuasion debate between the open reads;
    deadlock -> Claude's open read, LOW.
    """
    base = {"claude": cand_c, "gemini": cand_g}
    try:
        oc = claude_backend(crop, openread_prompt(label), NUDGE_SCHEMA)
        og = gemini_backend(crop, openread_prompt(label), NUDGE_SCHEMA)
    except Exception as e:
        return {**base, "value": cand_c, "confidence": "LOW", "tier": "error",
                "rationale": f"open-read call failed: {e}"}
    nc, ng = oc.get("name"), og.get("name")
    base = {**base, "open_claude": nc, "open_gemini": ng}
    if norm(nc) == norm(ng):
        return {**base, "value": nc, "tier": "open-agree", "rationale": None,
                "confidence": _conf([oc.get("confidence"), og.get("confidence")])}
    try:
        da = claude_backend(crop, defend_prompt(label, nc, ng), DEFEND_SCHEMA)
        db = gemini_backend(crop, defend_prompt(label, ng, nc), DEFEND_SCHEMA)
        rg = gemini_backend(crop, openrebut_prompt(label, nc, da["argument"]), OPENREBUT_SCHEMA)
        if norm(rg.get("final_name")) == norm(nc):
            return {**base, "value": nc, "tier": "open-debate", "rationale": da["argument"],
                    "confidence": _conf([oc.get("confidence"), rg.get("confidence")])}
        rc = claude_backend(crop, openrebut_prompt(label, ng, db["argument"]), OPENREBUT_SCHEMA)
        if norm(rc.get("final_name")) == norm(ng):
            return {**base, "value": ng, "tier": "open-debate", "rationale": db["argument"],
                    "confidence": _conf([og.get("confidence"), rc.get("confidence")])}
    except Exception:
        pass
    return {**base, "value": nc, "confidence": "LOW", "tier": "deadlock",
            "rationale": "open reads disagreed and debate did not converge; "
                         "defaulted to Claude's open read"}


def adjudicate_name_v4(crop: Path, label: str, cand_c, cand_g) -> dict:
    """V4 (anchored-strict): like v2 but with NO open round. The final answer is
    ALWAYS one of the two candidate readings; a deadlock goes straight to Claude's
    candidate (LOW). This BANS invented third readings, so a spliced blend like
    'Noward' (from Howard + Norwood) can never appear. Trade-off: it cannot
    recover a correct third reading (e.g. 'Mary A.' when both candidates are wrong).
    """
    base = {"claude": cand_c, "gemini": cand_g}
    try:
        c1 = claude_backend(crop, nudge_prompt(label, cand_c, cand_g), NUDGE_SCHEMA)
        g1 = gemini_backend(crop, nudge_prompt(label, cand_c, cand_g), NUDGE_SCHEMA)
    except Exception as e:
        return {**base, "value": cand_c, "confidence": "LOW", "tier": "error",
                "rationale": f"adjudication call failed: {e}"}
    cn, gn = c1.get("name"), g1.get("name")
    if norm(cn) == norm(gn):
        return {**base, "value": cn, "tier": "tier1", "rationale": None,
                "confidence": _conf([c1.get("confidence"), g1.get("confidence")])}
    try:
        da = claude_backend(crop, defend_prompt(label, cand_c, cand_g), DEFEND_SCHEMA)
        db = gemini_backend(crop, defend_prompt(label, cand_g, cand_c), DEFEND_SCHEMA)
        dp = decide_prompt(label, cand_c, da["argument"], cand_g, db["argument"])
        dc = claude_backend(crop, dp, DECIDE_SCHEMA)
        dg = gemini_backend(crop, dp, DECIDE_SCHEMA)

        def pick(v):
            if norm(v) == norm(cand_c):
                return cand_c
            if norm(v) == norm(cand_g):
                return cand_g
            return None
        pc, pg = pick(dc.get("final_name")), pick(dg.get("final_name"))
        if pc is not None and pc == pg:  # both chose the same candidate
            rationale = da["argument"] if pc == cand_c else db["argument"]
            return {**base, "value": pc, "tier": "tier2", "rationale": rationale,
                    "confidence": _conf([dc.get("confidence"), dg.get("confidence")])}
    except Exception:
        pass
    # deadlock -> Claude's candidate, LOW. NO invented third reading.
    return {**base, "value": cand_c, "confidence": "LOW", "tier": "tier3",
            "rationale": "no agreement between the two candidates; defaulted to Claude"}


def get_page_png(reel_dir: Path, reel: str, frame: int, scratch: Path) -> Path:
    png = reel_dir / f"{reel}_{frame:04d}.png"
    if png.exists():
        return png
    jp2 = reel_dir / f"{reel}_{frame:04d}.jp2"
    out = scratch / f"{reel}_{frame:04d}.png"
    if not out.exists():
        subprocess.run(["sips", "-s", "format", "png", str(jp2), "--out", str(out)],
                       capture_output=True)
    return out


def crop_name_cell(img: Image.Image, layout: dict, L: int, scratch: Path, stem: str) -> Path:
    """Crop to the NAME CELL (name column x row), not the full-width row, and
    upscale. The name is a tiny slice of a wide row; cropping to the cell gives
    the model the resolution it needs (proven: a full-row crop failed to read an
    L4 initial that the tight cell crop got right)."""
    top, pitch = layout["row1_top"], layout["row_pitch"]
    W, H = img.size
    y0 = max(0, top + (L - 1) * pitch - 40)
    y1 = min(H, top + L * pitch + 40)
    f0, f1 = layout.get("name_col_frac", [0.0, 1.0])  # fraction of width -> pixels
    x0, x1 = int(f0 * W), int(f1 * W)
    cell = img.crop((x0, y0, x1, y1))
    cell = cell.resize((cell.width * 2, cell.height * 2), Image.LANCZOS)
    p = scratch / f"{stem}_adj_row{L:02d}.png"
    cell.save(p)
    return p


def crop_row(img: Image.Image, layout: dict, L: int, scratch: Path, stem: str) -> Path:
    """Crop the FULL-WIDTH row (V1/V2 behaviour — the whole row, all columns)."""
    top, pitch = layout["row1_top"], layout["row_pitch"]
    W, H = img.size
    y0 = max(0, top + (L - 1) * pitch - 40)
    y1 = min(H, top + L * pitch + 40)
    p = scratch / f"{stem}_adj_row{L:02d}.png"
    img.crop((0, y0, W, y1)).save(p)
    return p


# strategy -> (crop function, adjudicate function). Swap the whole approach with
# --strategy so all three stay runnable for side-by-side comparison.
STRATEGIES = {
    "v1": (crop_row, adjudicate_name_v1),        # candidates shown, free re-read debate
    "v2": (crop_row, adjudicate_name_v2),        # candidates shown, anchored debate (committed)
    "v3": (crop_name_cell, adjudicate_name_v3),  # open-read-first, cell crop
    "v4": (crop_row, adjudicate_name_v4),        # anchored-strict: no third readings
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Adjudicate name disagreements via tiered debate.")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--reel", required=True)
    ap.add_argument("--frame", required=True, type=int)
    ap.add_argument("--agents", nargs="+", default=["claude", "gemini"])
    ap.add_argument("--lines", nargs="*", type=int, default=None,
                    help="only adjudicate these line numbers (for testing)")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="v1",
                    help="adjudication strategy (default v1, the free-debate flow — "
                         "best on the stability test: 2/3 on Mary A. + honest LOW)")
    args = ap.parse_args()
    if args.agents[:2] != ["claude", "gemini"]:
        sys.exit("adjudicate assumes agents 'claude gemini' (Claude is the Tier-3 default).")
    crop_fn, adjudicate_fn = STRATEGIES[args.strategy]

    corpus = args.corpus.resolve()
    out_dir = corpus / "output" / "rows" / args.reel
    per = {}
    for agent in ("claude", "gemini"):
        p = out_dir / f"{args.reel}_{args.frame:04d}.{agent}.json"
        if not p.exists():
            sys.exit(f"missing per-agent file: {p}")
        doc = json.loads(p.read_text())
        per[agent] = {str(r["line_number"]): r for r in doc.get("rows", [])}
    meta = json.loads((out_dir / f"{args.reel}_{args.frame:04d}.claude.json").read_text()).get("metadata", {})

    layout = json.loads((corpus / "config" / "layout.json").read_text())
    reel_dir = corpus / "data" / "reels" / args.reel
    scratch = out_dir / "_adj"
    scratch.mkdir(parents=True, exist_ok=True)
    page_png = get_page_png(reel_dir, args.reel, args.frame, scratch)
    img = Image.open(page_png)
    stem = f"{args.reel}_{args.frame:04d}"

    all_lines = sorted({ln for a in per for ln in per[a]},
                       key=lambda x: int(x) if str(x).isdigit() else 999)
    if args.lines:
        want = {str(x) for x in args.lines}
        all_lines = [ln for ln in all_lines if ln in want]
    # pre-count name disagreements so progress has a denominator
    n_names = sum(1 for ln in all_lines if per["claude"].get(ln) and per["gemini"].get(ln)
                  for f in NAME_FIELDS
                  if norm(per["claude"][ln].get(f)) != norm(per["gemini"][ln].get(f)))
    print(f"[adj] strategy={args.strategy}: {n_names} name disagreements to resolve",
          file=sys.stderr, flush=True)
    done_names = 0
    out_rows, resolutions, tally = [], [], {"agree": 0, "tier1": 0, "tier2": 0,
                                            "tier3": 0, "claude-default": 0, "error": 0}
    for ln in all_lines:
        rc, rg = per["claude"].get(ln), per["gemini"].get(ln)
        if rc is None or rg is None:  # a whole row is missing from one agent
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
                print(f"[adj] ({done_names}/{n_names}) L{ln} {field[12:]}: "
                      f"{c0!r} vs {g0!r} -> {res['value']!r} "
                      f"[{res['confidence']}/{res['tier']}]", file=sys.stderr, flush=True)
            else:  # non-name disagreement -> Claude default, LOW
                res = {"claude": c0, "gemini": g0, "value": c0, "confidence": "LOW",
                       "tier": "claude-default", "rationale": None}
            row[field] = res["value"]
            tally[res["tier"]] = tally.get(res["tier"], 0) + 1
            resolutions.append({"line_number": ln, "field": field, **res})
            if res["confidence"] == "LOW":
                row_conf = "LOW"
            elif res["confidence"] == "MEDIUM" and row_conf != "LOW":
                row_conf = "MEDIUM"
        row["confidence"] = row_conf
        out_rows.append(row)

    result = {"agents": ["claude", "gemini"], "strategy": args.strategy,
              "metadata": meta, "rows": out_rows, "resolutions": resolutions}
    part = ".partial" if args.lines else ""
    out_path = out_dir / f"{args.reel}_{args.frame:04d}.adjudicated.{args.strategy}{part}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"adjudicated -> {out_path}", file=sys.stderr)
    name_res = [r for r in resolutions if r["field"] in NAME_FIELDS]
    print(f"  name disagreements: {len(name_res)} | resolved "
          f"open-agree={tally.get('open-agree', 0)} "
          f"open-debate={tally.get('open-debate', 0)} "
          f"deadlock(LOW)={tally.get('deadlock', 0)} err={tally.get('error', 0)}",
          file=sys.stderr)
    for r in name_res:
        print(f"    L{r['line_number']} {r['field'][12:]}: "
              f"claude={r['claude']!r} gemini={r['gemini']!r} -> "
              f"{r['value']!r} [{r['confidence']}/{r['tier']}]", file=sys.stderr)


if __name__ == "__main__":
    main()
