#!/usr/bin/env python3
"""Scorecard runner for adjudication strategies.

Reads eval/fixtures.json (ground-truth cells), runs each requested strategy on
every case, and prints a matrix: rows = strategies, columns = cases, cells =
✓ / ✗ / ? (UNCONFIRMED — no ground truth yet). Also writes a JSON audit trail
so you can review actual vs expected readings after the run.

Usage:
    # score all registered strategies on all fixtures:
    python src/adjudicate2/eval/score.py

    # score just a subset of strategies / cases:
    python src/adjudicate2/eval/score.py --strategies v4
    python src/adjudicate2/eval/score.py --cases L1-last L33-last

The per-agent candidate readings are pulled from the CURRENT per-agent JSON
files at runtime — if you re-transcribe the page, the same fixtures apply
against the new readings without any fixture edits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

# make `import adjudicate2.*` work when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adjudicate2.common import norm
from adjudicate2.prompts import FIELD_LABEL
from adjudicate2.strategies import STRATEGIES

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures.json"


def _load_per_agent(out_dir: Path, reel: str, frame: int) -> dict:
    per = {}
    for agent in ("claude", "gemini"):
        p = out_dir / f"{reel}_{frame:04d}.{agent}.json"
        if not p.exists():
            sys.exit(f"missing per-agent file: {p}\n(run the transcribe step first)")
        doc = json.loads(p.read_text())
        per[agent] = {str(r["line_number"]): r for r in doc.get("rows", [])}
    return per


def _match(actual, expected) -> bool:
    return norm(actual) == norm(expected)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score adjudication strategies against ground-truth cells.")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--strategies", nargs="+", default=None,
                    help="subset to score (default: all registered)")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="subset of case ids to score (default: all)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write JSON audit trail here (default: eval/results/<ts>.json)")
    args = ap.parse_args()

    fx = json.loads(args.fixtures.read_text())
    corpus = (REPO / fx["corpus"]).resolve()
    reel, frame = fx["reel"], fx["frame"]
    cases = fx["cases"]
    if args.cases:
        want = set(args.cases)
        cases = [c for c in cases if c["id"] in want]
        missing = want - {c["id"] for c in cases}
        if missing:
            sys.exit(f"unknown case ids: {sorted(missing)}")

    strategies = args.strategies or sorted(STRATEGIES)
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        sys.exit(f"unknown strategies: {unknown}\navailable: {sorted(STRATEGIES)}")

    out_dir = corpus / "output" / "rows" / reel
    per = _load_per_agent(out_dir, reel, frame)
    layout = json.loads((corpus / "config" / "layout.json").read_text())
    scratch = out_dir / "_adj2" / "eval"
    scratch.mkdir(parents=True, exist_ok=True)
    stem = f"{reel}_{frame:04d}"
    page_png = out_dir / "_adj2" / f"{stem}.png"
    if not page_png.exists():
        # let the orchestrator convert; simplest is to re-use its helper
        from adjudicate2.orchestrator import _page_png
        page_png = _page_png(corpus / "data" / "reels" / reel, reel, frame,
                             out_dir / "_adj2")
    img = Image.open(page_png)

    # gather candidate readings per case
    prepared = []
    for c in cases:
        ln = str(c["line"])
        cand_c = per["claude"].get(ln, {}).get(c["field"])
        cand_g = per["gemini"].get(ln, {}).get(c["field"])
        prepared.append({**c, "cand_claude": cand_c, "cand_gemini": cand_g})

    results: dict[str, dict[str, dict]] = {}
    for s in strategies:
        info = STRATEGIES[s]
        crop_fn, adj_fn = info["crop_fn"], info["adjudicate_fn"]
        results[s] = {}
        print(f"\n[eval] === strategy: {s} ({info['module']}) ===", flush=True)
        for c in prepared:
            cid = c["id"]
            if _match(c["cand_claude"], c["cand_gemini"]):
                # candidates already agree — the strategy would never fire.
                # Mark it as agreement (no strategy call).
                r = {"value": c["cand_claude"], "tier": "agree(no-call)",
                     "confidence": "HIGH", "elapsed_s": 0.0}
            else:
                crop = crop_fn(img, layout, int(c["line"]), scratch, stem)
                t0 = time.monotonic()
                try:
                    res = adj_fn(crop, FIELD_LABEL[c["field"]],
                                 c["cand_claude"], c["cand_gemini"])
                    r = {"value": res.get("value"), "tier": res.get("tier"),
                         "confidence": res.get("confidence"),
                         "elapsed_s": round(time.monotonic() - t0, 1)}
                except Exception as e:
                    r = {"value": None, "tier": "error", "confidence": None,
                         "elapsed_s": round(time.monotonic() - t0, 1),
                         "error": f"{type(e).__name__}: {e}"}
            if c["correct"] is None:
                mark = "?"
            elif _match(r["value"], c["correct"]):
                mark = "PASS"
            else:
                mark = "FAIL"
            r["mark"] = mark
            r["expected"] = c["correct"]
            r["candidates"] = {"claude": c["cand_claude"], "gemini": c["cand_gemini"]}
            results[s][cid] = r
            print(f"[eval] {cid:10s}  claude={c['cand_claude']!r:>18} "
                  f"gemini={c['cand_gemini']!r:>18} -> {r['value']!r:>18}  "
                  f"expected={c['correct']!r:>15}  {mark} "
                  f"[{r['confidence']}/{r['tier']}]  {r['elapsed_s']}s",
                  flush=True)

    # scorecard matrix
    print("\n" + "=" * 78)
    print("scorecard  (PASS ✓ | FAIL ✗ | ? unconfirmed)")
    print("=" * 78)
    header = "strategy      " + " ".join(f"{c['id']:>11}" for c in prepared) + "   score"
    print(header)
    for s in strategies:
        row = f"{s:12s}  "
        pass_ct = fail_ct = confirmed = 0
        for c in prepared:
            m = results[s][c["id"]]["mark"]
            sym = {"PASS": "     ✓     ", "FAIL": "     ✗     ", "?": "     ?     "}[m]
            row += f" {sym}"
            if m == "PASS":
                pass_ct += 1; confirmed += 1
            elif m == "FAIL":
                fail_ct += 1; confirmed += 1
        row += f"   {pass_ct}/{confirmed} confirmed"
        print(row)
    print()

    # audit trail
    out_path = args.out or (Path(__file__).resolve().parent / "results"
                            / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "fixtures": str(args.fixtures),
        "strategies": strategies,
        "cases": [c["id"] for c in prepared],
        "results": results,
    }, indent=2))
    print(f"[eval] audit trail: {out_path}")


if __name__ == "__main__":
    main()
