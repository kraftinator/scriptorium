"""V2 — nudge → anchored debate → open persuasion round (fresh copy).

Show both candidates; on disagreement each defends its own candidate, then both
pick one of the two (no invented third reading). If they still disagree, an
OPEN persuasion round (any name, always LOW). Deadlock -> Claude, LOW.

The open round is what can occasionally emit an invented blend; v4 removes it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backends import claude_backend, gemini_backend  # noqa: E402

from adjudicate2.common import _conf, norm  # noqa: E402
from adjudicate2.crops import crop_row  # noqa: E402
from adjudicate2.prompts import (  # noqa: E402
    DECIDE_SCHEMA, DEFEND_SCHEMA, NUDGE_SCHEMA,
    OPENARG_SCHEMA, OPENREBUT_SCHEMA,
    decide_prompt, defend_prompt, nudge_prompt,
    openarg_prompt, openrebut_prompt,
)

NAME = "v2"


def adjudicate(crop: Path, label: str, cand_c, cand_g) -> dict:
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


CROP_FN = crop_row
ADJUDICATE_FN = adjudicate
