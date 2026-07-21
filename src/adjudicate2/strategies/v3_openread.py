"""V3 — open-read first, cell crop (fresh copy).

LEAD with an UNBIASED open read (tight name-cell crop, NO candidates shown, so
the read isn't anchored to the initial guesses). Then reconcile the clean reads:
agree -> accept; disagree -> persuasion debate; deadlock -> Claude's open read, LOW.

Underperformed the anchored variants in the pilot (2/5) — the tight-crop
no-anchor read garbled several cells — but kept as a comparison baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backends import claude_backend, gemini_backend  # noqa: E402

from adjudicate2.common import _conf, norm  # noqa: E402
from adjudicate2.crops import crop_name_cell  # noqa: E402
from adjudicate2.prompts import (  # noqa: E402
    DEFEND_SCHEMA, NUDGE_SCHEMA, OPENREBUT_SCHEMA,
    defend_prompt, openread_prompt, openrebut_prompt,
)

NAME = "v3"


def adjudicate(crop: Path, label: str, cand_c, cand_g) -> dict:
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


CROP_FN = crop_name_cell
ADJUDICATE_FN = adjudicate
