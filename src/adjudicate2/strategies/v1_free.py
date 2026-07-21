"""V1 — nudge → free re-read debate (fresh copy of adjudicate_name_v1).

Show each model both candidates, re-decide. If they agree, tier1. Otherwise a
FREE open-argument round (each proposes ANY name with a rationale, may drift to
a third reading) plus short-circuit rebuttals. Deadlock -> Claude, LOW.

Trade-off: can recover a correct third reading (Mary A. on the stability test)
but the free re-read can occasionally splice a wrong blend.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backends import claude_backend, gemini_backend  # noqa: E402

from adjudicate2.common import _conf, norm  # noqa: E402
from adjudicate2.crops import crop_row  # noqa: E402
from adjudicate2.prompts import (  # noqa: E402
    NUDGE_SCHEMA, OPENARG_SCHEMA, OPENREBUT_SCHEMA,
    nudge_prompt, openarg_prompt, openrebut_prompt,
)

NAME = "v1"


def adjudicate(crop: Path, label: str, cand_c, cand_g) -> dict:
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


CROP_FN = crop_row
ADJUDICATE_FN = adjudicate
