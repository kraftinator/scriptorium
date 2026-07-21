"""Auto-discover strategies.

Any .py file dropped into this directory that defines module-level:
    NAME             (str)   — the strategy key, e.g. "v5_foo"
    CROP_FN          (callable) — (img, layout, L, scratch, stem) -> Path
    ADJUDICATE_FN    (callable) — (crop_path, label, cand_c, cand_g) -> dict

is automatically added to the STRATEGIES registry. No edits to any central
list needed — drop a file, it shows up.

To add a strategy:
    1. Copy an existing strategies/vN_*.py to strategies/vNEW_yourthing.py
    2. Change NAME and edit the ADJUDICATE_FN body
    3. Done — `python src/adjudicate2/run.py --strategy vNEW_yourthing` works
       and the eval scorecard picks it up on the next run.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable

STRATEGIES: dict[str, dict] = {}


def _load() -> None:
    pkg_dir = Path(__file__).resolve().parent
    for mod_info in pkgutil.iter_modules([str(pkg_dir)]):
        if mod_info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{mod_info.name}")
        name = getattr(mod, "NAME", None)
        crop_fn = getattr(mod, "CROP_FN", None)
        adj_fn = getattr(mod, "ADJUDICATE_FN", None)
        if not (name and crop_fn and adj_fn):
            continue  # skip modules that don't declare the strategy contract
        if name in STRATEGIES:
            raise RuntimeError(
                f"duplicate strategy NAME={name!r} in {mod_info.name}; "
                f"already registered by {STRATEGIES[name]['module']}")
        STRATEGIES[name] = {
            "module": mod_info.name,
            "crop_fn": crop_fn,
            "adjudicate_fn": adj_fn,
            "doc": (mod.__doc__ or "").strip().splitlines()[0] if mod.__doc__ else "",
        }


_load()
