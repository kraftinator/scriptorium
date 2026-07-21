"""Adjudicate2 — parallel experimentation home for adjudication strategies.

Fresh copy of the adjudication tree so new strategies (v5+) can evolve without
touching the v1–v4 code in src/adjudicate.py. Nothing here imports from that
file; helpers (prompts, crops, norm/_conf) are duplicated locally.

Layout:
  prompts.py      — nudge/defend/decide/openarg/openread prompts + JSON schemas
  crops.py        — crop_row, crop_name_cell (upscale name cell)
  orchestrator.py — per-field dispatch + output writing (.adjudicated2.<strategy>.json)
  run.py          — CLI: python src/adjudicate2/run.py --strategy vN ...
  strategies/     — one file per strategy, auto-discovered by strategies/__init__.py
  eval/           — fixtures.json (ground-truth cells) + score.py (scorecard runner)
"""
