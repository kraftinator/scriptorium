# Scriptorium

Turn images of structured historic documents into clean, structured database
rows — with per-cell confidence and a targeted human-review loop.

The engine (`src/`) is corpus-agnostic. Each dataset is a self-contained folder
under `corpora/`. The first corpus is the **1850 U.S. Census**; the same engine
is meant to extend to later censuses and, eventually, any structured historic
document.

## Layout

```
scriptorium/
├── src/
│   ├── transcribe.py     pipeline: convert → dispatch to a backend → save
│   └── backends.py       model adapters (claude, gemini) + registry
└── corpora/
    └── us_census_1850/
        ├── config/       page_schema.json + transcription_prompt.txt
        ├── data/
        │   ├── reels/    source page images (gitignored)
        │   └── pages/    converted working JPGs (gitignored)
        ├── gold/         hand-verified pages for accuracy measurement
        └── output/
            └── rows/     per-page JSON, one file per page per agent
```

## How it works

One page image in, structured JSON out:

```
.venv/bin/python src/transcribe.py \
    --corpus corpora/us_census_1850 --reel <reel> --frame <n> --agent claude|gemini
```

1. Convert the page (JP2 → JPG, cached).
2. Send the image + the corpus prompt + the corpus schema to the chosen model.
3. Save `<reel>_<frame>.<agent>.json` under `output/rows/`.

The model-specific code lives in `backends.py`; adding a model is one function
there and the pipeline never changes.

- **claude** — runs the `claude` CLI (`--strict-mcp-config` to stay isolated).
  No API key.
- **gemini** — `google-genai` with native schema enforcement. Needs
  `GEMINI_API_KEY`.

## Approach

- **Store: DuckDB + Parquet.** Single file, scales to the full row set on a
  laptop; good for both name search and aggregate queries.
- **The hard part is proper-noun ambiguity, not image resolution.** Structured
  columns transcribe near-perfectly; errors concentrate in names (S/L and
  similar confusions in Spencerian script). Context/priors and review resolve
  those, not more pixels.
- **Cross-model consensus is the confidence signal.** Two independent models on
  the same page: where they agree, trust it; where they disagree, route to
  review. This beats a single model's self-assessment — on a pilot page one
  model marked every row high-confidence (including guesses on unreadable
  cells), while model *disagreement* pinpointed exactly the hard names.
- **Ditto marks stay raw (`[DITTO]`) in the per-page JSON;** resolution is a
  shared downstream step at ingestion, so every backend's output stays directly
  comparable.

## Status

- [x] Scaffold
- [x] Per-page transcription worker (claude + gemini backends)
- [x] Cross-model consensus validated on a pilot page (Barton, Tioga Co.)
- [ ] Reconcile step: merge per-agent files → consensus rows
- [ ] DuckDB ingestion (+ shared ditto resolution)
- [ ] Query layers (name / aggregate / natural-language)

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # google-genai (gemini backend)
```

Also required: macOS `sips` (conversion) and the `claude` CLI (claude backend).
