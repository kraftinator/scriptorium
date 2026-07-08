# Scriptorium — Project State & Handoff

Crash-recovery brief. If you're a fresh Claude picking this up, read this top to
bottom — it's the accumulated context that isn't obvious from the code alone.

## Goal

Scriptorium is a **corpus-agnostic engine** to digitize structured historic
documents into an open, searchable database from public-domain page images.
First corpus: the **1850 U.S. Census**. Pilot: **Tioga County, NY**, reel
`populationschedu0604unix`. The engine must stay corpus-agnostic — corpus
specifics (schema, prompts, geometry) live under `corpora/<corpus>/config/`,
never hardcoded in `src/`.

The user is a Rails developer; we use Python here. Communication is often over a
**Telegram channel** (the user reads Telegram, not the terminal — reply via the
telegram MCP `reply` tool). The user drives one decision at a time; confirm
before large/irreversible actions and before spending many model calls.

## Pipeline (the 5 steps)

1. Read file → 2. Convert JP2→PNG → 3. AI analyzes page → 4. Return JSON → 5. DB.
Steps 2–4 are **built and validated**. Step 5 (DB) is **not built yet**.

Flow: page image → per-agent tiled transcription (Claude *and* Gemini,
independently) → cross-model **reconcile** into a per-cell consensus → (future)
ditto resolution → (future) DuckDB.

## How to run

```bash
# One page, one model, tiled (the mode we use):
.venv/bin/python src/transcribe.py --corpus corpora/us_census_1850 \
    --reel populationschedu0604unix --frame 23 --agent gemini --mode tiled
# Claude backend needs no key (uses CLI login); Gemini needs GEMINI_API_KEY:
GEMINI_API_KEY=... .venv/bin/python src/transcribe.py ... --agent claude --mode tiled

# Reconcile the two per-agent outputs into a consensus (reads files, no model calls):
.venv/bin/python src/reconcile.py --corpus corpora/us_census_1850 \
    --reel populationschedu0604unix --frame 23 --agents claude gemini

# Web interface (Gemini playground at /, output viewer at /view):
WEBPLAY_HOST=0.0.0.0 GEMINI_API_KEY=... .venv/bin/python webplay.py   # port 5001
```

Outputs land in `corpora/us_census_1850/output/rows/<reel>/`:
`<reel>_<frame>.<agent>.json` per model, `<reel>_<frame>.consensus.json` merged.

## Architecture / file map

- `src/transcribe.py` — CLI entry. `--corpus --reel --frame --agent{claude,gemini} --mode{whole,tiled}`. Converts JP2→PNG via `sips`, dispatches to backend or tiler.
- `src/backends.py` — model adapters. `REGISTRY = {claude, gemini}`. Add a model = add a function + register; the pipeline never changes.
  - `claude_backend`: shells out to `claude -p ... --model claude-opus-4-8 --allowedTools Read --strict-mcp-config`. No API key (uses login). `--strict-mcp-config` is REQUIRED — it isolates the child from MCP servers so it can't steal the Telegram bot's poll connection.
  - `gemini_backend`: `google-genai` SDK, native `response_schema` (temp 0), retry/backoff. `GEMINI_MODEL` env picks flash (default) vs `gemini-2.5-pro`.
  - `_strip_fence`: extracts the JSON object (first `{` to last `}`) — tolerates Claude's conversational preamble + markdown fence (see Findings).
  - `to_gemini_schema`: normalizes our JSON Schema for Gemini (`["string","null"]` → `nullable:true`, drops `$schema`/`title`).
- `src/tile.py` — `transcribe_tiled`: crops the header for metadata, then row-bands. **Line numbers are anchored to the pre-printed margin numbers** the form shows (1–42), validated against each band's expected range — so a stray sliver row is filtered, not shifted. Logs missing lines.
- `src/reconcile.py` — merges per-agent files into consensus. Per cell: agree → value + HIGH; disagree → both readings in `conflicts` + REVIEW. Reports **cell-level agreement %** (the meaningful metric). `norm()` collapses ditto marks + blanks to one equivalence class. `raw_name` excluded (derived from first+last, double-counts).
- `webplay.py` — Flask dev tool. `/` = Gemini API playground (pick model/temp/prompt, feed a page or a cropped row-band). `/view` = output viewer: consensus as a table, conflict cells highlighted with both models' readings — doubles as the human-review UI.
- `play.py` — CLI Gemini playground (edit knobs, run).
- `corpora/us_census_1850/config/`:
  - `page_schema.json` — the canonical schema (metadata + rows[], nullable strings, confidence_score enum, transcription_notes). Single source of truth.
  - `transcription_prompt.txt` — whole-page prompt. `band_prompt.txt` — per-band prompt. Both enforce: transcribe EXACTLY (no expanding abbreviations — "Geo." stays "Geo."), read every digit of dwelling/family numbers, the Ditto Rule (any ditto mark in ANY column → literal `[DITTO]`, never the raw `"`, never resolved), `[ILLEGIBLE]` for unreadable, confidence scoring.
  - `layout.json` — tiling geometry: `n_rows=42, row1_top=1620, row_pitch=150, band_rows=6, crop_margin=80`. Calibrated on frame 0023; **validated to generalize across the reel** (other pages/towns have different pixel dimensions but the same vertical row grid).

## Key design decisions & why

- **Tiling (row-bands), not whole-page.** Vision APIs downsample a ~6600×8400 page so far that fine strokes blur — full page reads "Snickaback", a row-band reads "Knickerbocker". Slicing restores per-row resolution.
- **Cross-model consensus is the accuracy lever.** No single model (Gemini Flash, Gemini Pro, or Claude) reads 1850 Spencerian script perfectly — the ink is genuinely ambiguous. Design: run two decent models, where they AGREE = high confidence, where they DISAGREE = the (small) human-review pile. Proven: Gemini reads most of a page and nails many hard names; where it misses (e.g. Gibson, Howard), Claude gets them, and vice versa.
- **Margin-anchored line numbers** (not geometric position). The form prints its own row numbers 1–42; anchoring to those + range validation is robust to a band returning the wrong row count (which used to shift the whole page off-by-one).
- **Exact transcription, not inference.** User is firm on this: models write what's on the page, never expand ("Geo." not "George"). Ditto marks are the one deliberate normalization (→ `[DITTO]`), because we resolve them downstream.
- **Ditto marks stay raw (`[DITTO]`) through transcription**; resolution is a shared downstream step (so Claude and Gemini outputs stay directly comparable for consensus).

## Findings (established, don't re-litigate)

- Full-page downsampling is the core resolution problem; tiling fixes it.
- Pro ≈ Flash on these pages (no accuracy gain, more cost/latency) — use Flash.
- PNG vs JPG, "paleographer" prompt, dropping the schema, self-consistency voting — none crack the systematically-misread hard names. Cross-model consensus does.
- Geometry generalizes across the reel (validated on 0022 Barton + 0450 Owego).
- Claude's "flakiness" was never flakiness: the CLI wraps correct JSON in "Based on my reading: ```json …```" preamble; the parser now extracts JSON regardless. Fixed.
- Gemini drops the thin cursive leading "1" of 3-digit dwelling/family numbers (146→46); prompt nudged to read every digit.
- Current quality: **~87–91% cell agreement** on pilot pages (0022/0023/0450). Remaining disagreements are the genuinely hard cells: abbreviated/hard first names, the faint `attended_school` tick-mark column, real state/age reads.

## Open issues / gotchas

- **Page metadata is unreliable.** The tiled header pass misreads town (called Barton "Paxton"). For the DB, every row needs correct town/county/date/page — needs a dedicated, verified metadata pass. This is the #1 thing to fix before ingestion.
- **Ditto resolution not built.** `[DITTO]` values must be walked-up-and-filled at ingestion.
- `attended_school` and other faint flag columns are genuinely ambiguous — expect them to stay in the review pile; that's correct behavior, not a bug.
- Gemini **free tier = 20 requests/DAY** (not per-minute). Use the paid key for real work.
- Python: macOS PEP 668 requires the project `.venv/` (`google-genai`, `flask`, `pillow`). Don't `pip install` globally.

## Roadmap (agreed direction: depth before breadth)

1. **Reliable page metadata** (dedicated header pass, verified).
2. **Ditto resolution** (walk up each column, fill `[DITTO]` with the real value).
3. **DuckDB ingestion** (step 5) — consensus + resolved → one queryable table (search surnames, filter by town, list households). The payoff: it becomes a real searchable database.
4. **Scale to the full county** (~590 pages) on the proven recipe.

## Operational notes

- Git: `github.com:kraftinator/scriptorium`, branch `main`, SSH. `.gitignore` excludes `data/reels/`, `pages/`, `crops/`, `output/`, `.venv/`, `__pycache__`.
- **Secrets never in the repo.** API keys are passed via env at runtime (`GEMINI_API_KEY`). The user provides the paid Gemini key via the console, not the channel. Always secret-scan before committing.
- Communicate via the telegram MCP `reply` tool — terminal output does NOT reach the user. Use the `chat_id` from the inbound `<channel>` message; it arrives with every message, so it never needs to be stored here.
- Web interface runs on port 5001 (`/` playground, `/view` output), bound `0.0.0.0` via `WEBPLAY_HOST`, and is reachable over the user's Tailscale net (get the current IP with `tailscale status`).
