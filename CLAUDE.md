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
Steps 2–4 are **built and validated**, including cross-model reconciliation.
Step 5 (DB) is **not built yet**.

Flow: page image → per-agent tiled transcription (Claude *and* Gemini,
independently) → **adjudicate** (resolve name disagreements via a tiered debate)
→ finished reconciled page → (future) ditto resolution → (future) DuckDB.
`src/pipeline.py` chains transcribe(both agents) → adjudicate into one command.

## How to run

The Gemini key now auto-loads from a **gitignored `.env`** at the repo root
(`GEMINI_API_KEY=…`), read by `_load_dotenv()` in `src/backends.py` at import —
so the inline `GEMINI_API_KEY=…` prefix below is OPTIONAL (an inline value still
overrides the file). If a run reports an empty key, the `.env` is missing: it is
not in git and does not survive a machine move, so re-create it (one line).

```bash
# FULL PIPELINE for one page (transcribe claude+gemini tiled -> adjudicate):
.venv/bin/python src/pipeline.py --corpus corpora/us_census_1850 \
    --reel populationschedu0604unix --frame 23

# ...or run the steps by hand:
# 1) transcribe one model, tiled (Claude uses CLI login/no key; Gemini needs the key):
GEMINI_API_KEY=... .venv/bin/python src/transcribe.py --corpus corpora/us_census_1850 \
    --reel populationschedu0604unix --frame 23 --agent gemini --mode tiled
# 2) adjudicate: resolve name disagreements via the tiered debate (calls models):
.venv/bin/python src/adjudicate.py --corpus corpora/us_census_1850 \
    --reel populationschedu0604unix --frame 23 --agents claude gemini --strategy v4
#    --strategy {v1,v2,v3,v4} picks the debate flow (default v1; see adjudicate.py below).
#    (add `--lines 4 7` to adjudicate only those rows, into a .adjudicated.<strategy>.partial.json)
# reconcile: fast, model-free "how much do they disagree" check (NOT in the chain):
.venv/bin/python src/reconcile.py --corpus corpora/us_census_1850 \
    --reel populationschedu0604unix --frame 23 --agents claude gemini

# Web interface (Gemini playground at /, output viewer at /view):
WEBPLAY_HOST=0.0.0.0 .venv/bin/python webplay.py   # port 5001
```

Outputs land in `corpora/us_census_1850/output/rows/<reel>/`:
`<reel>_<frame>.<agent>.json` per model, `<reel>_<frame>.consensus.json` (reconcile),
`<reel>_<frame>.adjudicated.<strategy>.json` (final resolved page; `.partial` when `--lines` used).

## Architecture / file map

- `src/transcribe.py` — CLI entry. `--corpus --reel --frame --agent{claude,gemini} --mode{whole,tiled}`. Converts JP2→PNG via `sips`, dispatches to backend or tiler.
- `src/backends.py` — model adapters. `REGISTRY = {claude, gemini}`. Add a model = add a function + register; the pipeline never changes.
  - `_load_dotenv()`: runs at import; loads `KEY=VALUE` lines from a gitignored `.env` (repo root, then CWD) into `os.environ` for any key not already set. Zero-dependency (no `python-dotenv`). An already-set env var wins, so inline overrides still work.
  - `claude_backend`: shells out to `claude -p ... --model claude-opus-4-8 --allowedTools Read --strict-mcp-config`. No API key (uses login). `--strict-mcp-config` is REQUIRED — it isolates the child from MCP servers so it can't steal the Telegram bot's poll connection. Wrapped in a `CLAUDE_TIMEOUT_S=180` subprocess timeout so a hung CLI call can't wedge the run (a real 1.5h hang cost us a full page).
  - `gemini_backend`: `google-genai` SDK, native `response_schema` (temp 0), retry/backoff, `GEMINI_TIMEOUT_MS=120_000` HttpOptions timeout. `GEMINI_MODEL` env picks the model; **default `gemini-3.5-flash`** — the whole 2.x Flash line (2.0/2.5) was RETIRED by Google (404s as of 2026-07); bump this deliberately and re-validate a page when Google moves again.
  - `_strip_fence`: extracts the JSON object (first `{` to last `}`) — tolerates Claude's conversational preamble + markdown fence (see Findings).
  - `to_gemini_schema`: normalizes our JSON Schema for Gemini (`["string","null"]` → `nullable:true`, drops `$schema`/`title`).
- `src/tile.py` — `transcribe_tiled`: crops the header for metadata, then row-bands. **Line numbers are anchored to the pre-printed margin numbers** the form shows (1–42), validated against each band's expected range — so a stray sliver row is filtered, not shifted. Logs missing lines.
- `src/reconcile.py` — fast, MODEL-FREE merge of the two per-agent files into a consensus. Per cell: agree → value + HIGH; disagree → both readings in `conflicts`. Reports **cell-level agreement %**. `norm()` collapses ditto marks + blanks to one equivalence class. `raw_name` excluded (derived, double-counts). Superseded by `adjudicate.py` for the final answer — now mainly a quick "how much do they disagree" check.
- `src/adjudicate.py` — RESOLVES name disagreements (calls models). Reads the two per-agent files + the page image; for each first/last-name conflict runs one of several selectable **strategies** (`--strategy`, `STRATEGIES` dict pairs a crop fn + an adjudicate fn). Each strategy writes its own output file (`.adjudicated.<strategy>.json`) with a `strategy` field, so nothing is overwritten when comparing. Confidence is always derived from what actually happened (agreement=HIGH, resolved=MEDIUM, deadlock/defaulted=LOW — honest, never overconfident); every resolution keeps both original readings + the winning rationale (auditable). Non-name conflicts → Claude+LOW (`claude-default`). `--lines N…` runs a subset (→ `.partial.json`).
  - **v1** (default) — nudge → if agree, tier1 → else FREE debate (each re-reads and may propose ANY name, short-circuit rebuttal) → deadlock Claude+LOW. Best on the *stability test* for recovering a correct third reading (Mary A. 2/3), but the free re-read can occasionally invent a spliced blend.
  - **v2** — nudge → tier1 → else ANCHORED debate (must pick one of the two candidates) → on deadlock an OPEN persuasion round (any name, always LOW) → Claude+LOW. The open round is what can still emit an invented blend.
  - **v3** — open-read-FIRST (both re-read a tight name-cell crop with NO candidates shown, to avoid anchoring bias) → agree → else persuasion debate → deadlock Claude+LOW. Underperformed (2/5): the tight crop + no-anchor read garbled several cells; set aside.
  - **v4** (current best for correctness) — nudge → tier1 → else ANCHORED debate → on deadlock straight to Claude+LOW. **No open/free round at all**, so the final answer is ALWAYS one of the two real candidate readings — an invented blend like "Noward" (Howard+Norwood) is structurally impossible. Trade-off: cannot recover a correct third reading when BOTH candidates are wrong (Mary A. → stays wrong). On the 6-cell marquee test: 6/7 names correct, Howard finally correct, zero blends.
- `src/pipeline.py` — end-to-end orchestrator: subprocesses `transcribe.py` for each agent (tiled) then `adjudicate.py`. One command → finished reconciled page. `reconcile.py` is NOT in the chain.
- `webplay.py` — Flask dev tool. `/` = Gemini API playground (pick model/temp/prompt, feed a page or a cropped row-band). `/view` = output viewer: consensus as a table, conflict cells highlighted with both models' readings — doubles as the human-review UI.
- `play.py` — CLI Gemini playground (edit knobs, run).
- `corpora/us_census_1850/config/`:
  - `page_schema.json` — the canonical schema (metadata + rows[], nullable strings, confidence_score enum, transcription_notes). Single source of truth.
  - `transcription_prompt.txt` — whole-page prompt. `band_prompt.txt` — per-band prompt. Both enforce: transcribe EXACTLY — no expanding abbreviations ("Geo." not "George") AND no spelling normalization ("Bennit" not "Bennett", "Robbinson" not "Robinson"); read every digit of dwelling/family numbers; the Ditto Rule (any ditto mark in ANY column → literal `[DITTO]`, never the raw `"`, never resolved); `[ILLEGIBLE]` for unreadable; confidence scoring.
  - `layout.json` — tiling geometry: `n_rows=42, row1_top=1620, row_pitch=150, band_rows=6, crop_margin=80`, plus `header_top=0, header_bottom=1050` (the tight TITLE-BAND crop used for metadata). Calibrated on frame 0023; **validated to generalize across the reel** (other pages/towns have different pixel dimensions but the same vertical row grid).

## Key design decisions & why

- **Tiling (row-bands), not whole-page.** Vision APIs downsample a ~6600×8400 page so far that fine strokes blur — full page reads "Snickaback", a row-band reads "Knickerbocker". Slicing restores per-row resolution.
- **Cross-model consensus is the accuracy lever.** No single model (Gemini Flash, Gemini Pro, or Claude) reads 1850 Spencerian script perfectly — the ink is genuinely ambiguous. Design: run two decent models, where they AGREE = high confidence, where they DISAGREE = adjudicated (or the human-review pile). Proven: where ONE model misses a hard name (e.g. Howard), the other often gets it and the debate surfaces the right one. **Known limit (don't forget):** when BOTH models mis-read the same hard glyph the same way — L33 surname, where both read "Cissen" though the paper says "Gibson" — agreement produces FALSE confidence (tier1/MEDIUM) and reconciliation cannot catch it by design (nothing to debate). Needs a separate low-legibility signal, not a debate tweak.
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
- Metadata fixed: reading the header from a tight TITLE-BAND crop (not the full header) makes town/county/date legible and agreed ("Barton"/"Tioga", not "Paxton"). Marshal + census page number are intentionally NOT read (hard + low value); the frame index (0023) is the page key.
- Adjudication debate: showing a model the other reader's specific ARGUMENT (not just its answer) flips confident misreads — Clarinda, Hulett, and (in v4) Howard resolve correctly. But genuinely ambiguous ink (e.g. an ornate middle initial, A vs S) is a coin flip even for the debate, so those are honestly flagged LOW.
- **Invented-blend pitfall — fixed by v4.** A debate that lets a model re-read FREELY (v1's free round, v2's open deadlock round) can splice a wrong third reading and over-trust it: "Nulett" (Hulett/Sulles), and notably **"Noward" = Howard + Norwood**. The user's line: *getting Howard wrong is not acceptable.* Fix = **v4**: no free/open round anywhere, so the answer must be one of the two real candidate readings — the blend is structurally impossible. Verified: L1 → "Howard" [tier2/MEDIUM], no blend. Cost: v4 can't recover a correct THIRD reading (Mary A. stays wrong).
- **False-agreement finding (open).** L33 surname: Claude originally read "Gibson", Gemini "Cissen"; at the nudge BOTH converged to "Cissen" (tier1, MEDIUM), which is what every strategy then output. The user confirms the paper says "Gibson" (3rd letter is a cursive `b`). So agreement overrode a correct read — a distinct failure from the invented blend, and one no debate strategy can fix (they only fire on DISagreement). Fix must be upstream: flag low-legibility glyphs / force an independent re-read even when the two agree.

## Open issues / gotchas

- **False agreement on hard glyphs (unsolved).** When both models mis-read the same ambiguous glyph identically (L33 "Gibson"→both "Cissen"), consensus + adjudication give false confidence — they only act on DISagreement. Needs an upstream low-legibility signal or a forced independent re-read on hard cells, not a debate change. Open design question.
- **Ditto resolution not built.** `[DITTO]` values must be walked-up-and-filled at ingestion. This is the next downstream step.
- **Metadata: reliable per page, but no cross-page smoothing yet.** The title-band crop reads town/county/date reliably now, but a lone page could still misread. A future constancy-smoothing pass (town/county are constant across an enumeration district → stamp the majority reading across the range) would catch outliers at scale.
- `attended_school` and other faint flag columns are genuinely ambiguous — expect them to stay in the review pile; that's correct behavior, not a bug.
- Gemini **free tier = 20 requests/DAY** (not per-minute). Use the paid key for real work.
- Python: macOS PEP 668 requires the project `.venv/` (`google-genai`, `flask`, `pillow`). Don't `pip install` globally.

## Roadmap (agreed direction: depth before breadth)

DONE: title-band metadata, cross-model reconciliation (`adjudicate.py`), end-to-end pipeline (`pipeline.py`), selectable adjudication strategies (v1–v4; **v4 = best correctness**, no invented blends — but code default is still `v1`; pick the default deliberately before scaling), gitignored `.env` key loading, per-call timeouts.

1. **Ditto resolution** (walk up each column, fill `[DITTO]` with the real value) — the next downstream step.
2. **DuckDB ingestion** (step 5) — resolved rows → one queryable table (search surnames, filter by town, list households). The payoff: a real searchable database.
3. **Scale to the full county** (~590 pages) on the proven recipe — parallelize the slow Claude CLI calls, and add the metadata constancy-smoothing pass across town ranges.

## Operational notes

- Git: `github.com:kraftinator/scriptorium`, branch `main`, SSH. `.gitignore` excludes `data/reels/`, `pages/`, `crops/`, `output/`, `.venv/`, `__pycache__`.
- **Secrets never in the repo.** The paid Gemini key lives in a gitignored **`.env`** at the repo root (`GEMINI_API_KEY=…`), auto-loaded by `_load_dotenv()` in `backends.py`. `.gitignore` excludes `.env` + `*.env`; confirm with `git check-ignore .env` before any commit. The user provides the key via the console, not the channel. Always secret-scan before committing. (Note: the key has appeared in cleartext in old Claude Code transcript logs under `~/.claude/projects/…` — rotating it is advisable.)
- Communicate via the telegram MCP `reply` tool — terminal output does NOT reach the user. Use the `chat_id` from the inbound `<channel>` message; it arrives with every message, so it never needs to be stored here.
- Web interface runs on port 5001 (`/` playground, `/view` output), bound `0.0.0.0` via `WEBPLAY_HOST`, and is reachable over the user's Tailscale net (get the current IP with `tailscale status`).
