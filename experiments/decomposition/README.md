# Grapheme-decomposition experiment

Scratch proof-of-concept for the **false-agreement** problem (see CLAUDE.md →
Findings). NOT wired into `src/` — these scripts stand alone and import the
existing backends/adjudicator read-only.

## The problem

When BOTH models mis-read the same hard glyph identically, consensus +
adjudication give *false confidence* — they only fire on DISagreement. Pilot
case: L33 surname on frame 0023, where both models read **"Cissen"** though the
paper says **"Gibson"**.

Root cause: the models do holistic *word* recognition, so a top-down language
prior overrides the actual strokes — the "b" ascender is read as an archaic
long-s (→ phantom `ss`), and capital "G" read as "C" reshapes the rest into a
plausible surname.

## The fix (demonstrated here)

1. **`decomp_reread.py`** — crop JUST the name at high zoom and force the model
   to describe each letter's **physical structure** (ascender? does the stroke
   close into a baseline bowl → `b`, or stay open → long-s? descender? are
   adjacent letters the same or different shape?) BEFORE naming the word. This
   suppresses the word-prior. Result on L33: Claude → **"Gibson"** (correct),
   Gemini → **"Ogden"** (wrong, but no longer "Cissen"). The blind spot is no
   longer *shared*, so the false agreement becomes a real disagreement.

2. **`v4_debate_check.py`** — feed those two candidates (Gibson vs Ogden) into
   the **unchanged** `adjudicate_name_v4`. Result: **"Gibson" [MEDIUM, tier1]** —
   once "Gibson" is merely on the ballot, both models re-read the row and
   converge. The blind spot was never inability to read the glyph; it was that
   neither model ever *proposed* the right candidate.

## Run

```bash
.venv/bin/python experiments/decomposition/decomp_reread.py       # defaults = L33
.venv/bin/python experiments/decomposition/v4_debate_check.py     # Gibson vs Ogden
```

## Open (before this becomes real)

- **Which cells trigger decomposition?** Can't run it on every cell of ~590
  pages — need a signal (all surnames? a legibility heuristic?).
- **Wiring:** cleanest fit is a pre-adjudication step that, on flagged cells,
  swaps in the decomposition read so any new disagreement flows into v4. Put it
  behind a flag and validate on the 6-name marquee set before it touches the
  default path.
