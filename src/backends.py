"""Model backends for the Scriptorium transcription engine.

Each backend takes the SAME inputs — a page image, the corpus prompt, and the
corpus JSON schema — and returns a parsed JSON dict. Add a new model by writing
one function and registering it in REGISTRY below; the pipeline (transcribe.py)
never changes.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

CLAUDE_MODEL = "claude-opus-4-8"
GEMINI_MODEL = "gemini-2.5-flash"  # pro is not on the free tier (limit 0); flash has free quota


def _minify(schema: dict) -> str:
    """Schema as a compact string (no whitespace) to save structural tokens."""
    return json.dumps(schema, separators=(",", ":"))


def _strip_fence(text: str) -> str:
    """Remove a stray ```json fence if a model wraps output despite instructions."""
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    return text


def claude_backend(jpg: Path, prompt: str, schema: dict) -> dict:
    """Claude via the `claude` CLI in print mode. No API key (uses your login).

    --strict-mcp-config isolates this child from all MCP servers (including
    channel plugins like Telegram), so it can't contend for a bot's single poll
    connection and drop the channel. Read is built-in, not MCP, so it still works.
    """
    full = (
        f"{prompt}\n\n"
        f"Read the census page image at:\n{jpg}\n\n"
        f"Conform the output exactly to this JSON schema:\n{_minify(schema)}"
    )
    result = subprocess.run(
        ["claude", "-p", full, "--model", CLAUDE_MODEL,
         "--allowedTools", "Read", "--strict-mcp-config"],
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}); empty={not text}; "
            f"stderr:\n{result.stderr.strip()[:2000]}"
        )
    return json.loads(_strip_fence(text))


def to_gemini_schema(node):
    """Derive a Gemini-safe response_schema from our canonical JSON Schema.

    page_schema.json stays the single source of truth. Gemini's response_schema
    doesn't accept the JSON-array union `["string","null"]` or the top-level
    `$schema`/`title` keys, so we normalize a deep copy:
      - drop `$schema` / `title`
      - collapse `type: ["string","null"]` -> `type: "string"` + `nullable: true`
    The nullable form is semantically identical and universally accepted, so
    this works whether or not the SDK also accepts the raw union.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in ("$schema", "title"):
                continue
            if key == "type" and isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                out["type"] = non_null[0] if non_null else "string"
                if "null" in value:
                    out["nullable"] = True
            else:
                out[key] = to_gemini_schema(value)
        return out
    if isinstance(node, list):
        return [to_gemini_schema(item) for item in node]
    return node


def gemini_backend(jpg: Path, prompt: str, schema: dict) -> dict:
    """Gemini via the google-genai SDK, with native response_schema enforcement.

    Requires `pip install google-genai` and a GEMINI_API_KEY (or GOOGLE_API_KEY)
    in the environment. The corpus prompt is the system instruction; the image
    is an inline part; the schema is enforced natively (normalized for Gemini).

    Ditto marks are intentionally left raw ([DITTO]) here — resolution is a
    shared downstream step at DB ingestion, so Claude and Gemini output stay
    directly comparable for cross-model consensus.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=jpg.read_bytes(), mime_type="image/jpeg"),
            "Transcribe this 1850 census page completely according to your "
            "system instructions.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=to_gemini_schema(schema),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("gemini returned empty output")
    return json.loads(_strip_fence(text))


REGISTRY = {
    "claude": claude_backend,
    "gemini": gemini_backend,
}
