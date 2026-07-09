"""Model backends for the Scriptorium transcription engine.

Each backend takes the SAME inputs — a page image, the corpus prompt, and the
corpus JSON schema — and returns a parsed JSON dict. Add a new model by writing
one function and registering it in REGISTRY below; the pipeline (transcribe.py)
never changes.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

CLAUDE_MODEL = "claude-opus-4-8"
# Gemini model is pinned (not the -latest alias) so the corpus stays consistent:
# every page is read by a known model. Overridable via env to flip Flash <-> Pro,
# e.g. `GEMINI_MODEL=gemini-pro-latest`. NOTE: Google retires old IDs (the whole
# 2.x Flash line 404s as of 2026-07); bump this deliberately + re-validate a page.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


def _minify(schema: dict) -> str:
    """Schema as a compact string (no whitespace) to save structural tokens."""
    return json.dumps(schema, separators=(",", ":"))


def _strip_fence(text: str) -> str:
    """Extract the JSON object from a model reply, tolerating markdown fences AND
    any conversational preamble/postamble (e.g. Claude's "Based on my reading:
    ```json ... ```"). The JSON object is the span from the first '{' to the
    last '}'; prose around it has no braces.
    """
    text = text.strip()
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        return text[i:j + 1]
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
    last = ""
    for attempt in range(3):  # the CLI occasionally returns empty/partial output
        result = subprocess.run(
            ["claude", "-p", full, "--model", CLAUDE_MODEL,
             "--allowedTools", "Read", "--strict-mcp-config"],
            capture_output=True,
            text=True,
        )
        text = _strip_fence(result.stdout.strip())
        if result.returncode == 0 and text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                last = "invalid JSON from CLI"
        else:
            last = f"exit {result.returncode}, empty={not text}"
        time.sleep(2 ** attempt)
    raise RuntimeError(f"claude CLI failed after 3 attempts: {last}")


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
    from google.genai import errors as genai_errors

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env
    mime = "image/png" if jpg.suffix.lower() == ".png" else "image/jpeg"
    if os.environ.get("GEMINI_NO_SCHEMA"):
        # Free-form JSON: schema is described in the prompt, not natively enforced.
        # Lets the model reason (alphabet audit, candidate weighing) before emitting,
        # instead of being constrained straight into schema-shaped tokens.
        sys_prompt = (prompt + "\n\nConform your output exactly to this JSON "
                      f"schema:\n{_minify(schema)}")
        cfg = types.GenerateContentConfig(
            system_instruction=sys_prompt, temperature=0.0,
            response_mime_type="application/json")
    else:
        cfg = types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=to_gemini_schema(schema),
        )
    contents = [
        types.Part.from_bytes(data=jpg.read_bytes(), mime_type=mime),
        "Transcribe this image according to your system instructions.",
    ]
    for attempt in range(6):  # backoff on transient overload / rate limits
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=cfg)
            break
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            if getattr(e, "code", None) in (429, 500, 503) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("gemini returned empty output")
    return json.loads(_strip_fence(text))


REGISTRY = {
    "claude": claude_backend,
    "gemini": gemini_backend,
}
