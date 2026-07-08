"""Scratch playground for poking at the Gemini API.

Edit the knobs below and run:
    GEMINI_API_KEY=... .venv/bin/python play.py

(Not part of the pipeline — a dev scratchpad. Safe to edit/delete.)
"""
import json
from google import genai
from google.genai import types

# --- knobs ---------------------------------------------------------------
MODEL = "gemini-2.5-flash"          # or "gemini-2.5-pro"
TEMPERATURE = 0.0
IMAGE = None                         # e.g. a page/crop path, or None for text-only
#   IMAGE = "corpora/us_census_1850/data/reels/populationschedu0604unix/populationschedu0604unix_0023.png"
SYSTEM = "You are an expert transcriber of 1850 US Census handwriting."
PROMPT = "Read the names on lines 38 and 39."
JSON_MODE = False                    # True -> ask for application/json
# ------------------------------------------------------------------------

client = genai.Client()              # reads GEMINI_API_KEY from the environment

contents = []
if IMAGE:
    mime = "image/png" if IMAGE.lower().endswith(".png") else "image/jpeg"
    # small files can be inlined; large ones go through the Files API
    import os
    if os.path.getsize(IMAGE) > 15_000_000:
        contents.append(client.files.upload(file=IMAGE))
    else:
        contents.append(types.Part.from_bytes(data=open(IMAGE, "rb").read(), mime_type=mime))
contents.append(PROMPT)

cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=TEMPERATURE)
if JSON_MODE:
    cfg.response_mime_type = "application/json"

resp = client.models.generate_content(model=MODEL, contents=contents, config=cfg)
print(resp.text)
# handy extras:
# print("tokens:", resp.usage_metadata)
